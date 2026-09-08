import cupy

import numpy as np
from tqdm import tqdm

from utils import Layer, FLOAT_TYPE
from activations import SoftMax
from layers import *
from transformer_adapters import *


class CrossEntropy:

    """Fused softmax + cross entropy.

    gradients() returns d(loss)/d(the softmax layer's input), which is why
    SoftMax(fused_loss = True) passes its gradient straight through.

    Neither method builds a one hot. The identity matrix this used to hold is
    num_classes squared - 256 GiB at Gemma's 262144 token vocabulary - and indexing
    it expanded to a full (batch, sequence, vocab) array on every call. Each label
    selects exactly one entry per row, so both operations are a gather instead.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.positions = {}

    def _index(self, labels):

        """Index tuple addressing the entry each label selects.

        Broadcastable ranges over the leading axes, then the labels themselves, so
        this works for (batch,) classification labels and (batch, sequence) token
        labels alike. Cached per shape; batches only vary on the final partial one.
        """

        shape = labels.shape

        if shape not in self.positions:
            self.positions[shape] = tuple(
                cupy.arange(size).reshape((-1,) + (1,) * (len(shape) - axis - 1))
                for axis, size in enumerate(shape))

        return self.positions[shape] + (labels,)

    def gradients(self, logits, labels):
        # softmax output minus the one hot, without the one hot. Every (row, label)
        # pair is distinct, so the subtraction needs no scattered accumulation.
        gradient = logits.copy()
        gradient[self._index(labels)] -= 1
        return gradient

    def loss(self, logits, labels):
        eta = 1e-7
        return -1 * cupy.sum(cupy.log(logits[self._index(labels)] + eta))


class Network:

    def __init__(self, layers:list[Layer]):
        self.layers = layers
        self.rng    = cupy.random.default_rng() # generation only; training draws nothing here

        # a fused SoftMax returns its gradient unchanged, which is only right when
        # CrossEntropy produced that gradient for the last layer of the network
        for position, layer in enumerate(layers[:-1]):
            if isinstance(layer, SoftMax) and layer.fused_loss:
                raise ValueError(
                    f"SoftMax at position {position} of {len(layers)} has fused_loss = True but is "
                    "not the final layer. Its backward pass returns the gradient unchanged, which "
                    "is only correct when CrossEntropy supplied it. Use SoftMax(fused_loss = False) "
                    "to apply the real Jacobian mid network.")
        
    def predict(self, input):
        return self._forward(input)

    def predict_logits(self, input):
        if not isinstance(self.layers[-1], SoftMax):
            raise ValueError("predict_logits expects a final SoftMax")
        return self._forward(input, logits=True)

    def _forward(self, input, logits=False):
        for layer in (self.layers[:-1] if logits else self.layers):
            input = layer.forward(input)
            if layer.inference_only:
                # the next layer already holds what it needs, so drop this one's
                # activations now rather than pinning all of them until the forward ends
                layer.clear_cache()
        return input

    def _backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def _start_cache(self, batch_size, max_length, step):
        for layer in self.layers:
            layer.start_cache(batch_size, max_length, step)

    def _stop_cache(self):
        for layer in self.layers:
            layer.stop_cache()

    def _sample(self, probabilities, top_k = None):

        """Draw one token per row from the final position's distribution.

        probabilities is (batch, 1, vocab): GPTEmbedBack slices to the last position
        while a cache is active, so this is already the only distribution generation
        needs. Temperature was applied by the SoftMax layer itself on the way out.
        """

        probabilities = probabilities[:, -1, :]

        if top_k is not None and top_k < probabilities.shape[-1]:
            # keep the k largest and renormalize. partition puts the k largest last, so
            # entry -top_k is the threshold every survivor has to meet
            threshold     = cupy.partition(probabilities, -top_k, axis = -1)[:, -top_k, None]
            probabilities = cupy.where(probabilities >= threshold, probabilities, 0.0)
            probabilities = probabilities / cupy.sum(probabilities, axis = -1, keepdims = True)

        # inverse transform sampling, vectorized over the batch. A vocab sized compare
        # and reduce is free next to the unembedding matmul that just produced these.
        # Scaling the draw by the final cumulative entry rather than trusting it to be
        # exactly 1 keeps float error from ever running off the end of the row
        cumulative = cupy.cumsum(probabilities, axis = -1)
        draw = self.rng.random((probabilities.shape[0], 1), dtype = FLOAT_TYPE)

        return cupy.sum(cumulative < draw * cumulative[:, -1, None], axis = -1, keepdims = True)

    def generate(self, tokens, max_new_tokens, temperature = 1.0, top_k = None,
                       step = 256, stop = None, on_event = None):

        """Sample a continuation of `tokens`, decoding incrementally with a KV cache.

        tokens is (batch, prompt) of ids, or a single (prompt,) sequence; every row has
        to be the same length, since one cache position is shared by the whole batch.
        Returns the (batch, generated) ids, without the prompt.

        The prompt is run in chunks of `step` and then tokens come out one at a time,
        each forward costing one token of work instead of re-reading the whole sequence.
        For a 1024 token prompt and 256 sampled tokens that is ~1280 token-forwards
        against the ~295000 an uncached loop would do.

        `step` is what bounds a sliding layer's store, at window + step, so it trades
        cache headroom against how often the store compacts. It composes with each
        attention layer's own chunk_size, which independently bounds the score matrix:
        chunk_size smaller than step simply runs several blocks per forward.

        temperature=0 selects greedy decoding; otherwise sampling uses temperature
        and top_k. stop accepts one or several token IDs. Finished batch rows are
        filled with the first stop ID while the other rows finish. on_event, if given,
        receives prefill_start, prefill_end, and generation_end for instrumentation.
        Temperature, eval flags, and cache state are restored on exit.
        """

        if not isinstance(self.layers[-1], SoftMax):
            raise ValueError("generate() samples from a probability distribution, so the network "
                             "has to end in SoftMax")

        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError("step must be a positive integer")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative (0 means greedy)")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1):
            raise ValueError("top_k must be a positive integer or None")
        tokens = cupy.asarray(tokens)
        if tokens.ndim == 1:
            tokens = tokens[cupy.newaxis, :]
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("tokens must be a nonempty (batch, prompt) integer array")
        if tokens.dtype.kind not in 'iu':
            raise ValueError("tokens must contain integer IDs")
        batch_size, prompt_length = tokens.shape
        if isinstance(self.layers[0], GPTEmbedFront):
            vocab = self.layers[0].table.shape[0]
            if bool(cupy.any(tokens < 0)) or bool(cupy.any(tokens >= vocab)):
                raise ValueError("token ID outside the vocabulary")
        limit = getattr(self, 'context_length', None)
        if limit is not None and prompt_length + max_new_tokens > limit:
            raise ValueError(f"prompt plus generation exceeds context_length={limit}")
        stop_ids = [] if stop is None else ([stop] if np.isscalar(stop) else list(stop))
        if any(not isinstance(t, (int, np.integer)) or isinstance(t, (bool, np.bool_)) or t < 0
               for t in stop_ids):
            raise ValueError("stop must contain nonnegative integer token IDs")
        if isinstance(self.layers[0], GPTEmbedFront) and any(t >= vocab for t in stop_ids):
            raise ValueError("stop token ID outside the vocabulary")
        if max_new_tokens == 0:
            return tokens[:, :0]

        # Preserve eval flags even on nested/composite layers and on exceptions.
        states, seen = [], set()
        def remember(layer):
            if id(layer) in seen:
                return
            seen.add(id(layer))
            states.append((layer, layer.eval_mode))
            for value in vars(layer).values():
                children = value if isinstance(value, (list, tuple)) else (value,)
                for child in children:
                    if isinstance(child, Layer):
                        remember(child)
        for layer in self.layers:
            remember(layer)
        self.set_eval(True)
        softmax = self.layers[-1]
        previous_temperature = softmax.temperature
        softmax.temperature = temperature if temperature > 0 else 1.0
        generated = []
        finished = cupy.zeros((batch_size, 1), dtype=bool)
        try:
            # Cache allocation must also be covered by cleanup on failure.
            self._start_cache(batch_size, prompt_length + max_new_tokens, step)
            if on_event is not None:
                on_event('prefill_start')
            for i in range(0, prompt_length, step):
                probabilities = self._forward(tokens[:, i:i + step])
            if on_event is not None:
                on_event('prefill_end')
            for i in range(max_new_tokens):
                sampled = (cupy.argmax(probabilities[:, -1, :], axis=-1)[:, None]
                           if temperature == 0 else self._sample(probabilities, top_k))
                if stop_ids:
                    # Finished rows remain stopped while other rows continue.
                    sampled = cupy.where(finished, stop_ids[0], sampled)
                    finished |= cupy.isin(sampled, cupy.asarray(stop_ids))
                generated.append(sampled)
                if (stop_ids and bool(cupy.all(finished))) or i + 1 == max_new_tokens:
                    break
                probabilities = self._forward(sampled)
            result = cupy.concatenate(generated, axis=1)
            if on_event is not None:
                on_event('generation_end')
            return result
        finally:
            softmax.temperature = previous_temperature
            self._stop_cache()
            for layer, flag in states:
                layer.eval_mode = flag


    def set_eval(self, eval_mode):
        for layer in self.layers:
            layer.set_eval(eval_mode)
            layer.zero_grad()
            
    def _zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
    
    def _zero_adam(self):
        for layer in self.layers:
            layer.zero_adam()

    def _update(self, learning_rate, weight_decay, t, num_samples):

        beta1, beta2 = 0.9, 0.999

        for layer in self.layers:
            
            for param, grad, moment, variance in zip(layer.parameters,
                                                     layer.gradients,
                                                     layer.moments,
                                                     layer.variances):
                grad /= num_samples
                lmda = weight_decay
                
                if len(param.shape) == 1 or isinstance(layer, (VitProjector, VitMLPHead, 
                                                               GPTEmbedFront, GPTEmbedBack)):
                    lmda = 0.0
                    
                moment *= beta1
                moment += (1 - beta1)*grad

                # grad is scratch from here on: it is zeroed immediately after the
                # step, so squaring and scaling it in place saves two full sized
                # temporaries per parameter. Bitwise identical to (1 - beta2)*grad**2.
                grad *= grad
                grad *= (1 - beta2)

                variance *= beta2
                variance += grad

                mom_hat = moment / (1 - beta1**t)
                var_hat = variance / (1 - beta2**t)

                param -= learning_rate * (mom_hat / (var_hat**0.5 + 1e-7) + lmda * param)

    def train(self, criterion, train_data, train_labels, test_data = None, test_labels = None,
                    augments = None, epochs = 1, batch_size = 64, batches_per_step = 1,
                    learning_rate = 0.001, weight_decay = 0.01):

        if any(layer.inference_only for layer in self.layers):
            raise RuntimeError("this model was built inside inference_mode(): it has no gradient, "
                               "moment or variance buffers to train with. Rebuild it outside the "
                               "context to train.")

        step_count = 0
        samples_per_step = batch_size * batches_per_step
        
        self._zero_adam()
        
        for i in range(epochs):
            
            self.set_eval(False)

            batches_seen = 0
            train_loss, train_correct = 0, 0

            shuffle = np.random.permutation(len(train_labels))
            train_data   = train_data[shuffle]
            train_labels = train_labels[shuffle]

            for j in tqdm(range(0, len(train_data), batch_size)):
                
                x = train_data  [j: min(j + batch_size, len(train_data))]
                y = train_labels[j: min(j + batch_size, len(train_data))]
                
                if augments is not None:
                    x = augments(x)

                x = cupy.array(x)
                y = cupy.array(y)
                
                y_hat = self._forward(x)
                
                grad = criterion.gradients(y_hat, y)
                self._backward(grad)
                
                batches_seen += 1
                if batches_seen % batches_per_step == 0:
                    step_count += 1
                    self._update(learning_rate, weight_decay, step_count, samples_per_step)
                    self._zero_grad()

                train_loss += criterion.loss(y_hat, y)
                train_correct += cupy.equal(cupy.argmax(y_hat, axis = -1), y).astype(cupy.int32).sum()
                
            train_loss     = train_loss    / np.prod(train_labels.shape)
            train_accuracy = train_correct / np.prod(train_labels.shape)
            print("epoch:", i + 1, "train loss:", train_loss, "train accuracy:", train_accuracy)
            
            if test_data is not None and test_labels is not None:
                test_loss, test_accuracy = self.evaluate(test_data, test_labels, criterion, batch_size=batch_size)
                print("epoch:", i + 1, "test loss:", test_loss, "test accuracy:", test_accuracy, "\n")

    def evaluate(self, test_data, test_labels, criterion, batch_size = 64):
        
        self.set_eval(True)
        loss, correct = 0, 0

        for i in tqdm(range(0, len(test_data), batch_size)):

            x = cupy.array(test_data  [i: min(i + batch_size, len(test_data))])
            y = cupy.array(test_labels[i: min(i + batch_size, len(test_data))])

            y_hat = self.predict(x)

            loss += criterion.loss(y_hat, y)
            correct += cupy.equal(cupy.argmax(y_hat, axis = -1), y).astype(cupy.int32).sum()

        return loss / np.prod(test_labels.shape), correct / np.prod(test_labels.shape)