import cupy

from utils import Layer, FLOAT_TYPE, init_random_tensor, init_zeros_tensor, init_weight_tensor

class Convolution(Layer):

    CACHED = ("input", "output", "padded_input")

    def __init__(self, input_dim: tuple, num_kernels:int, kernel_size: tuple, padding = (0, 0)):
        super(Convolution, self).__init__()

        self.kernel_size =  kernel_size
        self.output_dims = (input_dim[1] + 2*padding[0] + 1 - self.kernel_size[0],
                            input_dim[2] + 2*padding[1] + 1 - self.kernel_size[1])
        
        self.pad_h, self.pad_w = padding
        self.padded_input = None

        self.weights = init_random_tensor((num_kernels, input_dim[0], kernel_size[0], kernel_size[1]))
        self.weights = self.weights / (input_dim[0] * kernel_size[0] * kernel_size[1])**0.5
        self.bias    = init_zeros_tensor(num_kernels).reshape(1, num_kernels, 1, 1)
        
        self.weight_grads = self.register(self.weights)
        self.bias_grads   = self.register(self.bias)

    def forward(self, input):

        self.input = input
        
        self.padded_input = cupy.pad(self.input, ((0, 0), (0, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w)))

        self.output = cupy.lib.stride_tricks.sliding_window_view(self.padded_input, self.kernel_size, (2, 3))
        self.output = cupy.einsum("nchwkl,ockl->nohw", self.output, self.weights)
        self.output = self.output + self.bias

        return self.output

    def backward(self, gradient):

        conv_in = cupy.lib.stride_tricks.sliding_window_view(self.padded_input, self.output_dims, (2, 3))
        self.weight_grads += cupy.einsum("ncklhw,nohw->ockl", conv_in, gradient)

        self.bias_grads += cupy.sum(gradient, axis = (0, 2, 3), keepdims = True)

        grad_pad_h = self.kernel_size[0] - 1
        grad_pad_w = self.kernel_size[1] - 1

        flipped_weights = cupy.flip(self.weights, axis=(2, 3))

        gradient = cupy.pad(gradient, ((0, 0), (0, 0), (grad_pad_h, grad_pad_h), (grad_pad_w, grad_pad_w)))
        gradient = cupy.lib.stride_tricks.sliding_window_view(gradient, self.kernel_size, (2, 3))
        gradient = cupy.einsum("nohwkl,ockl->nchw", gradient, flipped_weights)
        gradient = gradient[:, :, self.pad_h:gradient.shape[2]-self.pad_h, self.pad_w:gradient.shape[3]-self.pad_w]
        
        return gradient

class BatchNorm(Layer):
    CACHED = ("input", "output", "mean", "var", "std", "centered", "normed")

    def __init__(self, num_channels):
        super(BatchNorm, self).__init__()
        
        self.channels = num_channels
        self.eps = 1e-5
        
        self.momentum = 0.1
        
        self.running_mean = init_zeros_tensor((1, num_channels, 1, 1))
        self.running_var  = init_zeros_tensor((1, num_channels, 1, 1))

        self.gamma = init_zeros_tensor((1, num_channels, 1, 1)) + 1
        self.beta  = init_zeros_tensor((1, num_channels, 1, 1))

        self.mean = None
        self.var = None
        self.std = None
        self.centered = None
        self.normed = None

        self.gamma_grads = self.register(self.gamma)
        self.beta_grads  = self.register(self.beta)
    
    def forward(self, input):
        
        self.input = input
        
        if self.eval_mode is False:
            self.mean = cupy.mean(input, axis = (0, 2, 3), keepdims = True)
            self.var  = cupy.var(input, axis = (0, 2, 3), keepdims = True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * self.var
        else:
            self.mean = self.running_mean
            self.var = self.running_var
            
        self.centered = self.input - self.mean
        
        self.std = (self.var + self.eps)**0.5
        self.normed = self.centered / self.std
        
        self.output = self.gamma * self.normed + self.beta
        return self.output
            
    def backward(self, gradient):
        
        self.gamma_grads += cupy.sum(gradient * self.normed, axis = (0, 2, 3), keepdims = True)
        self.beta_grads  += cupy.sum(gradient,               axis = (0, 2, 3), keepdims = True)
        
        gradient = gradient * self.gamma
        
        gradient_normed = (gradient - cupy.mean(gradient, axis = (0, 2, 3), keepdims = True)) / self.std
        
        return gradient_normed - self.centered * (cupy.mean(gradient * self.centered, axis = (0, 2, 3), keepdims = True) / self.std**3)
    

class MaxPool(Layer):

    CACHED = ("input", "output", "mask")

    def __init__(self, pool_height, pool_width):
        super(MaxPool, self).__init__()

        self.pool_h, self.pool_w = (pool_height, pool_width)
        self.batch_size, self.channels, self.in_h, self.in_w = 0,0,0,0
        self.mask = None

    def forward(self, input):

        self.input = input
        self.batch_size, self.channels, self.in_h, self.in_w = self.input.shape

        out_h = self.in_h // self.pool_h
        out_w = self.in_w // self.pool_w

        view = self.input[:, :, :out_h*self.pool_h, :out_w*self.pool_w]
        view = view.reshape(self.batch_size, self.channels, out_h, self.pool_h, out_w, self.pool_w)
        
        self.output = view.max(axis=(3, 5), keepdims = True)

        self.mask   = self.output == view
        self.output = self.output.reshape(self.batch_size, self.channels, out_h, out_w)

        return self.output

    def backward(self, gradient):
        gradient = gradient[:, :, :, cupy.newaxis, :, cupy.newaxis]
        gradient = gradient * self.mask
        gradient = gradient.reshape(self.batch_size, self.channels, self.in_h, self.in_w)
        return gradient
    

class AveragePool(Layer):

    CACHED = ("input", "output")

    def __init__(self, pool_height, pool_width):
        super(AveragePool, self).__init__()

        self.pool_h, self.pool_w = (pool_height, pool_width)
        self.batch_size, self.channels, self.in_h, self.in_w = 0,0,0,0
        self.view_shape = None

    def forward(self, input):

        self.input = input
        self.batch_size, self.channels, self.in_h, self.in_w = self.input.shape

        out_h = self.in_h // self.pool_h
        out_w = self.in_w // self.pool_w

        view = self.input[:, :, :out_h*self.pool_h, :out_w*self.pool_w]
        view = view.reshape(self.batch_size, self.channels, out_h, self.pool_h, out_w, self.pool_w)
        
        self.output = view.mean(axis=(3, 5), keepdims = True)

        self.view_shape = view.shape
        self.output = self.output.reshape(self.batch_size, self.channels, out_h, out_w)
        return self.output

    def backward(self, gradient):
        # every input in a window contributed equally, so the gradient spreads evenly.
        # This used to multiply by a full input sized array holding one constant, which
        # also promoted the gradient to float64: cupy.ones() has no dtype by default.
        gradient = gradient[:, :, :, cupy.newaxis, :, cupy.newaxis] / (self.pool_h * self.pool_w)
        gradient = cupy.broadcast_to(gradient, self.view_shape).copy()
        return gradient.reshape(self.batch_size, self.channels, self.in_h, self.in_w)

class Flatten(Layer):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        self.input  = input
        self.output = self.input.reshape((self.input.shape[0], -1))
        return self.output

    def backward(self, gradient):
        return gradient.reshape(self.input.shape)
    
    
class Dense(Layer):

    def __init__(self, input_size, output_size):
        super(Dense, self).__init__()

        self.weights = init_weight_tensor((input_size, output_size), input_size**0.5)
        self.bias    = init_zeros_tensor(output_size)

        self.weight_grads = self.register(self.weights)
        self.bias_grads   = self.register(self.bias)

    def forward(self, input):
        self.input  = input
        self.output = input @ self.weights + self.bias
        return self.output

    def backward(self, gradient):
        self.weight_grads += self.input.transpose() @ gradient
        self.bias_grads   += cupy.sum(gradient, axis = 0)
        return gradient @ self.weights.transpose()

class Dropout(Layer):
    CACHED = ("input", "output", "dropout_neurons")

    def __init__(self, dropout_rate):
        super(Dropout, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.dropout_rng  = cupy.random.default_rng()
        self.dropout_neurons = None
    
    def forward(self, input):
        self.input = input
        if self.dropout_rate > 0.0 and self.eval_mode is False:
            self.dropout_neurons = self.dropout_rng.random(self.input.shape, dtype = FLOAT_TYPE) >= self.dropout_rate
            self.dropout_neurons = self.dropout_neurons / FLOAT_TYPE(1 - self.dropout_rate)
            self.output = self.input * self.dropout_neurons
        else:
            self.output = self.input
        return self.output
        
    def backward(self, gradient):
        if self.dropout_rate > 0.0 and self.eval_mode is False:
            gradient *= self.dropout_neurons
        return gradient

# Causal masks are cached by block geometry, not by sequence length, and shared across
# every attention layer. A sliding window model has only a handful of distinct block
# geometries no matter how long the sequence is, so this never grows with context.
_BLOCK_MASKS = {}

class RMSNorm(Layer):

    """Root mean square normalization, the norm Gemma style transformers use.

    Unlike LayerNorm there is no mean subtraction and no beta: the vector is only
    rescaled. Written shape agnostically over the last axis so the same layer serves
    both (batch, sequence, channels) activations and the (batch, kv_heads, group,
    sequence, head_dim) query/key tensors that QK norm operates on.

    Gemma 4 checkpoints store gamma directly; no offset is added when loading.
    """

    CACHED = ("input", "output", "normed", "rms")

    def __init__(self, num_channels, eps = 1e-6, with_scale = True):
        super(RMSNorm, self).__init__()

        self.channels = num_channels
        self.eps = eps

        self.gamma = init_zeros_tensor(num_channels) + 1 if with_scale else 1.0

        self.rms = None
        self.normed = None

        self.gamma_grads = self.register(self.gamma) if with_scale else None

    def forward(self, input):

        self.input = input

        self.rms    = (cupy.mean(input * input, axis = -1, keepdims = True) + self.eps)**0.5
        self.normed = input / self.rms

        self.output = self.gamma * self.normed
        return self.output

    def backward(self, gradient):

        # match the library convention of summing weight grads over the batch and
        # averaging over every axis between batch and channels
        axes = tuple(range(gradient.ndim - 1))
        span = 1
        for dim in gradient.shape[1:-1]:
            span *= dim

        if self.gamma_grads is not None:
            self.gamma_grads += cupy.sum(gradient * self.normed, axis = axes) / span

        gradient = gradient * self.gamma
        return (gradient - self.normed * cupy.mean(gradient * self.normed, axis = -1, keepdims = True)) / self.rms

class RotaryEmbedding:

    """Rotary position embeddings, applied to the head split query and key tensors.

    By default partial_rotary_factor rotates a contiguous prefix. proportional=True
    selects Gemma 4's convention: frequencies use the full head width, active pairs
    span the two head halves, and the remaining pairs have zero frequency. Gemma 4
    uses theta 10000 with full rotation locally, and theta 1e6 with factor 0.25 globally.

    This holds no parameters, so it is a plain helper rather than a Layer.
    """

    def __init__(self, head_dim, context_length, theta = 10000.0, partial_rotary_factor = 1.0, proportional = False):

        self.head_dim   = head_dim
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - self.rotary_dim % 2

        # a factor small enough to round the rotary width to zero would silently strip
        # all positional information rather than partially applying it
        assert self.rotary_dim >= 2, (f"partial_rotary_factor {partial_rotary_factor} leaves "
                                      f"no rotary dimensions for head_dim {head_dim}")

        half     = self.rotary_dim // 2
        inv_freq = 1.0 / (theta ** (cupy.arange(half, dtype = FLOAT_TYPE) * 2.0 / self.rotary_dim))

        if proportional:
            # Frequencies use the full head width; active pairs straddle its halves.
            inv_freq = 1.0 / (theta ** (cupy.arange(half, dtype = FLOAT_TYPE) * 2.0 / head_dim))
            inv_freq = cupy.concatenate((inv_freq, init_zeros_tensor(head_dim // 2 - half)))
            self.rotary_dim = head_dim

        angles = cupy.arange(context_length, dtype = FLOAT_TYPE)[:, None] * inv_freq[None, :]
        angles = cupy.concatenate((angles, angles), axis = -1)

        self.cos = cupy.cos(angles).astype(FLOAT_TYPE, copy = False)
        self.sin = cupy.sin(angles).astype(FLOAT_TYPE, copy = False)

    @staticmethod
    def _rotate_half(x):
        half = x.shape[-1] // 2
        return cupy.concatenate((-x[..., half:], x[..., :half]), axis = -1)

    def rotate(self, x, offset = 0):

        length = x.shape[-2]
        cos = self.cos[offset : offset + length]
        sin = self.sin[offset : offset + length]

        if self.rotary_dim == self.head_dim:
            return x * cos + self._rotate_half(x) * sin

        rotated = x[..., :self.rotary_dim] * cos + self._rotate_half(x[..., :self.rotary_dim]) * sin
        return cupy.concatenate((rotated, x[..., self.rotary_dim:]), axis = -1)

    def backward(self, gradient, offset = 0):

        # the rotation is orthogonal, so the reverse pass is the transposed rotation
        length = gradient.shape[-2]
        cos = self.cos[offset : offset + length]
        sin = self.sin[offset : offset + length]

        if self.rotary_dim == self.head_dim:
            return gradient * cos - self._rotate_half(gradient) * sin

        rotated = gradient[..., :self.rotary_dim] * cos - self._rotate_half(gradient[..., :self.rotary_dim]) * sin
        return cupy.concatenate((rotated, gradient[..., self.rotary_dim:]), axis = -1)

class MultiHeadAttention(Layer):

    """Multi head attention, optionally grouped query (GQA).

    Heads are laid out as (batch, kv_heads, group, sequence, head_dim) so that a
    single set of key/value heads broadcasts across the query heads that share it,
    with no materialized repeat. Plain MHA is the group = 1 case.

    num_kv_heads / head_dim / kv_shared all default to the plain MHA behaviour, in
    which case the query, key and value projections stay fused into one matrix and
    the parameter list is unchanged, so existing checkpoints still load.

    kv_shared is Gemma 4's attention_k_eq_v: global layers project once and use the
    result as both key and value. The value branches off before q/k norm and RoPE,
    then optionally receives its own unweighted RMSNorm via v_norm=True.
    attention_scale=None preserves inverse-square-root scaling; Gemma 4 uses 1.0.
    """

    CACHED = ("input", "output", "query", "key", "value", "softmax", "chunks", "heads_out")

    def __init__(self, embedding_dim, context_length, num_heads, decoder = False,
                       num_kv_heads = None, head_dim = None, qk_norm = False,
                       rope = None, sliding_window = None, kv_shared = False,
                       chunk_size = None, attention_scale = None, v_norm = False):
        super(MultiHeadAttention, self).__init__()

        self.embedding_dim  = embedding_dim
        self.context_length = context_length

        self.num_heads    = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.heads_dim    = embedding_dim // num_heads if head_dim is None else head_dim

        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"
        self.groups = self.num_heads // self.num_kv_heads
        self.attention_scale = attention_scale

        self.query_dim = self.num_heads    * self.heads_dim
        self.kv_dim    = self.num_kv_heads * self.heads_dim

        self.kv_shared = kv_shared
        self.rope      = rope

        # keep the fused projection (and therefore the parameter layout) whenever the
        # caller asked for nothing that would change its shape
        self.fused = num_kv_heads is None and head_dim is None and not kv_shared

        self.sliding_window = sliding_window

        # queries are processed in blocks of chunk_size. None means one block, which
        # is exactly the unchunked computation. Setting it bounds peak memory by the
        # block size instead of the sequence length, and for a sliding window layer it
        # also skips computing the scores that the mask would only throw away.
        assert chunk_size is None or chunk_size >= 1, "chunk_size must be positive or None"
        self.chunk_size = chunk_size

        self.query = None
        self.key   = None
        self.value = None

        self.softmax = None   # list of per block attention weights
        self.chunks  = None   # list of (q0, q1, k0, k1) bounds matching self.softmax
        self.heads_out = None

        self.is_decoder = decoder

        self.q_norm = RMSNorm(self.heads_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.heads_dim) if qk_norm else None
        self.v_norm = RMSNorm(self.heads_dim, with_scale = False) if v_norm else None

        if self.fused:
            self.qkv_weights = init_weight_tensor((embedding_dim, 3*embedding_dim), embedding_dim**0.5)
            self.qkv_weight_grads = self.register(self.qkv_weights)
        else:
            self.q_weights = init_weight_tensor((embedding_dim, self.query_dim), embedding_dim**0.5)
            self.k_weights = init_weight_tensor((embedding_dim, self.kv_dim), embedding_dim**0.5)

            self.q_weight_grads = self.register(self.q_weights)
            self.k_weight_grads = self.register(self.k_weights)

            if not self.kv_shared:
                self.v_weights = init_weight_tensor((embedding_dim, self.kv_dim), embedding_dim**0.5)
                self.v_weight_grads = self.register(self.v_weights)

        self.out_weights = init_weight_tensor((self.query_dim, embedding_dim), self.query_dim**0.5)
        self.out_weight_grads = self.register(self.out_weights)

        # the q/k norms own their own parameters; surface them through this layer so
        # the optimizer sees one flat set of parallel lists
        for norm in (self.q_norm, self.k_norm, self.v_norm):
            if norm is not None:
                self.parameters += norm.parameters
                self.gradients  += norm.gradients
                self.moments    += norm.moments
                self.variances  += norm.variances

    def _key_range(self, q0, q1, start, end):

        """Keys that queries [q0, q1) are allowed to see, in absolute positions.

        start is the oldest key still held - 0 without a cache, and the cache's base
        once a sliding window has evicted past it. Uncached, start = 0 and end = T,
        which is exactly the plain causal/sliding behaviour.
        """

        if not self.is_decoder:
            return start, end
        if self.sliding_window is None:
            return start, q1
        return max(start, q0 - self.sliding_window + 1), q1

    def _block_mask(self, q0, q1, k0, k1):

        """Additive mask for one query block, covering only the columns where it bites.

        Returns (mask, column offset). A layer with no sliding window spans keys
        [0, q1), and every key before the block is visible to every query in it, so the
        mask is only non-trivial over the diagonal corner - returning that corner keeps
        the cache at O(chunk_size^2) instead of growing with the sequence length. A
        sliding window block is non-trivial throughout, but is bounded by the window.

        A mask of None means every key in the block is visible and there is nothing to
        add. That is not a rare case: during single token decoding it is *every* block,
        on sliding and global layers alike. A sliding cache holds precisely the window,
        so all of it is in range, and a global block reduces to an all visible 1x1
        corner - so the hot path of generation adds no mask at all.

        The signature is the block's geometry, not its position, so interior blocks all
        share one entry and every layer shares the same cache.
        """

        offset = 0 if self.sliding_window is not None else q0 - k0

        rows_ahead = q0 - k0 - offset
        width      = k1 - k0 - offset
        signature  = (rows_ahead, q1 - q0, width, self.sliding_window)

        if signature not in _BLOCK_MASKS:
            rows = cupy.arange(q1 - q0)[:, None] + rows_ahead
            cols = cupy.arange(width)[None, :]

            allowed = cols <= rows
            if self.sliding_window is not None:
                allowed = allowed & ((rows - cols) < self.sliding_window)

            _BLOCK_MASKS[signature] = (None if bool(allowed.all()) else
                                       cupy.where(allowed, 0.0, -1e9).astype(FLOAT_TYPE, copy = False))

        return _BLOCK_MASKS[signature], offset

    def clear_cache(self):
        super(MultiHeadAttention, self).clear_cache()
        for norm in (self.q_norm, self.k_norm, self.v_norm):
            if norm is not None:
                norm.clear_cache()

    def start_cache(self, batch_size, max_length, step = 1):

        """Reserve the key/value store this layer needs to decode incrementally.

        A sliding layer's store is bounded by its window rather than by max_length,
        which is what keeps a long context affordable: at Gemma's 262144 the 31B's
        sliding layers hold 1.95 GiB between them instead of 440 GiB.

        step is the largest number of tokens a single forward will add. It has to be
        the prefill chunk size, since the store must hold the window plus everything
        one forward appends on top of it.
        """

        super(MultiHeadAttention, self).start_cache(batch_size, max_length, step)

        self.cache.allocate(batch_size, self.num_kv_heads, self.heads_dim,
                            max_length, self.sliding_window, step)

    def _split_heads(self, x, num_heads, groups):
        batch, length = x.shape[0], x.shape[1]
        x = x.reshape((batch, length, num_heads, groups, self.heads_dim))
        return x.transpose(0, 2, 3, 1, 4)

    def _merge_heads(self, x):
        batch, length = x.shape[0], x.shape[3]
        x = x.transpose(0, 3, 1, 2, 4)
        return x.reshape((batch, length, x.shape[2] * x.shape[3] * self.heads_dim))

    def forward(self, input):

        self.input = input
        B, T, C    = self.input.shape

        if self.fused:
            qkv = self.input @ self.qkv_weights
            query, key, value = cupy.split(qkv, 3, axis = 2)
        else:
            query = self.input @ self.q_weights
            key   = self.input @ self.k_weights
            value = key if self.kv_shared else self.input @ self.v_weights

        query = self._split_heads(query, self.num_kv_heads, self.groups)
        key   = self._split_heads(key,   self.num_kv_heads, 1)
        self.value = self._split_heads(value, self.num_kv_heads, 1)

        if self.v_norm is not None:
            self.value = self.v_norm.forward(self.value)

        if self.q_norm is not None:
            query = self.q_norm.forward(query)
            key   = self.k_norm.forward(key)

        # position of the first new token. Without a cache a forward always starts the
        # sequence at zero; with one, it continues where the last forward stopped
        position = self.cache.position if self.cache is not None else 0

        if self.rope is not None:
            query = self.rope.rotate(query, position)
            key   = self.rope.rotate(key,   position)

        if self.cache is not None:
            # keys and values go into the store already normed and rotated: both depend
            # only on the token and its position, so neither ever needs redoing
            key, self.value = self.cache.append(key, self.value)
            base = self.cache.base
        else:
            base = 0

        self.query, self.key = query, key
        end = position + T

        step = self.chunk_size or T
        self.softmax, self.chunks, outputs = [], [], []

        for q0 in range(0, T, step):

            q1     = min(q0 + step, T)
            a0, a1 = position + q0, position + q1              # absolute query bounds
            k0, k1 = self._key_range(a0, a1, base, end)        # absolute key bounds

            attends = (self.query[:, :, :, q0:q1, :] @
                       self.key[:, :, :, k0-base:k1-base, :].transpose(0, 1, 2, 4, 3))
            attends = (attends / self.heads_dim**.5 if self.attention_scale is None else
                       attends * self.attention_scale)

            if self.is_decoder:
                # attends is freshly allocated by the matmul, so this can be in place
                mask, offset = self._block_mask(a0, a1, k0, k1)
                if mask is not None:
                    attends[..., offset:] += mask

            normalization = cupy.max(attends, axis = -1, keepdims = True)
            exponent      = cupy.exp(attends - normalization)
            attends       = exponent / cupy.sum(exponent, axis = -1, keepdims=True)

            self.softmax.append(attends)
            self.chunks.append((a0, a1, k0, k1))
            outputs.append(attends @ self.value[:, :, :, k0-base:k1-base, :])

        self.heads_out = self._merge_heads(outputs[0] if len(outputs) == 1 else
                                           cupy.concatenate(outputs, axis = -2))

        self.output = self.heads_out @ self.out_weights

        if self.cache is not None:
            self.cache.position = end

        return self.output

    def backward(self, gradient):

        B, T, C = gradient.shape

        if self.cache is not None:
            raise RuntimeError("backward() while a key/value cache is active: the cache holds "
                               "keys and values computed by earlier forwards, which have no "
                               "gradient path back to this one. Call stop_cache() first.")

        self.out_weight_grads += cupy.tensordot(self.heads_out.transpose(2, 0, 1), gradient, 2) / T
        gradient = gradient @ self.out_weights.transpose()

        gradient = self._split_heads(gradient, self.num_kv_heads, self.groups)

        query_grads = init_zeros_tensor(self.query.shape)
        key_grads   = init_zeros_tensor(self.key.shape)
        value_grads = init_zeros_tensor(self.value.shape)

        for attends, (q0, q1, k0, k1) in zip(self.softmax, self.chunks):

            block = gradient[:, :, :, q0:q1, :]

            # key and value ranges overlap between query blocks so they accumulate, and
            # every query head in a group reads the same kv head, so the group axis
            # folds back onto that one head
            value_grads[:, :, :, k0:k1, :] += (attends.transpose(0, 1, 2, 4, 3) @ block).sum(axis = 2, keepdims = True)

            block = block @ self.value[:, :, :, k0:k1, :].transpose(0, 1, 2, 4, 3)
            block = attends * ( block - (block * attends).sum(axis = -1, keepdims=True))
            block = (block / self.heads_dim**.5 if self.attention_scale is None else
                     block * self.attention_scale)

            key_grads[:, :, :, k0:k1, :] += (block.transpose(0, 1, 2, 4, 3) @
                                             self.query[:, :, :, q0:q1, :]).sum(axis = 2, keepdims = True)

            # query blocks are disjoint, so this writes rather than accumulates
            query_grads[:, :, :, q0:q1, :] = block @ self.key[:, :, :, k0:k1, :]

        if self.rope is not None:
            query_grads = self.rope.backward(query_grads)
            key_grads   = self.rope.backward(key_grads)

        if self.q_norm is not None:
            query_grads = self.q_norm.backward(query_grads)
            key_grads   = self.k_norm.backward(key_grads)

        if self.v_norm is not None:
            value_grads = self.v_norm.backward(value_grads)

        query_grads = self._merge_heads(query_grads)
        key_grads   = self._merge_heads(key_grads)
        value_grads = self._merge_heads(value_grads)

        if self.fused:
            gradient = cupy.concatenate((query_grads, key_grads, value_grads), axis = 2)
            self.qkv_weight_grads += cupy.tensordot(self.input.transpose(2, 0, 1), gradient, 2) / T
            return gradient @ self.qkv_weights.transpose()

        if self.kv_shared:
            # one projection served as both key and value, so both paths accumulate onto it
            key_grads = key_grads + value_grads

        self.q_weight_grads += cupy.tensordot(self.input.transpose(2, 0, 1), query_grads, 2) / T
        self.k_weight_grads += cupy.tensordot(self.input.transpose(2, 0, 1), key_grads,   2) / T

        gradient = query_grads @ self.q_weights.transpose() + key_grads @ self.k_weights.transpose()

        if not self.kv_shared:
            self.v_weight_grads += cupy.tensordot(self.input.transpose(2, 0, 1), value_grads, 2) / T
            gradient = gradient + value_grads @ self.v_weights.transpose()

        return gradient

class GatedFeedForward(Layer):

    CACHED = ("input", "output", "act_output", "gate_output", "hidden_output")

    def __init__(self, num_channels, activation, dropout_rate = 0.0, multiplier = 4):
        super(GatedFeedForward, self).__init__()
        
        self.activation:Layer = activation()
        self.dropout = Dropout(dropout_rate)

        hidden_channels = multiplier * num_channels
        
        self.act_output = None
        self.gate_output = None
        self.hidden_output = None

        self.weights1 = init_weight_tensor((num_channels,    hidden_channels), num_channels**0.5)
        self.weights2 = init_weight_tensor((num_channels,    hidden_channels), num_channels**0.5)
        self.weights3 = init_weight_tensor((hidden_channels, num_channels), hidden_channels**0.5)

        self.weight_grads1 = self.register(self.weights1)
        self.weight_grads2 = self.register(self.weights2)
        self.weight_grads3 = self.register(self.weights3)

    def forward(self, input):

        self.input = input
        
        self.act_output = self.activation.forward(self.input @ self.weights1)
        self.gate_output = self.input @ self.weights2
        
        self.hidden_output = self.dropout.forward(self.act_output * self.gate_output)
        self.output = self.hidden_output @ self.weights3

        return self.output

    def backward(self, gradient):
        
        B, T, C = gradient.shape

        self.weight_grads3 += cupy.tensordot(self.hidden_output.transpose(2, 0, 1), gradient, 2) / T
        
        gradient = self.dropout.backward(gradient @ self.weights3.transpose())
        act_gradient = self.activation.backward(gradient * self.gate_output)
        gate_gradient = gradient * self.act_output

        self.weight_grads1 += cupy.tensordot(self.input.transpose(2, 0, 1), act_gradient,  2) / T
        self.weight_grads2 += cupy.tensordot(self.input.transpose(2, 0, 1), gate_gradient, 2) / T

        return act_gradient @ self.weights1.transpose() + gate_gradient @ self.weights2.transpose()

    def clear_cache(self):
        super(GatedFeedForward, self).clear_cache()
        self.activation.clear_cache()
        self.dropout.clear_cache()

    def set_eval(self, eval_mode):
        self.dropout.set_eval(eval_mode)

class LayerNorm(Layer):

    CACHED = ("input", "output", "mean", "var", "std", "centered", "normed")

    def __init__(self, num_channels):
        super(LayerNorm, self).__init__()

        self.channels = num_channels
        self.eps = 1e-5

        self.gamma = init_zeros_tensor(num_channels) + 1
        self.beta  = init_zeros_tensor(num_channels)

        self.mean = None
        self.var = None
        self.std = None
        self.centered = None
        self.normed = None

        self.gamma_grads = self.register(self.gamma)
        self.beta_grads  = self.register(self.beta)

    def forward(self, input):

        self.input = input

        self.mean = cupy.mean(input, axis=-1, keepdims=True)
        self.var  = cupy.var(input, axis=-1, keepdims=True)

        self.centered = input - self.mean

        self.std = (self.var + self.eps)**0.5
        self.normed = self.centered / self.std

        self.output = self.gamma * self.normed + self.beta
        return self.output

    def backward(self, gradient):
        
        B, T, C = gradient.shape

        self.gamma_grads += cupy.sum(gradient * self.normed, axis=(0, 1)) / T
        self.beta_grads  += cupy.sum(gradient, axis=(0, 1)) / T

        gradient = gradient * self.gamma
        
        gradient_normed = (gradient - cupy.mean(gradient, axis = -1, keepdims = True)) / self.std
        
        return gradient_normed - self.centered * (cupy.mean(gradient * self.centered, axis = -1, keepdims = True) / self.std**3)


class Softcap(Layer):

    """Logit soft capping, cap * tanh(x / cap). Gemma 4 caps final logits at 30.

    Sits between the unembedding and SoftMax. That composes with the fused
    softmax + cross entropy gradient for free: SoftMax.backward is the identity and
    CrossEntropy already hands back d(loss)/d(softmax input), so this layer only has
    to apply its own tanh derivative.
    """

    CACHED = ("input", "output", "tanh")

    def __init__(self, cap = 30.0):
        super(Softcap, self).__init__()

        self.cap = cap
        self.tanh = None

    def forward(self, input):
        self.input  = input
        self.tanh   = cupy.tanh(input / self.cap)
        self.output = self.cap * self.tanh
        return self.output

    def backward(self, gradient):
        return gradient * (1 - self.tanh**2)


class TransformerBlock(Layer):

    """Pre norm transformer block, optionally with Gemma's post norms.

    post_norm = True gives the sandwich arrangement, a second norm on the branch
    output before it rejoins the residual stream:

        h = x + post_attn_norm(attn(pre_attn_norm(x)))
        y = h + post_ffn_norm(ffn(pre_ffn_norm(h)))

    Attention keywords (num_kv_heads, head_dim, qk_norm, rope, sliding_window,
    kv_shared, chunk_size) pass straight through to MultiHeadAttention.
    """

    def __init__(self, embed_dim, context_length, num_heads, activation, decoder = False,
                       dropout_rate = 0.0, norm = LayerNorm, post_norm = False, ffn_multiplier = 4,
                       num_kv_heads = None, head_dim = None, qk_norm = False,
                       rope = None, sliding_window = None, kv_shared = False,
                       chunk_size = None, attention_scale = None, v_norm = False):
        super(TransformerBlock, self).__init__()

        # Non-trainable checkpoint buffer, applied after both residual branches.
        self.layer_scalar = init_zeros_tensor(1) + 1
        self.pre_attn_norm  = norm(embed_dim)
        self.attn_block     = MultiHeadAttention(embed_dim, context_length, num_heads, decoder = decoder,
                                                 num_kv_heads = num_kv_heads, head_dim = head_dim,
                                                 qk_norm = qk_norm, rope = rope,
                                                 sliding_window = sliding_window, kv_shared = kv_shared,
                                                 chunk_size = chunk_size,
                                                 attention_scale = attention_scale, v_norm = v_norm)
        self.post_attn_norm = norm(embed_dim) if post_norm else None

        self.pre_ffn_norm  = norm(embed_dim)
        self.ffn           = GatedFeedForward(embed_dim, activation, dropout_rate = dropout_rate,
                                              multiplier = ffn_multiplier)
        self.post_ffn_norm = norm(embed_dim) if post_norm else None

        self.blocks = [block for block in (self.pre_attn_norm, self.attn_block, self.post_attn_norm,
                                           self.pre_ffn_norm,  self.ffn,        self.post_ffn_norm)
                       if block is not None]

        self.parameters = [tensor for block in self.blocks for tensor in block.parameters]
        self.gradients  = [tensor for block in self.blocks for tensor in block.gradients]
        self.moments    = [tensor for block in self.blocks for tensor in block.moments]
        self.variances  = [tensor for block in self.blocks for tensor in block.variances]

    def forward(self,input):

        self.input = input

        branch = self.attn_block.forward(self.pre_attn_norm.forward(self.input))
        if self.post_attn_norm is not None:
            branch = self.post_attn_norm.forward(branch)
        self.output = self.input + branch

        branch = self.ffn.forward(self.pre_ffn_norm.forward(self.output))
        if self.post_ffn_norm is not None:
            branch = self.post_ffn_norm.forward(branch)
        self.output = (self.output + branch) * self.layer_scalar

        return self.output

    def backward(self, gradient):

        gradient = gradient * self.layer_scalar
        branch = gradient
        if self.post_ffn_norm is not None:
            branch = self.post_ffn_norm.backward(branch)
        gradient = gradient + self.pre_ffn_norm.backward(self.ffn.backward(branch))

        branch = gradient
        if self.post_attn_norm is not None:
            branch = self.post_attn_norm.backward(branch)
        gradient = gradient + self.pre_attn_norm.backward(self.attn_block.backward(branch))

        return gradient

    def clear_cache(self):
        super(TransformerBlock, self).clear_cache()
        for block in self.blocks:
            block.clear_cache()

    def start_cache(self, batch_size, max_length, step = 1):
        super(TransformerBlock, self).start_cache(batch_size, max_length, step)
        for block in self.blocks:
            block.start_cache(batch_size, max_length, step)

    def stop_cache(self):
        super(TransformerBlock, self).stop_cache()
        for block in self.blocks:
            block.stop_cache()

    def set_eval(self, eval_mode):
        self.ffn.dropout.set_eval(eval_mode)


def gemma_layer_types(num_layers, pattern = 6):
    """Gemma 4's local/global interleave: every pattern-th layer attends globally.

    Gemma 4 31B is 60 layers at pattern 6, giving the documented 5 sliding : 1 global
    ratio with layers 5, 11, ... 59 global. The last layer is always global so the
    final representation has seen the whole sequence.
    """
    types = ["global" if (index + 1) % pattern == 0 else "sliding" for index in range(num_layers)]
    types[-1] = "global"
    return types
