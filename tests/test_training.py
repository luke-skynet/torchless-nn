"""Accumulated updates must match a full batch, including epoch remainders."""
import numpy as np
import pytest

from backend import xp, to_numpy
from activations import GeLUTanh, SoftMax
from layers import LayerNorm, TransformerBlock
from network import CrossEntropy, Network
from transformer_adapters import GPTEmbeddingTable, GPTEmbedFront, GPTEmbedBack, VitMLPHead, VitProjector
import backend


@pytest.mark.parametrize('task', ['vision', 'tokens'])
@pytest.mark.parametrize('batch_size,batches_per_step', [(2, 2), (2, 4), (4, 1)])
@pytest.mark.parametrize('adam_eps', [1e-7, .05])
def test_accumulation_matches_full_batch(monkeypatch, task, batch_size, batches_per_step, adam_eps):
    monkeypatch.setattr(backend, 'FLOAT_TYPE', xp.float64)
    # Keep the grouping identical between training and the explicit reference steps.
    monkeypatch.setattr(np.random, 'permutation', lambda n: np.arange(n))
    rng = np.random.default_rng(27)
    count, classes, width, length = 7, 3, 4, 5

    def build():
        block = TransformerBlock(width, length, 2, GeLUTanh)
        if task == 'vision':
            stack = [VitProjector((1, 4, 4), (2, 2), width), block,
                     LayerNorm(width), VitMLPHead(width, classes), SoftMax()]
        else:
            table = GPTEmbeddingTable(classes, width)
            stack = [GPTEmbedFront(table, length, positional='none'), block, LayerNorm(width),
                     GPTEmbedBack(table), SoftMax()]
        return Network(stack)

    model, reference = build(), build()
    for layer, other in zip(model.layers, reference.layers):
        for param, target in zip(layer.parameters, other.parameters):
            param[...] = xp.asarray(rng.normal(scale=.3, size=param.shape))
            target[...] = param

    shape = (count, 1, 4, 4) if task == 'vision' else (count, length)
    label_shape = (count,) if task == 'vision' else (count, length)
    data = rng.normal(size=shape) if task == 'vision' else rng.integers(classes, size=shape)
    labels = rng.integers(classes, size=label_shape)
    criterion = CrossEntropy(classes)
    updates = []
    original_update = model._update

    def observe_update(learning_rate, weight_decay, t, num_samples, eps=1e-7):
        # Checking gradients also catches normalization errors hidden by Adam's
        # approximate invariance to a constant gradient scale.
        grads = [to_numpy(g).copy() / num_samples
                 for layer in model.layers for g in layer.gradients]
        updates.append((t, num_samples, grads))
        assert eps == adam_eps
        original_update(learning_rate, weight_decay, t, num_samples, eps=eps)

    monkeypatch.setattr(model, '_update', observe_update)
    model.train(criterion, data.copy(), labels.copy(), epochs=2,
                batch_size=batch_size, batches_per_step=batches_per_step,
                learning_rate=.003, weight_decay=.01, adam_eps=adam_eps)

    effective_batch = batch_size * batches_per_step
    step = 0
    for _ in range(2):
        reference.set_eval(False)
        for start in range(0, count, effective_batch):
            x = xp.asarray(data[start:start + effective_batch])
            y = xp.asarray(labels[start:start + effective_batch])
            prediction = reference.predict(x)
            reference._backward(criterion.gradients(prediction, y))
            t, num_samples, observed = updates[step]
            step += 1
            assert (t, num_samples) == (step, y.size)
            expected = [to_numpy(g) / y.size
                        for layer in reference.layers for g in layer.gradients]
            for actual, full_batch in zip(observed, expected):
                np.testing.assert_allclose(actual, full_batch, rtol=2e-5, atol=2e-8)
            reference._update(.003, .01, step, y.size, eps=adam_eps)
            reference._zero_grad()

    assert len(updates) == step
    for layer, other in zip(model.layers, reference.layers):
        for attr in ('parameters', 'moments', 'variances'):
            for actual, expected in zip(getattr(layer, attr), getattr(other, attr)):
                np.testing.assert_allclose(to_numpy(actual), to_numpy(expected),
                                           rtol=2e-5, atol=2e-8)
        for gradient in layer.gradients:
            assert not bool(xp.any(gradient))
