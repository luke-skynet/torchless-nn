"""bfloat16 training: float32 gradients and master weights under bfloat16 parameters."""
import ml_dtypes
import numpy as np
import pytest

from backend import xp, to_numpy
from gemma import gemma_gpt
from network import CrossEntropy
from utils import Layer
import layers
import transformer_adapters
import backend


CONFIG = dict(vocab_size=11, context_length=8, num_layers=2, embed_dim=16,
              num_heads=2, head_dim=8, num_kv_heads=1, global_head_dim=16,
              sliding_window=3, pattern=2, ffn_multiplier=2, chunk_size=3)


def gradients(model, tokens, labels):
    criterion = CrossEntropy(CONFIG['vocab_size'])
    model._backward(criterion.gradients(model.predict(tokens), labels))
    return [to_numpy(g).astype(np.float64) for layer in model.layers for g in layer.gradients]


def test_gradients_match_float32(monkeypatch):
    rng = np.random.default_rng(5)
    reference = gemma_gpt(**CONFIG)
    for layer in reference.layers:
        for param in layer.parameters:
            # values bfloat16 holds exactly, so both models start from the same weights
            param[...] = xp.asarray(rng.normal(scale=.3, size=param.shape).astype(ml_dtypes.bfloat16))

    for module in (backend, layers, transformer_adapters):
        monkeypatch.setattr(module, 'FLOAT_TYPE', ml_dtypes.bfloat16)
    model = gemma_gpt(**CONFIG)
    for layer, other in zip(model.layers, reference.layers):
        for param, source in zip(layer.parameters, other.parameters):
            assert param.dtype == ml_dtypes.bfloat16
            param[...] = source

    tokens = xp.asarray(rng.integers(CONFIG['vocab_size'], size=(3, 7)))
    labels = xp.asarray(rng.integers(CONFIG['vocab_size'], size=(3, 7)))
    expected = gradients(reference, tokens, labels)
    actual = gradients(model, tokens, labels)
    assert all(g.dtype == xp.float32 for layer in model.layers for g in layer.gradients)
    for ours, theirs in zip(actual, expected):
        assert np.linalg.norm(ours - theirs) <= .05 * np.linalg.norm(theirs) + 1e-6


def test_training_reduces_loss(bfloat16):
    rng = np.random.default_rng(8)
    model = gemma_gpt(**CONFIG)
    # one fixed sequence to memorize: next-token labels
    sequence = rng.integers(CONFIG['vocab_size'], size=(4, 8))
    data, labels = sequence[:, :-1], sequence[:, 1:]
    criterion = CrossEntropy(CONFIG['vocab_size'])

    def loss():
        model.set_eval(True)
        return float(criterion.loss(model.predict(xp.asarray(data)), xp.asarray(labels))) / labels.size

    before = loss()
    model.train(criterion, data, labels, epochs=30, batch_size=4, learning_rate=.01, weight_decay=0)
    assert loss() < .5 * before
    assert all(p.dtype == bfloat16 for layer in model.layers for p in layer.parameters)
    masters = [m for m in model.optimizer.masters if m is not None]
    assert len(masters) == len(model.optimizer.entries)
    assert all(m.dtype == np.float32 for m in masters)


@pytest.mark.skipif(xp is np, reason='NumPy promotes bfloat16 matmuls to float32; needs CUDA')
def test_activations_stay_bfloat16(bfloat16, monkeypatch):
    model = gemma_gpt(**CONFIG)
    seen = {}

    def watch(layer):
        forward = layer.forward
        def recorded(*args):
            result = forward(*args)
            seen.setdefault(type(layer).__name__, set()).add(result.dtype)
            return result
        monkeypatch.setattr(layer, 'forward', recorded)
        for value in vars(layer).values():
            for child in value if isinstance(value, (list, tuple)) else (value,):
                if isinstance(child, Layer) and 'forward' not in vars(child):
                    watch(child)

    for layer in model.layers:
        watch(layer)
    tokens = xp.asarray(np.random.default_rng(1).integers(CONFIG['vocab_size'], size=(2, 7)))
    model.predict(tokens)
    # only the final probabilities are float32, for sampling and the loss
    assert seen.pop('SoftMax') == {np.dtype(np.float32)}
    assert all(dtypes == {np.dtype(bfloat16)} for dtypes in seen.values()), seen
