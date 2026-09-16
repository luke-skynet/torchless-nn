"""Gemma ties gradients and Adam state without changing checkpoint array access."""
import numpy as np
import pytest

from backend import xp, to_numpy
from gemma import gemma_gpt, gemma_parameter_count
from network import CrossEntropy
from transformer_adapters import GPTEmbeddingTable, GPTEmbedFront, GPTEmbedBack
from utils import inference_mode, empty_weights
import utils
import layers


CONFIG = dict(vocab_size=7, context_length=8, num_layers=2, embed_dim=8,
              num_heads=2, head_dim=8, num_kv_heads=1, sliding_window=3,
              pattern=2, ffn_multiplier=2)


def embedding_pair(model):
    return model.layers[0], next(layer for layer in model.layers if isinstance(layer, GPTEmbedBack))


@pytest.mark.parametrize('layer_type', [GPTEmbedFront, GPTEmbedBack])
def test_embedding_layers_reject_raw_arrays(layer_type):
    table = xp.zeros((7, 8))
    with pytest.raises(TypeError, match='requires a GPTEmbeddingTable'):
        layer_type(table, 8) if layer_type is GPTEmbedFront else layer_type(table)


@pytest.mark.parametrize('supplied', ['none', 'array', 'shared'])
def test_gemma_shared_gradient_and_single_adam_update(monkeypatch, supplied):
    monkeypatch.setattr(utils, 'FLOAT_TYPE', xp.float64)
    monkeypatch.setattr(layers, 'FLOAT_TYPE', xp.float64)
    rng = np.random.default_rng(12)
    original = xp.asarray(rng.normal(scale=.2, size=(7, 8)))
    argument = (None if supplied == 'none' else original if supplied == 'array'
                else GPTEmbeddingTable(7, 8, table=original))
    model = gemma_gpt(**CONFIG, embedding_table=argument)
    front, back = embedding_pair(model)
    assert front.table is back.table
    assert front.table_grads is back.table_grads
    assert front.embedding_table is back.embedding_table
    if supplied != 'none':
        assert front.table is original
    if supplied == 'shared':
        assert front.embedding_table is argument
    assert not any((back.parameters, back.gradients, back.moments, back.variances))
    assert sum(p is front.table for layer in model.layers for p in layer.parameters) == 1
    assert sum(p.size for layer in model.layers for p in layer.parameters) == gemma_parameter_count(**CONFIG)

    # Deterministic parameters and repeated IDs exercise both embedding paths.
    for layer in model.layers:
        for param in layer.parameters:
            param[...] = xp.asarray(rng.normal(scale=.3, size=param.shape))
    tokens = xp.asarray([[1, 1, 2], [2, 3, 1]])
    labels = xp.asarray([[2, 3, 1], [1, 2, 0]])
    criterion = CrossEntropy(7)
    model._backward(criterion.gradients(model.predict(tokens), labels))
    analytic = to_numpy(front.table_grads).copy()

    def loss():
        probs = model.predict(tokens)
        return float(-xp.log(probs[xp.arange(2)[:, None], xp.arange(3), labels]).sum())

    numeric = np.empty((7, 8))
    for index in np.ndindex(numeric.shape):
        value = float(front.table[index])
        front.table[index] = value + 1e-5
        plus = loss()
        front.table[index] = value - 1e-5
        minus = loss()
        front.table[index] = value
        numeric[index] = (plus - minus) / 2e-5
    np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=2e-7)

    # A second backward accumulates into the same buffer, then exactly one Adam
    # update uses both microbatches and both embedding paths.
    model._backward(criterion.gradients(model.predict(tokens), labels))
    np.testing.assert_allclose(to_numpy(front.table_grads), 2 * analytic, rtol=1e-10, atol=1e-10)
    before = to_numpy(front.table).copy()
    grad = analytic / labels.size
    expected = before - .003 * grad / (np.abs(grad) + 1e-7)
    model._update(.003, .01, 1, 2 * labels.size)
    np.testing.assert_allclose(to_numpy(front.table), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(to_numpy(front.moments[0]), .1 * grad, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(to_numpy(front.variances[0]), .001 * grad**2, rtol=1e-10, atol=1e-10)
    model._zero_grad()
    assert not bool(xp.any(back.table_grads))


def test_empty_inference_embedding_has_no_training_state(monkeypatch):
    def unexpected_rng(*args, **kwargs):
        raise AssertionError('empty_weights must not initialize random weights')
    monkeypatch.setattr(utils, 'init_random_tensor', unexpected_rng)
    with inference_mode(), empty_weights():
        model = gemma_gpt(**CONFIG)
    front, back = embedding_pair(model)
    shared = front.embedding_table
    assert front.table is back.table is shared.table
    assert front.table_grads is back.table_grads is shared.table_grads is None
    assert not any((shared.gradients, shared.moments, shared.variances))
    assert not any(layer.gradients or layer.moments or layer.variances for layer in model.layers)


@pytest.mark.parametrize('layer_type', [GPTEmbedFront, GPTEmbedBack])
@pytest.mark.parametrize('table_inference', [False, True])
def test_shared_embedding_rejects_mixed_inference_modes(layer_type, table_inference):
    with inference_mode(table_inference):
        shared = GPTEmbeddingTable(7, 8)
    with inference_mode(not table_inference):
        with pytest.raises(ValueError, match='same inference mode'):
            layer_type(shared, 8) if layer_type is GPTEmbedFront else layer_type(shared)


@pytest.mark.parametrize('shared', [False, True])
def test_gemma_rejects_wrong_embedding_shape(shared):
    table = xp.zeros((6, 8))
    if shared:
        table = GPTEmbeddingTable(6, 8, table=table)
    with pytest.raises(ValueError, match='embedding table shape'):
        gemma_gpt(**CONFIG, embedding_table=table)
