"""Biases, chunk-local attention dropout, and learned positions compose correctly."""
import numpy as np
import pytest

from backend import xp, to_numpy
from activations import GeLUTanh, SoftMax
from layers import MultiHeadAttention, TransformerBlock, LayerNorm
from transformer_adapters import GPTEmbeddingTable, GPTEmbedFront, GPTEmbedBack
from network import Network
from backend import inference_mode, empty_weights
import backend
import layers
from test_feedforward import numerical_gradient


@pytest.fixture
def fp64(monkeypatch):
    monkeypatch.setattr(backend, 'FLOAT_TYPE', xp.float64)
    monkeypatch.setattr(layers, 'FLOAT_TYPE', xp.float64)


def attention(kind, **kwargs):
    options = {} if kind == 'fused' else dict(num_kv_heads=1, head_dim=2, qk_norm=True,
                                             kv_shared=kind == 'shared')
    return MultiHeadAttention(4, 8, 2, bias=True, **options, **kwargs)


def initialize(layer, rng):
    for param in layer.parameters:
        param[...] = xp.asarray(rng.normal(scale=.3, size=param.shape))


@pytest.mark.parametrize('kind', ['fused', 'gqa', 'shared'])
@pytest.mark.parametrize('decoder', [False, True])
def test_attention_bias_and_dropout_gradients(fp64, kind, decoder):
    layer = attention(kind, decoder=decoder, chunk_size=2,
                      sliding_window=3 if decoder else None,
                      attn_dropout_rate=.25, res_dropout_rate=.3)
    rng = np.random.default_rng(62)
    initialize(layer, rng)
    x = xp.asarray(rng.normal(size=(2, 5, 4)))
    upstream = xp.asarray(rng.normal(size=x.shape))

    def forward():
        layer.attn_dropout.dropout_rng = xp.random.default_rng(45)
        layer.res_dropout.dropout_rng = xp.random.default_rng(46)
        return layer.forward(x)

    def loss():
        return float(xp.sum(forward() * upstream))

    forward()
    assert len(layer.attn_masks) == 3
    assert len({id(mask) for mask in layer.attn_masks}) == 3
    dx = to_numpy(layer.backward(upstream)).copy()
    gradients = [to_numpy(g).copy() for g in layer.gradients]
    np.testing.assert_allclose(dx, numerical_gradient(loss, x), rtol=3e-5, atol=3e-7)
    for param, gradient in zip(layer.parameters, gradients):
        np.testing.assert_allclose(gradient, numerical_gradient(loss, param), rtol=3e-5, atol=3e-7)


@pytest.mark.parametrize('kind', ['fused', 'gqa', 'shared'])
def test_chunking_eval_and_cache_cleanup(fp64, kind):
    chunked = attention(kind, decoder=True, chunk_size=2, attn_dropout_rate=.4, res_dropout_rate=.3)
    full = attention(kind, decoder=True)
    rng = np.random.default_rng(3)
    initialize(chunked, rng)
    for a, b in zip(chunked.parameters, full.parameters):
        b[...] = a
    x = xp.asarray(rng.normal(size=(2, 5, 4)))
    chunked.forward(x)
    chunked.set_eval(True)
    assert chunked.attn_dropout.eval_mode and chunked.res_dropout.eval_mode
    np.testing.assert_allclose(to_numpy(chunked.forward(x)), to_numpy(full.forward(x)), atol=1e-12)
    assert all(mask is None for mask in chunked.attn_masks)
    upstream = xp.asarray(rng.normal(size=x.shape))
    np.testing.assert_allclose(to_numpy(chunked.backward(upstream)), to_numpy(full.backward(upstream)), atol=1e-12)
    for a, b in zip(chunked.gradients, full.gradients):
        np.testing.assert_allclose(to_numpy(a), to_numpy(b), atol=1e-12)
    chunked.clear_cache()
    for child in (chunked, chunked.attn_dropout, chunked.res_dropout, chunked.q_norm, chunked.k_norm):
        if child is not None:
            assert all(getattr(child, name) is None for name in child.CACHED)


def test_learned_positions_forward_and_gradients(fp64):
    rng = np.random.default_rng(7)
    table = xp.asarray(rng.normal(size=(6, 4)))
    owner = GPTEmbeddingTable(6, 4, table=table)
    layer = GPTEmbedFront(owner, 8, positional='learned', scale_embeddings=True)
    layer.positional_encoding[...] = xp.asarray(rng.normal(size=(8, 4)))
    ids = xp.asarray([[1, 1, 2], [3, 1, 2]])
    upstream = xp.asarray(rng.normal(size=(2, 3, 4)))
    expected = table[ids] * 2 + layer.positional_encoding[:3]
    np.testing.assert_array_equal(to_numpy(layer.forward(ids)), to_numpy(expected))
    layer.backward(upstream)
    analytic = [to_numpy(g).copy() for g in layer.gradients]
    for param, gradient in zip(layer.parameters, analytic):
        numeric = numerical_gradient(lambda: float(xp.sum(layer.forward(ids) * upstream)), param)
        np.testing.assert_allclose(gradient, numeric, atol=1e-9)
    assert not bool(xp.any(layer.positional_encoding_grads[3:]))
    assert len(owner.parameters) == len(owner.gradients) == 1
    assert layer.table_grads is owner.table_grads


def learned_model():
    table = GPTEmbeddingTable(7, 4)
    model = Network([
        GPTEmbedFront(table, 8, positional='learned'),
        TransformerBlock(4, 8, 2, GeLUTanh, decoder=True, chunk_size=2,
                         attn_bias=True, ffn_bias=True, attn_dropout_rate=.3, res_dropout_rate=.2),
        LayerNorm(4), GPTEmbedBack(table), SoftMax()])
    model.context_length = 8
    return model


def test_learned_positions_cached_generation(fp64):
    model = learned_model()
    rng = np.random.default_rng(73)
    for layer in model.layers:
        initialize(layer, rng)
    model.set_eval(True)
    tokens = xp.asarray([[1, 2, 1, 3, 2, 4]])
    full = model.predict_logits(tokens)
    model._start_cache(1, 8, 2)
    try:
        for start in range(0, 6, 2):
            actual = model.predict_logits(tokens[:, start:start + 2])
            np.testing.assert_allclose(to_numpy(actual), to_numpy(full[:, start + 1:start + 2]), atol=1e-12)
    finally:
        model._stop_cache()
    prompt = tokens[:, :3]
    sequence = prompt.copy()
    for _ in range(4):
        token = xp.argmax(model.predict_logits(sequence)[:, -1], axis=-1)[:, None]
        sequence = xp.concatenate((sequence, token), axis=1)
    model.set_eval(False)
    actual = model.generate(prompt, 4, temperature=0, step=2)
    np.testing.assert_array_equal(to_numpy(actual), to_numpy(sequence[:, 3:]))
    block = model.layers[1]
    assert not block.attn_block.attn_dropout.eval_mode
    assert not block.attn_block.res_dropout.eval_mode
    assert not block.ffn.output_dropout.eval_mode


def test_position_bounds_and_inference_allocation(monkeypatch):
    def no_rng(*args, **kwargs):
        raise AssertionError('empty_weights must not use random weight initialization')
    monkeypatch.setattr(backend, 'init_random_tensor', no_rng)
    with inference_mode(), empty_weights():
        model = learned_model()
    for layer in model.layers:
        assert not any((layer.gradients, layer.moments, layer.variances))
        for param in layer.parameters:
            param.fill(.1)
    model.set_eval(True)
    model.predict(xp.asarray([[1, 2, 3]]))
    block = model.layers[1]
    for child in (block.attn_block, block.attn_block.attn_dropout, block.attn_block.res_dropout):
        assert all(getattr(child, name) is None for name in child.CACHED)
    front = model.layers[0]
    assert front.positional_encoding_grads is None
    with pytest.raises(ValueError, match='context length'):
        front.forward(xp.ones((1, 9), dtype=xp.int64))
    front.start_cache(1, 8)
    front.forward(xp.ones((1, 7), dtype=xp.int64))
    with pytest.raises(ValueError, match='context length'):
        front.forward(xp.ones((1, 2), dtype=xp.int64))
    assert front.cache.position == 7
    front.stop_cache()


def test_ffn_residual_dropout_override():
    block = TransformerBlock(4, 8, 2, GeLUTanh, res_dropout_rate=.3, output_dropout_rate=.1)
    assert block.attn_block.res_dropout.dropout_rate == .3
    assert block.ffn.output_dropout.dropout_rate == .1


def test_biased_attention_matches_pytorch(fp64):
    import torch
    layer = attention('fused')
    rng = np.random.default_rng(19)
    initialize(layer, rng)
    reference = torch.nn.MultiheadAttention(4, 2, bias=True, batch_first=True).double()
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.tensor(to_numpy(layer.qkv_weights).T))
        reference.in_proj_bias.copy_(torch.tensor(to_numpy(layer.qkv_bias)))
        reference.out_proj.weight.copy_(torch.tensor(to_numpy(layer.out_weights).T))
        reference.out_proj.bias.copy_(torch.tensor(to_numpy(layer.out_bias)))
    values = rng.normal(size=(2, 5, 4))
    x = torch.tensor(values, requires_grad=True)
    expected, _ = reference(x, x, x, need_weights=True)
    upstream = rng.normal(size=values.shape)
    expected.backward(torch.tensor(upstream))
    actual = layer.forward(xp.asarray(values))
    dx = layer.backward(xp.asarray(upstream))
    np.testing.assert_allclose(to_numpy(actual), expected.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(to_numpy(dx), x.grad.numpy(), atol=1e-12)
    for actual_grad, expected_grad in (
        (layer.qkv_weight_grads, reference.in_proj_weight.grad.numpy().T),
        (layer.qkv_bias_grads, reference.in_proj_bias.grad.numpy()),
        (layer.out_weight_grads, reference.out_proj.weight.grad.numpy().T),
        (layer.out_bias_grads, reference.out_proj.bias.grad.numpy()),
    ):
        np.testing.assert_allclose(to_numpy(actual_grad), expected_grad, atol=1e-12)
