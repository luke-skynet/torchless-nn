"""Custom epsilon values reach normalization, loss, and optimizer computations."""
import numpy as np
import pytest

from backend import xp, to_numpy
from layers import BatchNorm, LayerNorm, RMSNorm, MultiHeadAttention, TransformerBlock, Dense
from activations import GeLUTanh
from network import CrossEntropy, Network
from gemma import gemma_gpt


@pytest.mark.parametrize('norm,shape,axes', [
    (BatchNorm, (2, 3, 2, 2), (0, 2, 3)),
    (LayerNorm, (2, 2, 3), (-1,)),
    (RMSNorm, (2, 2, 3), (-1,)),
])
def test_custom_norm_epsilon(norm, shape, axes):
    values = np.random.default_rng(3).normal(scale=.01, size=shape).astype(np.float32)
    layer = norm(3, eps=.2)
    if norm is RMSNorm:
        expected = values / np.sqrt(np.mean(values**2, axis=axes, keepdims=True) + .2)
    else:
        centered = values - values.mean(axis=axes, keepdims=True)
        expected = centered / np.sqrt(values.var(axis=axes, keepdims=True) + .2)
    np.testing.assert_allclose(to_numpy(layer.forward(xp.asarray(values))), expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize('norm', [LayerNorm, RMSNorm])
def test_nested_norm_epsilon(norm):
    block = TransformerBlock(8, 8, 2, GeLUTanh, norm=norm, post_norm=True,
                             qk_norm=True, v_norm=True, eps=.02)
    nested = [block.pre_attn_norm, block.post_attn_norm, block.pre_ffn_norm, block.post_ffn_norm,
              block.attn_block.q_norm, block.attn_block.k_norm, block.attn_block.v_norm]
    assert all(child.eps == .02 for child in nested)
    attn = MultiHeadAttention(8, 8, 2, qk_norm=True, v_norm=True, eps=.03)
    assert all(child.eps == .03 for child in (attn.q_norm, attn.k_norm, attn.v_norm))


def test_existing_norm_defaults_and_custom_factory():
    assert BatchNorm(3).eps == LayerNorm(3).eps == 1e-5
    assert RMSNorm(3).eps == 1e-6
    for norm, eps in ((LayerNorm, 1e-5), (RMSNorm, 1e-6)):
        block = TransformerBlock(8, 8, 2, GeLUTanh, norm=norm, qk_norm=True, v_norm=True)
        assert block.pre_attn_norm.eps == block.pre_ffn_norm.eps == eps
        assert block.attn_block.q_norm.eps == block.attn_block.k_norm.eps == block.attn_block.v_norm.eps == 1e-6
    block = TransformerBlock(8, 8, 2, GeLUTanh, norm=lambda width: LayerNorm(width, eps=.04))
    assert block.pre_attn_norm.eps == .04


def test_gemma_epsilon_reaches_all_norms():
    model = gemma_gpt(vocab_size=7, context_length=8, num_layers=2, embed_dim=8,
                      num_heads=2, head_dim=8, num_kv_heads=1, sliding_window=3,
                      pattern=2, rms_norm_eps=.07)
    norms = []
    for layer in model.layers:
        if isinstance(layer, TransformerBlock):
            norms.extend(child for child in layer.blocks if isinstance(child, RMSNorm))
            norms.extend((layer.attn_block.q_norm, layer.attn_block.k_norm, layer.attn_block.v_norm))
        elif isinstance(layer, RMSNorm):
            norms.append(layer)
    assert len(norms) == 15
    assert all(norm.eps == .07 for norm in norms)


def test_loss_epsilon():
    criterion = CrossEntropy(2, eps=.1)
    probs = xp.asarray([[0., 1.], [.5, .5]])
    labels = xp.asarray([0, 1])
    np.testing.assert_allclose(float(criterion.loss(probs, labels)), -np.log(.1) - np.log(.6))
    assert CrossEntropy(2).eps == 1e-7


def test_adam_epsilon_in_denominator():
    layer = Dense(2, 2)
    for param, grad in zip(layer.parameters, layer.gradients):
        param.fill(1)
        grad.fill(.4)
    model = Network([layer])
    model._update(.1, 0., 1, 2, eps=.3)
    for param in layer.parameters:
        np.testing.assert_allclose(to_numpy(param), 1 - .1 * .2 / (.2 + .3), rtol=1e-6)
