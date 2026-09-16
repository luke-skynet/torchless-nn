"""Finite differences with fixed dropout masks exercise both FFN branches."""
import numpy as np
import pytest
from backend import xp

import layers
import utils
from activations import SiLU
from layers import GatedFeedForward, TransformerFeedForward, TransformerBlock


def host(value):
    return value.get() if hasattr(value, 'get') else np.asarray(value)


def numerical_gradient(loss, tensor):
    result = np.empty(tensor.shape)
    for index in np.ndindex(tensor.shape):
        original = float(tensor[index])
        tensor[index] = original + 1e-5
        plus = loss()
        tensor[index] = original - 1e-5
        minus = loss()
        tensor[index] = original
        result[index] = (plus - minus) / 2e-5
    return result


@pytest.mark.parametrize('glu', [False, True])
@pytest.mark.parametrize('rates', [(0., 0.), (.25, 0.), (0., .4), (.25, .4)])
@pytest.mark.parametrize('in_block', [False, True])
@pytest.mark.parametrize('bias', [False, True])
def test_gradients(monkeypatch, glu, rates, in_block, bias):
    monkeypatch.setattr(utils, 'FLOAT_TYPE', xp.float64)
    monkeypatch.setattr(layers, 'FLOAT_TYPE', xp.float64)
    kwargs = dict(glu=glu, hidden_dropout_rate=rates[0], output_dropout_rate=rates[1])
    layer = (TransformerBlock(4, 3, 2, SiLU, ffn_multiplier=2, attn_bias=bias, ffn_bias=bias,
                              **kwargs) if in_block
             else TransformerFeedForward(4, SiLU, multiplier=2, bias=bias, **kwargs))
    ffn = layer.ffn if in_block else layer
    rng = np.random.default_rng(82)
    for param in layer.parameters:
        param[...] = xp.asarray(rng.normal(scale=.3, size=param.shape))
    x = xp.asarray(rng.normal(size=(2, 3, 4)))
    upstream = xp.asarray(rng.normal(size=x.shape))

    def forward():
        # Replay independent masks for every finite-difference evaluation.
        ffn.dropout.dropout_rng = xp.random.default_rng(12)
        ffn.output_dropout.dropout_rng = xp.random.default_rng(34)
        return layer.forward(x)

    def loss():
        return float(xp.sum(forward() * upstream))

    forward()
    dx = layer.backward(upstream.copy())
    analytic = [host(grad).copy() for grad in layer.gradients]
    np.testing.assert_allclose(host(dx), numerical_gradient(loss, x), atol=2e-7, rtol=2e-5)
    for param, grad in zip(layer.parameters, analytic):
        # Parameter and input gradients both differentiate the summed loss.
        np.testing.assert_allclose(grad, numerical_gradient(loss, param),
                                   atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize('glu', [False, True])
def test_eval_cache_and_inference(glu):
    assert GatedFeedForward is TransformerFeedForward
    with utils.inference_mode():
        block = TransformerBlock(4, 3, 2, SiLU, glu=glu,
                                 hidden_dropout_rate=.25, output_dropout_rate=.4)
    ffn = block.ffn
    assert len(ffn.parameters) == (3 if glu else 2)
    assert not ffn.gradients and not ffn.moments and not ffn.variances
    if not glu:
        assert ffn.gate_weights is None and ffn.gate_weight_grads is None
    block.set_eval(True)
    assert ffn.dropout.eval_mode and ffn.output_dropout.eval_mode
    x = xp.ones((2, 3, 4), dtype=xp.float32)
    hidden = SiLU().forward(x @ ffn.act_weights)
    if glu:
        hidden *= x @ ffn.gate_weights
    np.testing.assert_allclose(host(ffn.forward(x)), host(hidden @ ffn.out_weights))
    block.clear_cache()
    for child in (ffn, ffn.activation, ffn.dropout, ffn.output_dropout):
        assert all(getattr(child, name) is None for name in child.CACHED)
    block.set_eval(False)
    assert not ffn.dropout.eval_mode and not ffn.output_dropout.eval_mode


def test_output_dropout_preserves_upstream_gradient():
    layer = TransformerFeedForward(4, SiLU, output_dropout_rate=.5)
    assert not layer.glu and len(layer.parameters) == 2
    assert not TransformerBlock(4, 3, 2, SiLU).ffn.glu
    layer.forward(xp.ones((2, 3, 4), dtype=xp.float32))
    upstream = xp.ones((2, 3, 4), dtype=xp.float32)
    layer.backward(upstream)
    np.testing.assert_array_equal(host(upstream), np.ones((2, 3, 4)))
