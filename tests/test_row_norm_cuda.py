"""Fused LayerNorm/RMSNorm against independent NumPy FP64 references.

Run with XP_RUNTIME=CUDA; BF16 cases need the patched CuPy build.
"""
import numpy as np
import pytest

import backend
import kernel_ops as ops
from layers import LayerNorm, RMSNorm

xp = backend.xp
pytestmark = pytest.mark.skipif(xp is np, reason='Requires CUDA')


def storage_dtype(name):
    if name == 'float32':
        return np.dtype('float32')
    if int(xp.cuda.Device().compute_capability) < 80:
        pytest.skip('BF16 requires SM80+')
    try:
        xp.empty(0, dtype=backend.AMP_TYPE)
    except (TypeError, ValueError):
        pytest.skip('BF16 requires the patched CuPy build')
    return backend.AMP_TYPE


def host(x):
    return xp.asnumpy(x.astype(xp.float32)).astype(np.float64)


def reference(x, g, gamma, beta, eps):
    centered = beta is not None
    mean = x.mean(-1, keepdims=True) if centered else 0.
    v = x - mean
    inv = 1 / np.sqrt((v * v).mean(-1, keepdims=True) + eps)
    normalized = v * inv
    scaled = g * gamma
    correction = (scaled * normalized).mean(-1, keepdims=True)
    avg = scaled.mean(-1, keepdims=True) if centered else 0.
    dx = (scaled - avg - normalized * correction) * inv
    axes = tuple(range(x.ndim - 1))
    dg = (g * normalized).sum(axes)
    db = g.sum(axes) if centered else None
    y = normalized * gamma + (beta if centered else 0.)
    return y, mean, inv, dx, dg, db


def run(x, g, gamma, beta, eps):
    if beta is not None:
        y, mean, inv = ops.layer_norm_forward(x, gamma, beta, eps)
        dx, dg, db = ops.layer_norm_backward(g, x, gamma, mean, inv)
    else:
        y, inv = ops.rms_norm_forward(x, gamma, eps)
        dx, dg = ops.rms_norm_backward(g, x, gamma, inv, isinstance(gamma, xp.ndarray))
        mean, db = None, None
    return y, mean, inv, dx, dg, db


@pytest.mark.parametrize('kind', ['layer', 'rms', 'rms_unscaled'])
@pytest.mark.parametrize('dtype_name', ['float32', 'bfloat16'])
@pytest.mark.parametrize('shape', [(1,), (37,), (3, 1), (2, 7, 37),
                                    (2, 3, 2, 5, 128), (2, 4097), (259, 33)])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'broadcast'])
def test_forward_backward(kind, dtype_name, shape, layout):
    dtype = storage_dtype(dtype_name)
    rng = np.random.default_rng(17)
    x = xp.asarray(rng.normal(size=shape).astype(np.float32), dtype=dtype)
    # Independent FP32 upstream gradient also covers mixed BF16/FP32 kernels.
    g = xp.asarray(rng.normal(size=shape).astype(np.float32))
    if layout == 'strided':
        x = x[..., ::-1]
        g = xp.ascontiguousarray(g[..., ::-1])[..., ::-1]
        if x.ndim > 2:
            x, g = x.swapaxes(0, 1), g.swapaxes(0, 1)
    elif layout == 'broadcast':
        x = xp.broadcast_to(x[..., :1], x.shape)
        g = xp.broadcast_to(g[..., :1], g.shape)
    width = shape[-1]
    gamma = xp.linspace(-.7, 1.3, width * 2, dtype=xp.float32)[::2]
    if kind == 'rms_unscaled':
        gamma = np.float32(1)
    beta = xp.linspace(-.2, .4, width, dtype=xp.float32)[::-1] if kind == 'layer' else None
    eps = np.float32(1e-5)
    expected = reference(host(x), host(g), host(gamma) if isinstance(gamma, xp.ndarray) else gamma,
                         host(beta) if beta is not None else None, eps)
    actual = run(x, g, gamma, beta, eps)
    for i, (value, wanted) in enumerate(zip(actual, expected)):
        if value is None:
            continue
        if i in (0, 3):
            assert value.dtype == dtype
            tolerance = .008 if dtype_name == 'bfloat16' else 5e-5
        else:
            assert value.dtype == xp.float32
            tolerance = 4e-4
        np.testing.assert_allclose(host(value), wanted, rtol=tolerance, atol=tolerance)
    if kind == 'rms_unscaled':
        assert actual[4] is None


@pytest.mark.parametrize('centered', [True, False])
@pytest.mark.parametrize('input_dtype,grad_dtype', [('float32', 'bfloat16'), ('bfloat16', 'bfloat16')])
def test_gradient_storage_and_layer_accumulation(centered, input_dtype, grad_dtype):
    layer = LayerNorm(37) if centered else RMSNorm(37)
    rng = np.random.default_rng(3)
    x = xp.asarray(rng.normal(size=(3, 5, 37)).astype(np.float32), dtype=storage_dtype(input_dtype))
    g = xp.asarray(rng.normal(size=x.shape).astype(np.float32), dtype=storage_dtype(grad_dtype))
    layer.gamma[...] = xp.linspace(-1, 1, 37, dtype=xp.float32)
    beta = layer.beta if centered else None
    expected = reference(host(x), host(g), host(layer.gamma), host(beta) if centered else None, layer.eps)
    layer.forward(x)
    for count in (1, 2):
        dx = layer.backward(g)
        tolerance = .008 if input_dtype == 'bfloat16' else 5e-5
        np.testing.assert_allclose(host(dx), expected[3], rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(host(layer.gamma_grads), expected[4] * count, rtol=4e-4, atol=4e-4)
        if centered:
            np.testing.assert_allclose(host(layer.beta_grads), expected[5] * count, rtol=4e-4, atol=4e-4)


def test_large_offset_and_constant_layer_norm():
    rng = np.random.default_rng(5)
    values = (10000 + rng.normal(size=(3, 1025))).astype(np.float32)
    values[0] = 10000
    x = xp.asarray(values)
    g = xp.asarray(rng.normal(size=values.shape).astype(np.float32))
    gamma, beta = xp.ones(1025, dtype=xp.float32), xp.zeros(1025, dtype=xp.float32)
    expected = reference(host(x), host(g), host(gamma), host(beta), 1e-5)
    actual = run(x, g, gamma, beta, np.float32(1e-5))
    for value, wanted in zip(actual, expected):
        np.testing.assert_allclose(host(value), wanted, rtol=.003, atol=.005)


@pytest.mark.parametrize('centered', [True, False])
def test_bounded_parameter_scratch_multiple_rows_per_chunk(centered):
    # Exercise the partial-count cap and a final short chunk.
    shape = (128 * ops._ROW_NORM_MAX_PARTIALS + 3, 7)
    rng = np.random.default_rng(23)
    x = xp.asarray(rng.normal(size=shape).astype(np.float32))
    g = xp.asarray(rng.normal(size=shape).astype(np.float32))
    gamma = xp.ones(shape[-1], dtype=xp.float32)
    beta = xp.zeros_like(gamma) if centered else None
    actual = run(x, g, gamma, beta, np.float32(1e-5))
    expected = reference(host(x), host(g), host(gamma), host(beta) if centered else None, 1e-5)
    np.testing.assert_allclose(host(actual[4]), expected[4], rtol=4e-4, atol=.003)
    if centered:
        np.testing.assert_allclose(host(actual[5]), expected[5], rtol=4e-4, atol=.003)


@pytest.mark.parametrize('centered', [True, False])
def test_empty_rows(centered):
    x = xp.empty((0, 3, 7), dtype=xp.float32)
    gamma = xp.ones(7, dtype=xp.float32)
    beta = xp.zeros_like(gamma) if centered else None
    y, mean, inv, dx, dg, db = run(x, x, gamma, beta, np.float32(1e-5))
    assert y.shape == dx.shape == x.shape
    assert inv.shape == (0, 3, 1)
    np.testing.assert_array_equal(host(dg), np.zeros(7))
    if centered:
        np.testing.assert_array_equal(host(db), np.zeros(7))
