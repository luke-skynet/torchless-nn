"""Split BatchNorm reductions versus independent NumPy FP64 calculations.

Run with XP_RUNTIME=CUDA (FP32 on stock CuPy; BF16 on the patched build).
"""
import numpy as np
import pytest

import backend
import kernel_ops as ops
from layers import BatchNorm

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
    axes = (0, 2, 3)
    count = x.shape[0] * x.shape[2] * x.shape[3]
    mean = x.mean(axes, keepdims=True)
    var = x.var(axes, keepdims=True)
    inv = 1 / np.sqrt(var + eps)
    normalized = (x - mean) * inv
    dg = (g * normalized).sum(axes, keepdims=True)
    db = g.sum(axes, keepdims=True)
    dx = gamma * inv * (g - db / count - normalized * dg / count)
    return normalized * gamma + beta, mean, var, inv, dx, dg, db


@pytest.mark.parametrize('dtype_name', ['float32', 'bfloat16'])
@pytest.mark.parametrize('shape', [(1, 3, 1, 1), (2, 5, 17, 19), (1, 3, 64, 64), (3, 7, 33, 47)])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'broadcast'])
def test_statistics_and_gradients(dtype_name, shape, layout):
    dtype = storage_dtype(dtype_name)
    rng = np.random.default_rng(64)
    x = xp.asarray(rng.normal(size=shape).astype(np.float32), dtype=dtype)
    g = xp.asarray(rng.normal(size=shape).astype(np.float32), dtype=dtype)
    if layout == 'strided':
        # Transposed/reversed activations and independently padded/cropped grads.
        x = x.transpose(0, 1, 3, 2)[:, :, :, ::-1]
        padded = xp.pad(g.transpose(0, 1, 3, 2), ((0, 0), (0, 0), (1, 1), (2, 2)))
        g = padded[:, :, 1:-1, 2:-2]
    elif layout == 'broadcast':
        x = xp.broadcast_to(x[:1, :1], shape)
        g = xp.broadcast_to(g[:, :, :1, :1], shape)
    channels = shape[1]
    gamma = xp.linspace(-1, 1, channels, dtype=xp.float32).reshape(1, channels, 1, 1)
    beta = xp.linspace(.2, .7, channels, dtype=xp.float32).reshape(gamma.shape)
    eps = np.float32(1e-5)
    expected = reference(host(x), host(g), host(gamma), host(beta), eps)
    y, mean, var, inv = ops.batch_norm_forward(x, gamma, beta, eps)
    dx, dg, db = ops.batch_norm_backward(g, x, gamma, mean, inv)
    for actual, wanted in zip((mean, var, inv, dg, db),
                              (expected[1], expected[2], expected[3], expected[5], expected[6])):
        assert actual.dtype == xp.float32
        np.testing.assert_allclose(host(actual), wanted, rtol=3e-4, atol=3e-4)
    for actual, wanted in ((y, expected[0]), (dx, expected[4])):
        assert actual.dtype == dtype
        tolerance = .008 if dtype_name == 'bfloat16' else 3e-5
        np.testing.assert_allclose(host(actual), wanted, rtol=tolerance, atol=tolerance)


def test_large_offset_stable_variance():
    # E[x*x] - E[x]**2 catastrophically cancels for these FP32 values.
    rng = np.random.default_rng(7)
    x = xp.asarray((10000 + rng.normal(size=(5, 3, 33, 47))).astype(np.float32))
    gamma = xp.ones((1, 3, 1, 1), dtype=xp.float32)
    beta = xp.zeros_like(gamma)
    y, mean, var, inv = ops.batch_norm_forward(x, gamma, beta, np.float32(1e-5))
    expected = reference(host(x), np.ones(x.shape), host(gamma), host(beta), 1e-5)
    np.testing.assert_allclose(host(mean), expected[1], rtol=0, atol=.003)
    np.testing.assert_allclose(host(var), expected[2], rtol=.002, atol=.002)
    np.testing.assert_allclose(host(y), expected[0], rtol=.003, atol=.005)


def test_finish_reduces_more_chunks_than_threads():
    size = ops._BATCH_NORM_CHUNK_SIZE * (ops._BATCH_NORM_BLOCK_SIZE + 1) + 1
    rng = np.random.default_rng(8)
    x = xp.asarray(rng.normal(size=(1, 1, 1, size)).astype(np.float32))
    g = xp.asarray(rng.normal(size=x.shape).astype(np.float32))
    gamma = xp.ones((1, 1, 1, 1), dtype=xp.float32)
    beta = xp.zeros_like(gamma)
    expected = reference(host(x), host(g), host(gamma), host(beta), 1e-5)
    y, mean, var, inv = ops.batch_norm_forward(x, gamma, beta, np.float32(1e-5))
    dx, dg, db = ops.batch_norm_backward(g, x, gamma, mean, inv)
    for actual, wanted in zip((y, mean, var, inv, dx, dg, db), expected):
        np.testing.assert_allclose(host(actual), wanted, rtol=3e-4, atol=2e-3)


@pytest.mark.parametrize('dtype_name', ['float32', 'bfloat16'])
def test_layer_state_and_evaluation(monkeypatch, dtype_name):
    dtype = storage_dtype(dtype_name)
    layer = BatchNorm(3)
    rng = np.random.default_rng(6)
    x = xp.asarray(rng.normal(size=(3, 3, 43, 37)).astype(np.float32), dtype=dtype)
    # Mixed storage: FP32 upstream gradient with either input dtype.
    g = xp.asarray(rng.normal(size=x.shape).astype(np.float32))
    layer.gamma[...] = xp.asarray([-.7, 0., 1.3], dtype=xp.float32).reshape(1, 3, 1, 1)
    expected = reference(host(x), host(g), host(layer.gamma), host(layer.beta), layer.eps)
    layer.forward(x)
    np.testing.assert_allclose(host(layer.running_mean), .1 * expected[1], atol=1e-6)
    np.testing.assert_allclose(host(layer.running_var), .1 * expected[2], rtol=1e-5)
    for multiplier in (1, 2):
        dx = layer.backward(g)
        assert dx.dtype == dtype
        np.testing.assert_allclose(host(layer.gamma_grads), multiplier * expected[5], rtol=3e-4, atol=3e-4)
        np.testing.assert_allclose(host(layer.beta_grads), multiplier * expected[6], rtol=3e-4, atol=3e-4)
    layer.set_eval(True)
    mean, var = host(layer.running_mean), host(layer.running_var)

    def no_reduction(*args):
        pytest.fail('Evaluation must not launch BatchNorm reductions')

    monkeypatch.setattr(ops, '_batch_norm_module', no_reduction)
    y = layer.forward(x)
    wanted = (host(x) - mean) / np.sqrt(var + layer.eps) * host(layer.gamma) + host(layer.beta)
    tolerance = .008 if dtype_name == 'bfloat16' else 3e-5
    np.testing.assert_allclose(host(y), wanted, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(host(layer.running_mean), mean)
    np.testing.assert_array_equal(host(layer.running_var), var)
