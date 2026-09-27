"""Run with XP_RUNTIME=CUDA XP_PRECISION=float32 python -m pytest.

Compare shared kernels with the previous CuPy FP32 expressions, including
gradients. CPU runs skip these tests rather than pretending to validate CUDA.
"""
import numpy as np
import pytest

import backend
import kernel_ops as ops
from activations import ReLU, GeLU, GeLUTanh, SiLU, SoftMax
from layers import BatchNorm, LayerNorm, RMSNorm, Softcap, RotaryEmbedding, MultiHeadAttention

xp = backend.xp
pytestmark = pytest.mark.skipif(
    xp is np or backend.MODEL_DTYPE != np.dtype('float32'),
    reason='Requires CUDA and XP_PRECISION=float32')


def random(shape):
    return xp.asarray(np.random.default_rng(42).normal(size=shape).astype(np.float32))


def close(actual, expected):
    np.testing.assert_allclose(xp.asnumpy(actual), xp.asnumpy(expected),
                               rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize('factory,shape', [
    (ReLU, (2, 3, 8)), (GeLU, (2, 3, 8)), (GeLUTanh, (2, 3, 8)),
    (SiLU, (2, 3, 8)), (Softcap, (2, 3, 8)),
    (lambda: SoftMax(.7, fused_loss=False), (2, 3, 37)),
    (lambda: SoftMax(.7, fused_loss=True), (2, 3, 37)),
    (lambda: BatchNorm(4), (2, 4, 5, 5)),
    (lambda: LayerNorm(8), (2, 3, 8)),
    (lambda: LayerNorm(8), (3, 8)),
    (lambda: LayerNorm(8), (2, 3, 4, 8)),
    (lambda: RMSNorm(8), (2, 3, 8)),
    (lambda: RMSNorm(8), (2, 3, 2, 5, 8)),
    (lambda: RMSNorm(8, with_scale=False), (2, 3, 8)),
])
def test_forward_backward_reference(monkeypatch, factory, shape):
    layer = factory()
    x, g = random(shape), random(shape)
    y = layer.forward(x).copy()
    dx = layer.backward(g).copy()
    grads = [p.grad.copy() for p in layer.parameters]
    layer.zero_grad()
    with monkeypatch.context() as reference:
        reference.setattr(ops, 'use_kernels', lambda x: False)
        close(y, layer.forward(x))
        close(dx, layer.backward(g))
    for actual, p in zip(grads, layer.parameters):
        close(actual, p.grad)
    if isinstance(layer, BatchNorm):
        layer.set_eval(True)
        y = layer.forward(x).copy()
        with monkeypatch.context() as reference:
            reference.setattr(ops, 'use_kernels', lambda x: False)
            close(y, layer.forward(x))


@pytest.mark.parametrize('fraction', [1., .5])
def test_rope_strided_reference(monkeypatch, fraction):
    rope = RotaryEmbedding(8, 16, partial_rotary_factor=fraction)
    x = random((2, 7, 3, 8)).transpose(0, 2, 1, 3)
    y, dx = rope.rotate(x, 2), rope.backward(x, 2)
    monkeypatch.setattr(ops, 'use_kernels', lambda x: False)
    close(y, rope.rotate(x, 2))
    close(dx, rope.backward(x, 2))


@pytest.mark.parametrize('decoder', [False, True])
def test_chunked_attention_reference(monkeypatch, decoder):
    layer = MultiHeadAttention(8, 8, 2, num_kv_heads=1,
                               chunk_size=2, decoder=decoder)
    layer.set_eval(True)  # Disable random dropout while retaining backward state.
    x, g = random((2, 5, 8)), random((2, 5, 8))
    y = layer.forward(x).copy()
    dx = layer.backward(g).copy()
    grads = [p.grad.copy() for p in layer.parameters]
    layer.zero_grad()
    monkeypatch.setattr(ops, 'use_kernels', lambda x: False)
    close(y, layer.forward(x))
    close(dx, layer.backward(g))
    for actual, p in zip(grads, layer.parameters):
        close(actual, p.grad)


def test_embedding_scatter_repeated_ids():
    ids = xp.asarray([[1, 2, 1], [0, 2, 1]])
    g = random((2, 3, 8))
    actual = xp.zeros((4, 8), dtype=xp.float32)
    expected = xp.zeros_like(actual)
    ops.scatter_embedding(actual, ids, g, np.float32(.7))
    xp.add.at(expected, ids, g * np.float32(.7))
    close(actual, expected)
