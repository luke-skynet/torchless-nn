"""CPU reference and checkpoint tests; no CUDA or torch dependency."""
import json
import struct
import subprocess
import sys
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import backend

pytestmark = pytest.mark.skipif(backend.xp is not np, reason='CPU reference suite')

from backend import AMP_TYPE, inference_mode, empty_weights
from utils import Parameter
from layers import Dense, RMSNorm, LayerNorm, BatchNorm, Dropout, Convolution, MultiHeadAttention
from activations import GeLUTanh, SoftMax
from network import Network, CrossEntropy
from optimizer import Adam
from gemma import gemma_gpt
from transformer_adapters import GPTEmbeddingTable, GPTEmbedFront, GPTEmbedBack
from checkpoint import Checkpoint, StoredTensor, TensorSpec, model_config, tensor_specs, target_array


def tiny_gemma():
    return gemma_gpt(vocab_size=16, context_length=12, num_layers=2, embed_dim=8,
                     num_heads=2, head_dim=4, num_kv_heads=1, global_head_dim=8,
                     num_global_kv_heads=1, ffn_multiplier=2, sliding_window=4,
                     chunk_size=2)


def test_master_preserves_unrounded_initialization():
    value = np.array([1.0001, 0.1234567], dtype=np.float32)
    p = Parameter(value, False, dtype=AMP_TYPE)
    assert p.master is value
    assert p.data.dtype == AMP_TYPE
    assert np.any(p.data.astype(np.float32) != p.master)
    assert all(a.dtype == np.float32 for a in (p.grad, p.moment, p.variance))
    q = Parameter(value, True, dtype=AMP_TYPE)
    assert q.master is q.grad is q.moment is q.variance is None
    r = Parameter(value, False, dtype=np.float32)
    assert r.master is r.data is value


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_adam_reference_and_duplicate_update(dtype):
    p = np.array([1., -2.], dtype=dtype)
    g = np.array([0.5, -0.25], dtype=dtype)
    m, v = np.zeros_like(p), np.zeros_like(p)
    initial, grad = p.copy(), g.copy() / 3
    entry = (p, g, m, v, True)
    opt = Adam()
    opt.step([entry, entry], .01, .1, 1, 3)
    expected = initial - .01 * (grad / (np.abs(grad) + 1e-7) + .1 * initial)
    np.testing.assert_allclose(p, expected, rtol=1e-6)
    np.testing.assert_allclose(m, .1 * grad)
    np.testing.assert_allclose(v, .001 * grad**2, rtol=1e-6)
    assert not g.any()
    assert len(opt.entries) == 1


def test_adam_rejects_overlap_and_conflicting_sharing():
    a = np.ones(6, dtype=np.float32)
    opt = Adam()
    with pytest.raises(ValueError, match='overlap'):
        opt.step([(a[:2], a[1:3], np.zeros(2,np.float32), np.zeros(2,np.float32), False)], .01, 0, 1, 1)
    p, g, m, v = [np.zeros(2,np.float32) for _ in range(4)]
    with pytest.raises(ValueError, match='Shared'):
        opt.step([(p,g,m,v,False), (p,g,m,v,True)], .01, 0, 1, 1)


@pytest.mark.parametrize('norm,shape', [(RMSNorm,(2,3,4)), (LayerNorm,(2,3,4)), (BatchNorm,(2,4,2,2))])
def test_norm_backward_finite_difference(norm, shape):
    rng = np.random.default_rng(2)
    x = rng.normal(size=shape).astype(np.float32)
    g = rng.normal(size=shape).astype(np.float32)
    layer = norm(4)
    layer.forward(x)
    dx = layer.backward(g)
    index = (0,) * x.ndim
    old = x[index]
    eps = .002
    x[index] = old + eps
    plus = (layer.forward(x) * g).sum()
    x[index] = old - eps
    minus = (layer.forward(x) * g).sum()
    np.testing.assert_allclose(dx[index], (plus-minus)/(2*eps), atol=2e-3, rtol=1e-2)


def test_dropout_boolean_masks_and_backward():
    layer = Dropout(.25)
    x = np.ones((100,20),np.float32)
    y = layer.forward(x)
    assert layer.dropout_neurons.dtype == np.bool_
    np.testing.assert_array_equal(y, layer.backward(x))
    layer.set_eval(True)
    assert layer.forward(x) is x


def test_gemma_training_and_cached_prediction():
    model = tiny_gemma()
    ids = np.array([[1,2,1,3,4,5]])
    y = model.predict(ids)
    np.testing.assert_allclose(y.sum(-1), 1, atol=1e-6)
    criterion = CrossEntropy(16)
    model._backward(criterion.gradients(y, ids))
    shared = model.layers[0].embedding_table
    back = next(l for l in model.layers if isinstance(l, GPTEmbedBack))
    assert back.embedding_table is shared
    old = shared.table.copy()
    model._update(.001, .01, 1, ids.size)
    assert np.any(shared.table != old)
    assert not shared.table_grads.any()
    model.set_eval(True)
    full = model.predict_logits(ids)[:, -1:]
    model._start_cache(1, 12, 2)
    try:
        for i in range(0,6,2):
            cached = model.predict_logits(ids[:,i:i+2])
        np.testing.assert_allclose(cached, full, rtol=2e-5, atol=2e-5)
    finally:
        model._stop_cache()


def test_inference_has_no_training_allocations():
    with inference_mode(), empty_weights():
        model = tiny_gemma()
    for layer in model.layers:
        for p in layer.parameters:
            assert p.master is p.grad is p.moment is p.variance is None
    with pytest.raises(RuntimeError, match='inference'):
        model._update(.001,0,1,1)


def test_partial_accumulation_trains():
    model = Network([Dense(3,4), GeLUTanh(), Dense(4,2), SoftMax()])
    x = np.array([[1,0,0],[0,1,0],[0,0,1]],np.float32)
    y = np.array([0,1,0])
    original = model.layers[0].weights.copy()
    model.train(CrossEntropy(2), x, y, epochs=1, batch_size=1, batches_per_step=2)
    assert np.any(model.layers[0].weights != original)
    assert all(not p.grad.any() for l in model.layers for p in l.parameters)


def raw_checkpoint(tmp_path, values, dtype='BF16'):
    path = tmp_path / 'tensor.bin'
    values.tofile(path)
    ck = object.__new__(Checkpoint)
    ck.tensors = {'x': StoredTensor(path, 0, values.shape, dtype)}
    return ck


def test_bf16_checkpoint_chunks_preserve_bits_and_transpose(tmp_path):
    values = np.array([[1., -0., 1.125], [.25, -3.5, 2.]], dtype=AMP_TYPE)
    ck = raw_checkpoint(tmp_path, values)
    chunks = list(ck.chunks('x', transpose=True, chunk_bytes=8, dtype=AMP_TYPE))
    result = np.concatenate([b for _,b in chunks])
    assert all(b.flags.c_contiguous for _,b in chunks)
    np.testing.assert_array_equal(result.view(np.uint16), values.T.view(np.uint16))
    expanded = np.concatenate([b for _,b in ck.chunks('x', dtype=np.float32)])
    np.testing.assert_array_equal(expanded, values.astype(np.float32))


@pytest.mark.parametrize('bits', [0x7f80,0xff80,0x7fc1])
def test_nonfinite_bf16_rejected(tmp_path, bits):
    values = np.array([[bits]], dtype=np.uint16).view(AMP_TYPE)
    ck = raw_checkpoint(tmp_path, values)
    with pytest.raises(ValueError, match='non-finite'):
        list(ck.chunks('x',dtype=AMP_TYPE))


def test_checkpoint_loads_masters_and_compute_storage(tmp_path):
    values = np.array([[1.0001,.1234567]],np.float32)
    ck = raw_checkpoint(tmp_path,values,'F32')
    p = Parameter(np.zeros_like(values),False,dtype=AMP_TYPE)
    model = SimpleNamespace(layers=[SimpleNamespace(parameters=[p], weights=p.data)])
    ck.specs = {'x':TensorSpec(values.shape,('layers',0,'weights'))}
    ck.prefix = ''; ck.aliases = []; ck.report = {}
    report = ck.load_into(model)
    np.testing.assert_array_equal(p.master, values)
    np.testing.assert_array_equal(p.data, values.astype(AMP_TYPE))
    assert report['weight_and_scalar_bytes'] == 4
    assert not report['storage_estimated']


def test_bf16_cpu_mode_fails_clearly():
    env = dict(os.environ, XP_RUNTIME='CPU', XP_PRECISION='bfloat16')
    result = subprocess.run([sys.executable,'-c','import backend'],env=env,capture_output=True,text=True)
    assert result.returncode != 0
    assert 'patched CuPy' in result.stderr


@pytest.mark.parametrize('source_dtype', ['BF16','F16','F32'])
@pytest.mark.parametrize('destination_dtype', ['float32','bfloat16'])
def test_complete_checkpoint_roundtrip(checkpoint_factory, monkeypatch, source_dtype, destination_dtype):
    directory, values = checkpoint_factory(source_dtype)
    # Storage/loading can be tested with NumPy BF16 without simulating GPU execution.
    monkeypatch.setattr(backend,'MODEL_DTYPE',AMP_TYPE if destination_dtype=='bfloat16' else np.dtype('float32'))
    checkpoint=Checkpoint(directory,dtype=destination_dtype)
    estimate=checkpoint.report['weight_and_scalar_bytes']
    model,report=checkpoint.load_model(chunk_size=2)
    assert report['weight_and_scalar_bytes']==estimate
    for name,spec in checkpoint.specs.items():
        actual=target_array(model,spec.target)
        source=values[name].T if spec.transpose else values[name]
        np.testing.assert_array_equal(actual,source.astype(actual.dtype))
        assert actual.dtype==(np.dtype('float32') if spec.target[-1]=='gamma' else backend.MODEL_DTYPE)
    if destination_dtype=='float32':
        y=model.predict(np.array([[1,2,3]]))
        assert np.isfinite(y).all()


def test_alias_validation_uses_original_precision(checkpoint_factory, monkeypatch):
    directory,_=checkpoint_factory('F32',alias_delta=1e-6)
    monkeypatch.setattr(backend,'MODEL_DTYPE',AMP_TYPE)
    checkpoint=Checkpoint(directory,dtype='bfloat16')
    with pytest.raises(ValueError,match='differs'):
        checkpoint.load_model()


@pytest.mark.parametrize('name', ['ReLU', 'GeLU', 'GeLUTanh', 'SiLU', 'Softcap'])
def test_activation_scalars_preserve_float_type(name):
    import activations
    import layers
    cls = getattr(activations, name, None) or getattr(layers, name)
    layer = cls()
    if name == 'Softcap':
        layer.cap = np.float64(3.7)
    x = np.linspace(-3, 3, 48, dtype=backend.FLOAT_TYPE).reshape(2, 3, 8)
    y = layer.forward(x)
    dx = layer.backward(np.ones_like(x))
    assert y.dtype == dx.dtype == np.dtype(backend.FLOAT_TYPE)
    assert np.isfinite(y).all() and np.isfinite(dx).all()


@pytest.mark.parametrize('dtype', [np.dtype(backend.FLOAT_TYPE), AMP_TYPE])
def test_pooling_scalar_preserves_input_dtype(dtype):
    from layers import AveragePool
    layer = AveragePool(2, 3)
    x = np.arange(24, dtype=backend.FLOAT_TYPE).reshape(1, 1, 4, 6).astype(dtype)
    y = layer.forward(x)
    dx = layer.backward(np.ones_like(y))
    assert y.dtype == dx.dtype == dtype
    expected = np.full(x.shape, 1 / 6, dtype=dtype)
    np.testing.assert_array_equal(dx, expected)


def test_norm_and_position_scalars_preserve_float_type():
    from layers import RotaryEmbedding
    x = np.arange(48, dtype=backend.FLOAT_TYPE).reshape(2, 3, 8)
    for norm in [RMSNorm(8, eps=np.float64(1e-5)), LayerNorm(8, eps=np.float64(1e-5))]:
        y = norm.forward(x)
        assert y.dtype == np.dtype(backend.FLOAT_TYPE)
        assert norm.backward(np.ones_like(y)).dtype == np.dtype(backend.FLOAT_TYPE)
    rope = RotaryEmbedding(8, 16, theta=np.float64(10000))
    assert rope.cos.dtype == rope.sin.dtype == np.dtype(backend.FLOAT_TYPE)
    assert rope.rotate(x).dtype == np.dtype(backend.FLOAT_TYPE)
    table = GPTEmbeddingTable(16, 8)
    front = GPTEmbedFront(table, 16, positional='sinusoidal', scale_embeddings=True)
    assert front.positional_encoding.dtype == np.dtype(backend.FLOAT_TYPE)
    assert front.forward(np.array([[1, 2]])).dtype == np.dtype(backend.FLOAT_TYPE)


def test_temperature_loss_and_sampling_scalars_preserve_float_type():
    x = np.arange(12, dtype=backend.FLOAT_TYPE).reshape(3, 4) / backend.FLOAT_TYPE(10)
    layer = SoftMax(temperature=np.float64(.7))
    y = layer.forward(x)
    criterion = CrossEntropy(4, eps=np.float64(1e-7))
    labels = np.array([0, 1, 2])
    assert y.dtype == criterion.loss(y, labels).dtype == np.dtype(backend.FLOAT_TYPE)
    assert layer.backward(criterion.gradients(y, labels)).dtype == np.dtype(backend.FLOAT_TYPE)
    model = Network([layer])
    assert model._sample(y[:,None,:], top_k=2).shape == (3,1)


def test_temperature_updates_and_generation_restore():
    x = np.linspace(-2, 2, 24, dtype=backend.FLOAT_TYPE).reshape(2,3,4)
    layer = SoftMax(.5)
    first = layer.forward(x).copy()
    layer.temperature = np.float64(1.7)
    np.testing.assert_array_equal(layer.forward(x), SoftMax(1.7).forward(x))
    assert not np.array_equal(first, layer.output)
    model = tiny_gemma()
    final = model.layers[-1]
    final.temperature = np.float64(.7)
    inverse = final._inverse_temperature
    model.generate(np.array([[1,2,3]]), 1, temperature=0, step=2)
    assert final.temperature == backend.FLOAT_TYPE(.7)
    assert final._inverse_temperature == inverse


def test_dropout_rate_updates_cached_scale():
    x = np.ones((2,3), dtype=backend.FLOAT_TYPE)
    mask = np.array([[True,False,True],[False,True,True]])
    layer = Dropout(.25)
    first = layer.apply_mask(x, mask)
    layer.dropout_rate = np.float64(.5)
    second = layer.apply_mask(x, mask)
    np.testing.assert_array_equal(second, Dropout(.5).apply_mask(x, mask))
    assert not np.array_equal(first, second)
    with pytest.raises(ValueError):
        layer.dropout_rate = 1


def test_attention_scale_updates_after_construction():
    options = dict(embedding_dim=8,context_length=6,num_heads=2,chunk_size=2)
    layer, reference = MultiHeadAttention(**options), MultiHeadAttention(**options)
    for p,q in zip(layer.parameters, reference.parameters):
        q.data[...] = p.data
    x = np.random.default_rng(5).normal(size=(2,4,8)).astype(backend.FLOAT_TYPE)
    default = layer.forward(x).copy()
    layer.attention_scale = np.float64(.2)
    reference.attention_scale = .2
    np.testing.assert_array_equal(layer.forward(x), reference.forward(x))
    assert not np.array_equal(default, layer.output)
    layer.attention_scale = None
    np.testing.assert_array_equal(layer.forward(x), default)


def test_batchnorm_momentum_updates_running_statistics():
    layer = BatchNorm(3)
    layer.momentum = np.float64(.75)
    x = np.arange(24, dtype=backend.FLOAT_TYPE).reshape(2,3,2,2)
    layer.forward(x)
    np.testing.assert_allclose(layer.running_mean, backend.FLOAT_TYPE(.75)*x.mean((0,2,3),keepdims=True))
    previous = layer.running_mean.copy()
    layer.momentum = np.float64(.25)
    shifted = x + backend.FLOAT_TYPE(3)
    layer.forward(shifted)
    expected = backend.FLOAT_TYPE(.75)*previous + backend.FLOAT_TYPE(.25)*shifted.mean((0,2,3),keepdims=True)
    np.testing.assert_allclose(layer.running_mean, expected)
