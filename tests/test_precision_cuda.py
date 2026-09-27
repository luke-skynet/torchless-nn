"""Run with XP_RUNTIME=CUDA XP_PRECISION=bfloat16 python -m pytest.

These tests compile/execute every fused kernel on the patched CuPy backend.
They deliberately do not simulate CUDA with NumPy.
"""
import numpy as np
import pytest
import backend

pytestmark = pytest.mark.skipif(
    backend.xp is np or backend.MODEL_DTYPE != backend.AMP_TYPE,
    reason='Requires patched CuPy, CUDA SM80+, and XP_PRECISION=bfloat16')

from backend import xp, FLOAT_TYPE, AMP_TYPE, inference_mode, empty_weights
import kernel_ops as ops
from layers import (RMSNorm, LayerNorm, BatchNorm, RotaryEmbedding, Dense,
                    Convolution, MultiHeadAttention, Softcap, Dropout)
from activations import ReLU, GeLU, GeLUTanh, SiLU, SoftMax
from transformer_adapters import GPTEmbeddingTable, GPTEmbedFront, GPTEmbedBack, VitProjector, VitMLPHead
from network import CrossEntropy, Network
from gemma import gemma_gpt
from optimizer import Adam


def host(x):
    return xp.asnumpy(x.astype(FLOAT_TYPE))


def random(shape, seed=1):
    return xp.asarray(np.random.default_rng(seed).normal(size=shape).astype(np.float32), dtype=AMP_TYPE)


def test_norm_kernels_against_fp32_reference():
    for cls, shape in [(RMSNorm,(2,3,16)), (LayerNorm,(2,3,16)), (BatchNorm,(2,16,3,4))]:
        low, ref = cls(16), cls(16)
        x, g = random(shape), random(shape,2)
        low.gamma[...] = xp.linspace(.5,1.5,16).reshape(low.gamma.shape)
        ref.gamma[...] = low.gamma
        expected = ref.forward(x.astype(FLOAT_TYPE))
        y = low.forward(x)
        assert y.dtype == AMP_TYPE
        np.testing.assert_allclose(host(y),host(expected),atol=.012,rtol=.015)
        dx, expected_dx = low.backward(g), ref.backward(g.astype(FLOAT_TYPE))
        assert dx.dtype == AMP_TYPE
        np.testing.assert_allclose(host(dx),host(expected_dx),atol=.015,rtol=.025)
        for p,q in zip(low.parameters,ref.parameters):
            assert p.grad.dtype == FLOAT_TYPE
            np.testing.assert_allclose(host(p.grad),host(q.grad),atol=2e-4,rtol=2e-4)
    unscaled = RMSNorm(16,with_scale=False)
    y = unscaled.forward(random((2,3,16)))
    assert unscaled.backward(y).dtype == AMP_TYPE


@pytest.mark.parametrize('width,factor,proportional', [(16,1,False),(16,.5,False),(16,.25,True)])
def test_rope_strides_and_backward(width,factor,proportional):
    rope = RotaryEmbedding(width,20,partial_rotary_factor=factor,proportional=proportional)
    x = random((2,5,3,2,width)).transpose(0,2,3,1,4)
    y = rope.rotate(x,3)
    xf = x.astype(FLOAT_TYPE)
    expected = rope.rotate(xf,3)
    np.testing.assert_allclose(host(y),host(expected),atol=.012,rtol=.015)
    restored = rope.backward(y,3)
    np.testing.assert_allclose(host(restored),host(x),atol=.025,rtol=.025)
    assert y.dtype == restored.dtype == AMP_TYPE


@pytest.mark.parametrize('cls', [ReLU,GeLU,GeLUTanh,SiLU,Softcap])
def test_activation_kernels(cls):
    low, ref = cls(), cls()
    x = random((4,128)); g = random(x.shape,3)
    np.testing.assert_allclose(host(low.forward(x)),host(ref.forward(x.astype(FLOAT_TYPE))),atol=.012,rtol=.015)
    actual = low.backward(g)
    assert actual.dtype == AMP_TYPE
    np.testing.assert_allclose(host(actual),host(ref.backward(g.astype(FLOAT_TYPE))),atol=.015,rtol=.025)


@pytest.mark.parametrize('width', [1,17,257,4097])
def test_softmax_fused_mask_dropout_and_backward(width):
    x = random((2,3,2,3,width))
    mask = xp.asarray(np.random.default_rng(3).random(x.shape) > .25)
    y,p = ops.softmax(x,AMP_TYPE,scale=.5,mask=mask,dropout_scale=4/3,save=True,
                      causal=True,qstart=width-2,kstart=0,window=7)
    xf=host(x)*.5
    q=(width-2+np.arange(3))[:,None]
    k=np.arange(width)[None,:]
    allowed=(k<=q)&(q-k<7)
    xf += np.where(allowed,0,-1e9).astype(np.float32)
    e=np.exp(xf-xf.max(-1,keepdims=True)); expected=e/e.sum(-1,keepdims=True)
    np.testing.assert_allclose(host(p),expected,rtol=3e-5,atol=3e-6)
    np.testing.assert_allclose(host(y),expected*xp.asnumpy(mask)*(4/3),rtol=.008,atol=.004)
    g=random(x.shape,6)
    dx=ops.softmax_backward(g,p,AMP_TYPE,mask=mask,dropout_scale=4/3,scale=.5)
    gf=host(g)*xp.asnumpy(mask)*(4/3)
    expected_dx=expected*(gf-(gf*expected).sum(-1,keepdims=True))*.5
    np.testing.assert_allclose(host(dx),expected_dx,rtol=.02,atol=.003)
    final,_=ops.softmax(x,FLOAT_TYPE,save=False)
    assert final.dtype==FLOAT_TYPE
    np.testing.assert_allclose(host(final).sum(-1),1,atol=2e-6)


def test_softmax_layer_both_backward_modes():
    x=random((2,3,17)); g=random(x.shape,4)
    for fused in [False,True]:
        layer=SoftMax(temperature=.7,fused_loss=fused)
        y=layer.forward(x)
        assert y.dtype==(FLOAT_TYPE if fused else AMP_TYPE)
        dx=layer.backward(g)
        p=host(x)/.7; p=np.exp(p-p.max(-1,keepdims=True)); p/=p.sum(-1,keepdims=True)
        expected=host(g)/.7 if fused else p*(host(g)-(host(g)*p).sum(-1,keepdims=True))/.7
        np.testing.assert_allclose(host(dx),expected,rtol=.025,atol=.006)


def test_embedding_scatter_repeated_ids_and_sinusoidal():
    table=GPTEmbeddingTable(16,8)
    layer=GPTEmbedFront(table,8,scale_embeddings=True)
    ids=xp.asarray([[1,2,1],[2,2,3]])
    y=layer.forward(ids); g=random(y.shape)
    layer.backward(g)
    expected=np.zeros((16,8),np.float32)
    np.add.at(expected,xp.asnumpy(ids),host(g)*8**.5)
    np.testing.assert_allclose(host(table.table_grads),expected,rtol=1e-6,atol=1e-6)
    positions=np.arange(8)[:,None]; dims=np.arange(8)[None,:]
    angles=positions/10000**(2*(dims//2)/8)
    expected=np.where(dims%2==0,np.sin(angles),np.cos(angles))
    np.testing.assert_allclose(host(layer.positional_encoding),expected,atol=.004)
    assert y.dtype==AMP_TYPE


def test_bf16_adam_master_updates_and_state_validation():
    layer=Dense(16,8)
    p=layer.parameters[0]
    p.grad[...] = .25
    initial=host(p.master)
    entry=(p.master,p.grad,p.moment,p.variance,True,p.data)
    opt=Adam(); opt.step([entry,entry],.001,.01,1,2)
    expected=initial-.001*(.125/(.125+1e-7)+.01*initial)
    np.testing.assert_allclose(host(p.master),expected,atol=1e-7,rtol=1e-6)
    np.testing.assert_array_equal(host(p.data),host(p.master.astype(AMP_TYPE)))
    assert not bool(xp.any(p.grad))
    assert len(opt.entries)==1
    # The same BF16 storage cannot serve two different masters.
    with pytest.raises(ValueError,match='overlap'):
        opt.step([entry,(p.master.copy(),p.grad.copy(),p.moment.copy(),p.variance.copy(),True,p.data)],.001,0,1,1)


@pytest.mark.parametrize('shared', [False,True])
def test_chunked_attention_dtype_gradients_and_cache(shared):
    layer=MultiHeadAttention(16,12,4,num_kv_heads=2,head_dim=4,decoder=True,
                             qk_norm=True,v_norm=True,kv_shared=shared,chunk_size=2,
                             sliding_window=4,rope=RotaryEmbedding(4,12),bias=True)
    x=random((2,6,16)); g=random(x.shape,4)
    y=layer.forward(x)
    assert all(a.dtype==AMP_TYPE for a in (y,layer.query,layer.key,layer.value))
    dx=layer.backward(g)
    assert dx.dtype==AMP_TYPE
    assert all(p.grad.dtype==FLOAT_TYPE and bool(xp.isfinite(p.grad).all()) for p in layer.parameters)
    layer.set_eval(True)
    full=layer.forward(x)
    layer.start_cache(2,12,2)
    try:
        chunks=[layer.forward(x[:,i:i+2]) for i in range(0,6,2)]
        assert layer.cache.keys.dtype==layer.cache.values.dtype==AMP_TYPE
        np.testing.assert_allclose(host(xp.concatenate(chunks,axis=1)),host(full),atol=.08,rtol=.08)
    finally:
        layer.stop_cache()


def test_convolution_and_vit_dtype_paths():
    conv=Convolution((2,4,4),3,(3,3),padding=(1,1))
    x=random((2,2,4,4)); y=conv.forward(x)
    assert y.dtype==AMP_TYPE
    assert conv.backward(random(y.shape,4)).dtype==AMP_TYPE
    projector=VitProjector((2,4,4),(2,2),8)
    head=VitMLPHead(8,3)
    tokens=projector.forward(x); y=head.forward(tokens)
    assert tokens.dtype==y.dtype==AMP_TYPE
    assert projector.backward(head.backward(random(y.shape,5))).dtype==AMP_TYPE


def test_tiny_gemma_training_and_inference_storage():
    config=dict(vocab_size=32,context_length=12,num_layers=2,embed_dim=16,
                num_heads=4,head_dim=4,num_kv_heads=2,global_head_dim=8,
                num_global_kv_heads=1,ffn_multiplier=2,sliding_window=4,chunk_size=2)
    model=gemma_gpt(**config)
    tokens=xp.asarray([[1,2,3,2,4]])
    ce=CrossEntropy(32)
    losses=[]
    for step in range(1,9):
        y=model.predict(tokens)
        losses.append(float(ce.loss(y,tokens)))
        model._backward(ce.gradients(y,tokens))
        model._update(.002,0,step,tokens.size)
    assert np.isfinite(losses).all() and losses[-1]<losses[0]
    with inference_mode(), empty_weights():
        inference=gemma_gpt(**config)
    for layer in inference.layers:
        for p in layer.parameters:
            assert p.master is p.grad is p.moment is p.variance is None
    # Copy through matching parameter lists, preserving tied ownership.
    for a,b in zip(model.layers,inference.layers):
        for p,q in zip(a.parameters,b.parameters):
            q.data[...] = p.data
    for a,b in zip(model.layers,inference.layers):
        if hasattr(a,'layer_scalar'):
            b.layer_scalar[...] = a.layer_scalar
    inference.set_eval(True)
    y=inference.predict(tokens)
    assert y.dtype==FLOAT_TYPE and bool(xp.isfinite(y).all())
    generated=inference.generate(tokens,2,temperature=0,step=2)
    assert generated.shape==(1,2)


def test_bf16_checkpoint_gpu_roundtrip(checkpoint_factory):
    from checkpoint import Checkpoint, target_array
    directory,values=checkpoint_factory('BF16')
    checkpoint=Checkpoint(directory,dtype='bfloat16')
    model,report=checkpoint.load_model(chunk_size=2)
    assert not report['storage_estimated']
    for name,spec in checkpoint.specs.items():
        actual=target_array(model,spec.target)
        source=values[name].T if spec.transpose else values[name]
        if actual.dtype==AMP_TYPE:
            np.testing.assert_array_equal(xp.asnumpy(actual).view(np.uint16),source.view(np.uint16))
        else:
            np.testing.assert_array_equal(host(actual),source.astype(np.float32))
    y=model.predict(xp.asarray([[1,2,3]]))
    assert bool(xp.isfinite(y).all())


def test_bf16_pooling_and_float_scalar_boundaries():
    from layers import AveragePool
    layer = AveragePool(2,3)
    x = random((2,3,4,6))
    y = layer.forward(x)
    dx = layer.backward(xp.ones_like(y))
    assert y.dtype == dx.dtype == AMP_TYPE
    np.testing.assert_array_equal(host(dx), np.full(x.shape, AMP_TYPE.type(1/6), dtype=FLOAT_TYPE))
    dropout = Dropout(.3)
    y = dropout.forward(x)
    assert y.dtype == dropout.backward(xp.ones_like(y)).dtype == AMP_TYPE
    for op in [Softcap(np.float64(3.7)), SoftMax(np.float64(.7), fused_loss=False)]:
        y = op.forward(x)
        assert y.dtype == op.backward(xp.ones_like(y)).dtype == AMP_TYPE
