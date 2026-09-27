"""BF16 boundaries with FP32 arithmetic inside CUDA kernels.

No full-sized FP32 cast buffers are needed for norms, RoPE, activations,
softmax, dropout, or embedding scatter. Small norm statistics and training
softmax probabilities deliberately remain FP32. The ordinary FP32 layer paths
remain the reference implementation.
"""
from functools import lru_cache
import math
import numpy as np
from backend import xp, FLOAT_TYPE, AMP_TYPE

# Stateless helpers share immutable scalars, initialized once at import.
_ZERO = FLOAT_TYPE(0)
_ONE = FLOAT_TYPE(1)
_NEGATIVE_ONE = FLOAT_TYPE(-1)


def _float_scalar(value):
    """Layer configuration already supplies typed scalars; convert external inputs only."""
    return value if isinstance(value, FLOAT_TYPE) else FLOAT_TYPE(value)


def _array_scalar(value, dtype):
    return value if isinstance(value, dtype.type) else dtype.type(value)


@lru_cache(maxsize=128)
def _reduction_count(shape, axes):
    # BatchNorm's reduction size changes with batch/spatial shape.
    return FLOAT_TYPE(math.prod(shape[a] for a in axes))


def is_bf16(x):
    return x.dtype == AMP_TYPE


@lru_cache(None)
def _elementwise(inputs, outputs, code, name):
    return xp.ElementwiseKernel(inputs, outputs, code, 'torchless_' + name)


@lru_cache(None)
def _reduce(inputs, expression, name):
    return xp.ReductionKernel(inputs, 'float32 y', expression, 'a + b',
                              'y = a', '0', 'torchless_' + name,
                              reduce_type='float')


def scale(x, factor, dtype=None):
    """Scale and store in the requested dtype in one pass."""
    dtype = x.dtype if dtype is None else dtype
    if xp is np:
        return (x * _array_scalar(factor, x.dtype)).astype(dtype, copy=False)
    out = xp.empty(x.shape, dtype=dtype)
    _elementwise('T x, float32 factor', 'U y', 'y = U(float(x) * factor);',
                 'scale')(x, _float_scalar(factor), out)
    return out


def dropout(x, mask, factor, dtype=None):
    dtype = x.dtype if dtype is None else dtype
    if mask is None:
        return x if x.dtype == dtype else scale(x, _ONE, dtype)
    if xp is np:
        return (x * mask * _array_scalar(factor, x.dtype)).astype(dtype, copy=False)
    out = xp.empty(x.shape, dtype=dtype)
    _elementwise('T x, bool keep, float32 factor', 'U y',
                 'y = keep ? U(float(x) * factor) : U(0);',
                 'dropout')(x, mask, _float_scalar(factor), out)
    return out


def norm_forward(x, gamma, beta, axes, eps, rms=False, stats=None, count=None):
    """Fused load conversions in reductions; affine output stores BF16 directly."""
    count = _reduction_count(x.shape, axes) if count is None else count
    if stats is None:
        mean = _ZERO if rms else _reduce(
            'T x', 'float(x)', 'norm_sum')(x, axis=axes, keepdims=True) / count
        variance = _reduce('T x, float32 mean',
                           '(float(x)-mean)*(float(x)-mean)', 'norm_variance')(
                               x, mean, axis=axes, keepdims=True) / count
    else:
        mean, variance = stats
    inv = _ONE / xp.sqrt(variance + _float_scalar(eps))
    out = xp.empty_like(x)
    _elementwise('T x, float32 mean, float32 inv, float32 gamma, float32 beta',
                 'T y', 'y = T((float(x)-mean)*inv*gamma + beta);',
                 'norm_forward')(x, mean, inv, gamma, beta, out)
    return out, mean, variance, inv


def norm_backward(g, x, gamma, mean, inv, axes, param_axes, rms=False,
                  param_keepdims=False, with_scale=True, with_bias=True, count=None):
    count = _reduction_count(x.shape, axes) if count is None else count
    args = 'G g, T x, float32 mean, float32 inv'
    dgamma = _reduce(args, 'float(g)*(float(x)-mean)*inv', 'norm_dgamma')(
        g, x, mean, inv, axis=param_axes, keepdims=param_keepdims) if with_scale else None
    dbeta = _reduce('G g', 'float(g)', 'norm_dbeta')(
        g, axis=param_axes, keepdims=param_keepdims) if with_bias else None
    args += ', float32 gamma'
    dot = _reduce(args, 'float(g)*gamma*(float(x)-mean)*inv', 'norm_dot')(
        g, x, mean, inv, gamma, axis=axes, keepdims=True) / count
    avg = _ZERO if rms else _reduce(
        'G g, float32 gamma', 'float(g)*gamma', 'norm_avg')(
            g, gamma, axis=axes, keepdims=True) / count
    dx = xp.empty_like(x)
    _elementwise(args + ', float32 dot, float32 avg', 'T dx',
                 'dx = T((float(g)*gamma - avg - (float(x)-mean)*inv*dot)*inv);',
                 'norm_backward')(g, x, mean, inv, gamma, dot, avg, dx)
    return dx, dgamma, dbeta


def rope(x, cos, sin, rotary_dim, backward=False, dtype=None):
    """Index paired dimensions directly, including strided head-split inputs."""
    dtype = x.dtype if dtype is None else dtype
    out = xp.empty(x.shape, dtype=dtype)
    width = x.shape[-1]
    # raw CArray indexing is logical and honors the input's strides.
    _elementwise('raw T x, raw float32 c, raw float32 s, int64 width, '
                 'int64 rotary, int64 length, float32 direction', 'U y', r'''
        long long d = i % width;
        if (d >= rotary) { y = U(float(x[i])); }
        else {
            long long half = rotary / 2;
            long long pair = d < half ? d + half : d - half;
            float other = float(x[i - d + pair]) * (d < half ? -1.f : 1.f);
            long long pos = ((i / width) % length) * rotary + d;
            y = U(float(x[i])*c[pos] + direction*other*s[pos]);
        }
    ''', 'rope')(x, cos, sin, np.int64(width), np.int64(rotary_dim),
                np.int64(x.shape[-2]), _NEGATIVE_ONE if backward else _ONE, out)
    return out


def scatter_embedding(dest, ids, gradient, factor):
    """Convert/scale BF16 gradient values in registers, then FP32 atomicAdd."""
    if xp is np or not is_bf16(gradient):
        xp.add.at(dest, ids, gradient * _array_scalar(factor, gradient.dtype))
        return
    _elementwise('T g, raw I ids, int64 width, int64 vocab, float32 factor',
                 'raw float32 dest',
                 'long long row = ((ids[i / width] % vocab) + vocab) % vocab; '
                 'atomicAdd(&dest[row*width + i % width], float(g)*factor);',
                 'embedding_scatter')(gradient, ids, np.int64(gradient.shape[-1]),
                                      np.int64(dest.shape[0]), _float_scalar(factor), dest)


def sinusoidal(length, width, dtype):
    out = xp.empty((length, width), dtype=dtype)
    _elementwise('int64 width', 'T y', r'''
        long long d = i % width;
        float angle = float(i / width) / powf(10000.f, float(2*(d/2))/float(width));
        y = T(d % 2 ? cosf(angle) : sinf(angle));
    ''', 'sinusoidal')(np.int64(width), out)
    return out


_ACTIVATIONS = {
    'relu': ('fmaxf(v,0.f)', 'v > 0.f ? 1.f : 0.f'),
    'gelu': ('0.5f*v*(1.f+erff(v*0.7071067811865475f))',
             '0.5f*(1.f+erff(v*0.7071067811865475f)) + v*expf(-0.5f*v*v)*0.3989422804014327f'),
    'gelutanh': ('0.5f*v*(1.f+tanhf(0.7978845608028654f*(v+0.044715f*v*v*v)))',
                 '0.5f*(1.f+t) + 0.5f*v*(1.f-t*t)*0.7978845608028654f*(1.f+0.134145f*v*v)'),
    'silu': ('v*s', 's+v*s*(1.f-s)'),
    'softcap': ('cap*tanhf(v/cap)', '1.f-tanhf(v/cap)*tanhf(v/cap)'),
}


def activation(x, kind, gradient=None, cap=_ONE):
    expression = _ACTIVATIONS[kind][gradient is not None]
    preamble = 'float v = float(x);'
    if kind == 'gelutanh' and gradient is not None:
        preamble += 'float t = tanhf(0.7978845608028654f*(v+0.044715f*v*v*v));'
    if kind == 'silu':
        preamble += 'float s = (1.f+tanhf(v*0.5f))*0.5f;'
    args = 'T x, float32 cap'
    values = [x, _float_scalar(cap)]
    if gradient is not None:
        args += ', G g'
        values.append(gradient)
        expression = '(' + expression + ')*float(g)'
    return _elementwise(args, 'T y', preamble + 'y = T(' + expression + ');',
                        kind + ('_backward' if gradient is not None else '_forward'))(*values)


_SOFTMAX_SOURCE = r'''
#include <cuda_bf16.h>
extern "C" __global__ void softmax_forward(
 const INPUT* x, OUTPUT* y, float* saved, const bool* keep,
 long long width, long long qlength, long long qstart, long long kstart,
 long long window, int causal, float scale, float dropout_scale, int save) {
    __shared__ float scratch[256];
    long long row = blockIdx.x;
    long long q = qstart + row % qlength;
    float mx = -3.402823466e38f;
    for (long long j=threadIdx.x; j<width; j+=blockDim.x) {
        long long k = kstart+j;
        float v = float(x[row*width+j])*scale;
        if (causal && (k>q || (window>0 && q-k>=window))) v += -1.e9f;
        mx = fmaxf(mx,v);
    }
    scratch[threadIdx.x]=mx; __syncthreads();
    for (int s=128;s;s/=2) { if(threadIdx.x<s) scratch[threadIdx.x]=fmaxf(scratch[threadIdx.x],scratch[threadIdx.x+s]); __syncthreads(); }
    mx=scratch[0]; __syncthreads();
    float sum=0.f;
    for(long long j=threadIdx.x;j<width;j+=blockDim.x) {
        long long k=kstart+j;
        float v=float(x[row*width+j])*scale;
        if(causal && (k>q || (window>0 && q-k>=window))) v+=-1.e9f;
        sum+=expf(v-mx);
    }
    scratch[threadIdx.x]=sum; __syncthreads();
    for(int s=128;s;s/=2) { if(threadIdx.x<s) scratch[threadIdx.x]+=scratch[threadIdx.x+s]; __syncthreads(); }
    sum=scratch[0];
    for(long long j=threadIdx.x;j<width;j+=blockDim.x) {
        long long k=kstart+j, idx=row*width+j;
        float v=float(x[idx])*scale;
        if(causal && (k>q || (window>0 && q-k>=window))) v+=-1.e9f;
        float p=expf(v-mx)/sum;
        if(save) saved[idx]=p;
        y[idx]=OUTPUT(keep ? (keep[idx] ? p*dropout_scale : 0.f) : p);
    }
}
extern "C" __global__ void softmax_backward(
 const INPUT* g, const float* p, const bool* keep, OUTPUT* dx,
 long long width, float dropout_scale, float scale) {
    __shared__ float scratch[256];
    long long row=blockIdx.x;
    float dot=0.f;
    for(long long j=threadIdx.x;j<width;j+=blockDim.x) {
        long long idx=row*width+j;
        float v=float(g[idx]);
        if(keep) v=keep[idx] ? v*dropout_scale : 0.f;
        dot+=v*p[idx];
    }
    scratch[threadIdx.x]=dot; __syncthreads();
    for(int s=128;s;s/=2) { if(threadIdx.x<s) scratch[threadIdx.x]+=scratch[threadIdx.x+s]; __syncthreads(); }
    dot=scratch[0];
    for(long long j=threadIdx.x;j<width;j+=blockDim.x) {
        long long idx=row*width+j;
        float v=float(g[idx]);
        if(keep) v=keep[idx] ? v*dropout_scale : 0.f;
        dx[idx]=OUTPUT(p[idx]*(v-dot)*scale);
    }
}
'''


@lru_cache(None)
def _softmax_module(input_bf16, output_bf16):
    source = _SOFTMAX_SOURCE.replace('INPUT', '__nv_bfloat16' if input_bf16 else 'float')
    source = source.replace('OUTPUT', '__nv_bfloat16' if output_bf16 else 'float')
    return xp.RawModule(code=source, options=('--std=c++11',))


def softmax(x, dtype, *, scale=_ONE, mask=None, dropout_scale=_ONE,
            save=False, causal=False, qstart=0, kstart=0, window=None):
    x = xp.ascontiguousarray(x)
    out = xp.empty(x.shape, dtype=dtype)
    saved = xp.empty(x.shape if save else (0,), dtype=FLOAT_TYPE)
    kernel = _softmax_module(is_bf16(x), np.dtype(dtype) == AMP_TYPE).get_function('softmax_forward')
    width = x.shape[-1]
    rows = x.size // width
    if rows:
        kernel((rows,), (256,), (x, out, saved, mask if mask is not None else np.uint64(0),
            np.int64(width), np.int64(x.shape[-2] if x.ndim > 1 else 1),
            np.int64(qstart), np.int64(kstart), np.int64(window or 0),
            np.int32(causal), _float_scalar(scale), _float_scalar(dropout_scale), np.int32(save)))
    return out, saved if save else None


def softmax_backward(g, p, dtype, *, mask=None, dropout_scale=_ONE, scale=_ONE):
    g = xp.ascontiguousarray(g)
    out = xp.empty(g.shape, dtype=dtype)
    kernel = _softmax_module(is_bf16(g), np.dtype(dtype) == AMP_TYPE).get_function('softmax_backward')
    width = g.shape[-1]
    if g.size:
        kernel((g.size // width,), (256,), (g, p, mask if mask is not None else np.uint64(0),
               out, np.int64(width), _float_scalar(dropout_scale), _float_scalar(scale)))
    return out
