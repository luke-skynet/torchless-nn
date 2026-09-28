"""Shared FP32/BF16 CUDA kernel operations, grouped by layer type.

Kernels compute in FP32 and store the requested dtype. Normalization statistics,
parameter gradients, and saved softmax probabilities remain FP32. NumPy reference
implementations live in the layer classes.
"""
from functools import lru_cache
import numpy as np

from backend import xp, FLOAT_TYPE, AMP_TYPE


# precision types and kernel builders

_ONE = FLOAT_TYPE(1)
_NEGATIVE_ONE = FLOAT_TYPE(-1)


def _float_scalar(value):
    """Layer configuration already supplies typed scalars; convert external inputs only."""
    return value if isinstance(value, FLOAT_TYPE) else FLOAT_TYPE(value)


def _array_scalar(value, dtype):
    return value if isinstance(value, dtype.type) else dtype.type(value)


def is_bf16(array):
    return array.dtype == AMP_TYPE


def use_kernels(array):
    """Use shared CUDA kernels only for their supported storage dtypes."""
    return xp is not np and (array.dtype == np.dtype(FLOAT_TYPE) or is_bf16(array))


@lru_cache(None)
def _elementwise(inputs, outputs, code, name):
    return xp.ElementwiseKernel(inputs, outputs, code, 'torchless_' + name)


@lru_cache(None)
def _reduce(inputs, expression, name):
    return xp.ReductionKernel(inputs, 'float32 y', expression, 'a + b',
                              'y = a', '0', 'torchless_' + name,
                              reduce_type = 'float')


def _element_strides(array):
    return tuple(np.int64(stride // array.itemsize) for stride in array.strides)


# scaling and dropout

def scale(input, factor, dtype = None):
    """Scale and store in the requested dtype in one pass."""
    dtype = input.dtype if dtype is None else dtype
    if xp is np:
        return (input * _array_scalar(factor, input.dtype)).astype(dtype, copy = False)
    output = xp.empty(input.shape, dtype = dtype)
    _elementwise('T x, float32 factor', 'U y', 'y = U(float(x) * factor);',
                 'scale')(input, _float_scalar(factor), output)
    return output


def dropout(input, mask, factor, dtype = None):
    dtype = input.dtype if dtype is None else dtype
    if mask is None:
        return input if input.dtype == dtype else scale(input, _ONE, dtype)
    if xp is np:
        return (input * mask * _array_scalar(factor, input.dtype)).astype(dtype, copy = False)
    output = xp.empty(input.shape, dtype = dtype)
    _elementwise('T x, bool keep, float32 factor', 'U y',
                 'y = keep ? U(float(x) * factor) : U(0);',
                 'dropout')(input, mask, _float_scalar(factor), output)
    return output


# normalization: shared affine output

def _centered_affine(input, gamma, beta, mean, inv_std):
    output = xp.empty_like(input)
    _elementwise('T x, float32 mean, float32 inv, float32 gamma, float32 beta',
                 'T y', 'y = T((float(x)-mean)*inv*gamma + beta);',
                 'centered_norm_affine')(input, mean, inv_std, gamma, beta, output)
    return output


# batch normalization

# BatchNorm reduces N*H*W values into only C outputs. Generic ReductionKernel
# scheduling can leave just C/32 blocks for broadcast multi-input reductions.
# Split that work explicitly; scratch is two floats per (channel, chunk), and
# neither activations nor gradients need a full-sized cast/contiguous copy.
_BATCH_NORM_BLOCK_SIZE = 256
_BATCH_NORM_CHUNK_SIZE = 4096
_BATCH_NORM_SOURCE = r'''
#include <cuda_bf16.h>
#define BLOCK_SIZE 256
#define CHUNK_SIZE 4096

struct Moments {
    long long count;
    float mean, m2;
};

__device__ Moments merge_moments(Moments a, Moments b) {
    if (!a.count) return b;
    if (!b.count) return a;
    long long count = a.count + b.count;
    float delta = b.mean - a.mean;
    float weight = float(b.count) / float(count);
    Moments result = {count, a.mean + delta * weight,
        a.m2 + b.m2 + delta * delta * (float(a.count) * weight)};
    return result;
}

// k flattens the reduction axes (N,H,W); c is the unreduced channel.
// Strides are in elements, including zero/negative strides for views.
__device__ long long offset(long long k, long long c, long long spatial,
                            long long width, long long s0, long long s1,
                            long long s2, long long s3) {
    long long n = k / spatial, hw = k % spatial;
    if (s3 == 1 && s2 == width)
        return n * s0 + c * s1 + hw;
    return n * s0 + c * s1 + (hw / width) * s2 + (hw % width) * s3;
}

extern "C" __global__ void batch_norm_moments_partial(
    const INPUT* x, float* partial, long long count, long long channels,
    long long spatial, long long width, long long chunks,
    long long s0, long long s1, long long s2, long long s3) {
    long long c = blockIdx.x / chunks, chunk = blockIdx.x % chunks;
    long long end = (chunk + 1) * CHUNK_SIZE;
    if (end > count) end = count;
    Moments value = {0, 0.f, 0.f};
    for (long long k = chunk * CHUNK_SIZE + threadIdx.x; k < end; k += BLOCK_SIZE) {
        float v = float(x[offset(k, c, spatial, width, s0, s1, s2, s3)]);
        ++value.count;
        float delta = v - value.mean;
        value.mean += delta / float(value.count);
        value.m2 += delta * (v - value.mean);
    }
    __shared__ Moments scratch[BLOCK_SIZE];
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s; s /= 2) {
        if (threadIdx.x < s)
            scratch[threadIdx.x] = merge_moments(scratch[threadIdx.x], scratch[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        long long i = c * chunks + chunk;
        partial[i] = scratch[0].mean;
        partial[channels * chunks + i] = scratch[0].m2;
    }
}

extern "C" __global__ void batch_norm_moments_finish(
    const float* partial, float* mean, float* variance, float* inv,
    long long count, long long channels, long long chunks, float eps) {
    long long c = blockIdx.x;
    Moments value = {0, 0.f, 0.f};
    for (long long chunk = threadIdx.x; chunk < chunks; chunk += BLOCK_SIZE) {
        long long size = count - chunk * CHUNK_SIZE;
        if (size > CHUNK_SIZE) size = CHUNK_SIZE;
        long long i = c * chunks + chunk;
        Moments other = {size, partial[i], partial[channels * chunks + i]};
        value = merge_moments(value, other);
    }
    __shared__ Moments scratch[BLOCK_SIZE];
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s; s /= 2) {
        if (threadIdx.x < s)
            scratch[threadIdx.x] = merge_moments(scratch[threadIdx.x], scratch[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mean[c] = scratch[0].mean;
        // Population variance matches the existing BatchNorm/running-var rule.
        float var = scratch[0].m2 / float(count);
        variance[c] = var;
        inv[c] = 1.f / sqrtf(var + eps);
    }
}

extern "C" __global__ void batch_norm_grads_partial(
    const INPUT* x, const GRADIENT* g, const float* mean, const float* inv,
    float* partial, long long count, long long channels,
    long long spatial, long long width, long long chunks,
    long long x0, long long x1, long long x2, long long x3,
    long long g0, long long g1, long long g2, long long g3) {
    long long c = blockIdx.x / chunks, chunk = blockIdx.x % chunks;
    long long end = (chunk + 1) * CHUNK_SIZE;
    if (end > count) end = count;
    float dg = 0.f, db = 0.f, m = mean[c], r = inv[c];
    for (long long k = chunk * CHUNK_SIZE + threadIdx.x; k < end; k += BLOCK_SIZE) {
        float v = float(x[offset(k, c, spatial, width, x0, x1, x2, x3)]);
        float grad = float(g[offset(k, c, spatial, width, g0, g1, g2, g3)]);
        dg += grad * ((v - m) * r);
        db += grad;
    }
    __shared__ float gamma_sum[BLOCK_SIZE], beta_sum[BLOCK_SIZE];
    gamma_sum[threadIdx.x] = dg;
    beta_sum[threadIdx.x] = db;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s; s /= 2) {
        if (threadIdx.x < s) {
            gamma_sum[threadIdx.x] += gamma_sum[threadIdx.x + s];
            beta_sum[threadIdx.x] += beta_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        long long i = c * chunks + chunk;
        partial[i] = gamma_sum[0];
        partial[channels * chunks + i] = beta_sum[0];
    }
}

extern "C" __global__ void batch_norm_grads_finish(
    const float* partial, float* dgamma, float* dbeta,
    long long channels, long long chunks) {
    long long c = blockIdx.x;
    float dg = 0.f, db = 0.f;
    for (long long chunk = threadIdx.x; chunk < chunks; chunk += BLOCK_SIZE) {
        long long i = c * chunks + chunk;
        dg += partial[i];
        db += partial[channels * chunks + i];
    }
    __shared__ float gamma_sum[BLOCK_SIZE], beta_sum[BLOCK_SIZE];
    gamma_sum[threadIdx.x] = dg;
    beta_sum[threadIdx.x] = db;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s; s /= 2) {
        if (threadIdx.x < s) {
            gamma_sum[threadIdx.x] += gamma_sum[threadIdx.x + s];
            beta_sum[threadIdx.x] += beta_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dgamma[c] = gamma_sum[0];
        dbeta[c] = beta_sum[0];
    }
}
'''


@lru_cache(None)
def _batch_norm_module(input_bf16, gradient_bf16):
    source = _BATCH_NORM_SOURCE.replace('INPUT', '__nv_bfloat16' if input_bf16 else 'float')
    source = source.replace('GRADIENT', '__nv_bfloat16' if gradient_bf16 else 'float')
    source = source.replace('#define BLOCK_SIZE 256',
                            f'#define BLOCK_SIZE {_BATCH_NORM_BLOCK_SIZE}')
    source = source.replace('#define CHUNK_SIZE 4096',
                            f'#define CHUNK_SIZE {_BATCH_NORM_CHUNK_SIZE}')
    return xp.RawModule(code = source, options = ('--std=c++11',))


def _batch_norm_layout(input):
    if input.ndim != 4 or not all(input.shape):
        raise ValueError('BatchNorm kernels require a nonempty NCHW array')
    if not use_kernels(input):
        raise TypeError('BatchNorm kernels require CUDA FP32 or BF16 inputs')
    n, channels, height, width = input.shape
    spatial_size = height * width
    reduction_size = n * spatial_size
    num_chunks = ((reduction_size + _BATCH_NORM_CHUNK_SIZE - 1)
                  // _BATCH_NORM_CHUNK_SIZE)
    return reduction_size, channels, spatial_size, width, num_chunks


def batch_norm_forward(input, gamma, beta, eps, statistics = None):
    """NCHW BatchNorm: split Welford statistics, then shared affine output."""
    reduction_size, channels, spatial_size, width, num_chunks = _batch_norm_layout(input)
    if statistics is not None:
        # Evaluation uses running statistics; no reduction/scratch allocation.
        mean, variance = statistics
        inv_std = _ONE / xp.sqrt(variance + _float_scalar(eps))
        return _centered_affine(input, gamma, beta, mean, inv_std), mean, variance, inv_std
    kernel_module = _batch_norm_module(is_bf16(input), is_bf16(input))
    partial_statistics = xp.empty((2, channels, num_chunks), dtype = FLOAT_TYPE)
    statistic_shape = (1, channels, 1, 1)
    mean, variance, inv_std = (
        xp.empty(statistic_shape, dtype = FLOAT_TYPE) for _ in range(3))
    partial_kernel = kernel_module.get_function('batch_norm_moments_partial')
    finish_kernel = kernel_module.get_function('batch_norm_moments_finish')
    partial_kernel((channels * num_chunks,), (_BATCH_NORM_BLOCK_SIZE,),
        (input, partial_statistics,
         *map(np.int64, (reduction_size, channels, spatial_size, width, num_chunks)),
         *_element_strides(input)))
    finish_kernel((channels,), (_BATCH_NORM_BLOCK_SIZE,),
        (partial_statistics, mean, variance, inv_std,
         *map(np.int64, (reduction_size, channels, num_chunks)),
         _float_scalar(eps)))
    return _centered_affine(input, gamma, beta, mean, inv_std), mean, variance, inv_std


def batch_norm_backward(gradient, input, gamma, mean, inv_std):
    """Reduce dgamma/dbeta together and reuse them for the input gradient."""
    reduction_size, channels, spatial_size, width, num_chunks = _batch_norm_layout(input)
    if gradient.shape != input.shape:
        raise ValueError('BatchNorm gradient shape must match its input')
    if not use_kernels(gradient):
        raise TypeError('BatchNorm kernels require CUDA FP32 or BF16 gradients')
    kernel_module = _batch_norm_module(is_bf16(input), is_bf16(gradient))
    partial_gradients = xp.empty((2, channels, num_chunks), dtype = FLOAT_TYPE)
    statistic_shape = (1, channels, 1, 1)
    gamma_grads, beta_grads = (
        xp.empty(statistic_shape, dtype = FLOAT_TYPE) for _ in range(2))
    partial_kernel = kernel_module.get_function('batch_norm_grads_partial')
    finish_kernel = kernel_module.get_function('batch_norm_grads_finish')
    partial_kernel((channels * num_chunks,), (_BATCH_NORM_BLOCK_SIZE,),
        (input, gradient, mean, inv_std, partial_gradients,
         *map(np.int64, (reduction_size, channels, spatial_size, width, num_chunks)),
         *_element_strides(input), *_element_strides(gradient)))
    finish_kernel((channels,), (_BATCH_NORM_BLOCK_SIZE,),
        (partial_gradients, gamma_grads, beta_grads,
         *map(np.int64, (channels, num_chunks))))
    input_gradient = xp.empty_like(input)
    _elementwise('G g, T x, float32 gamma, float32 mean, float32 inv, '
                 'float32 dg, float32 db, float32 count', 'T dx',
                 'dx = T(gamma * inv * (float(g) - db/count '
                 '- ((float(x)-mean)*inv) * (dg/count)));',
                 'batch_norm_backward')(
                     gradient, input, gamma, mean, inv_std,
                     gamma_grads, beta_grads, FLOAT_TYPE(reduction_size), input_gradient)
    return input_gradient, gamma_grads, beta_grads


# layer and RMS normalization

# One block per row for statistics/affine and input gradients. Parameter gradients
# reduce along the other axis, using coalesced column tiles and bounded scratch.
_ROW_NORM_BLOCK_SIZE = 256
_ROW_NORM_MAX_PARTIALS = 128
_ROW_NORM_SOURCE = r'''
#include <cuda_bf16.h>
#define CENTERED IS_CENTERED
#define BLOCK_SIZE 256

// All threads call this together. The final barrier also makes scratch reusable.
__device__ float row_sum(float v, float* scratch) {
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s; s /= 2) {
        if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
        __syncthreads();
    }
    float result = scratch[0];
    __syncthreads();
    return result;
}

ROW_OFFSET_FUNCTION

extern "C" __global__ void row_norm_forward(
    const INPUT* x, const float* gamma, const float* beta, INPUT* y,
    float* mean, float* inv, long long width, float eps, float scalar_gamma
    LAYOUT_ARGS) {
    long long row = blockIdx.x, xb = X_ROW;
    __shared__ float scratch[BLOCK_SIZE];
    float m = 0.f;
    if (CENTERED) {
        // Shift before summation to avoid losing small variations on large offsets.
        float anchor = float(x[xb]), sum = 0.f;
        for (long long d = threadIdx.x; d < width; d += BLOCK_SIZE)
            sum += float(x[xb + d * X_STEP]) - anchor;
        m = anchor + row_sum(sum, scratch) / float(width);
    }
    float squares = 0.f;
    for (long long d = threadIdx.x; d < width; d += BLOCK_SIZE) {
        float v = float(x[xb + d * X_STEP]) - m;
        squares += v * v;
    }
    float r = 1.f / sqrtf(row_sum(squares, scratch) / float(width) + eps);
    if (threadIdx.x == 0) {
        if (CENTERED) mean[row] = m;
        inv[row] = r;
    }
    for (long long d = threadIdx.x; d < width; d += BLOCK_SIZE) {
        float scale = gamma ? gamma[d] : scalar_gamma;
        float v = (float(x[xb + d * X_STEP]) - m) * r;
        y[row * width + d] = INPUT(v * scale + (CENTERED ? beta[d] : 0.f));
    }
}

extern "C" __global__ void row_norm_backward(
    const INPUT* x, const GRADIENT* g, const float* gamma,
    const float* mean, const float* inv, INPUT* dx,
    long long width, float scalar_gamma LAYOUT_ARGS) {
    long long row = blockIdx.x, xb = X_ROW, gb = G_ROW;
    __shared__ float scratch[BLOCK_SIZE];
    float m = CENTERED ? mean[row] : 0.f, r = inv[row];
    float sum = 0.f, dot = 0.f;
    for (long long d = threadIdx.x; d < width; d += BLOCK_SIZE) {
        float v = (float(x[xb + d * X_STEP]) - m) * r;
        float grad = float(g[gb + d * G_STEP]) * (gamma ? gamma[d] : scalar_gamma);
        sum += grad;
        dot += grad * v;
    }
    float avg = 0.f;
    if (CENTERED) avg = row_sum(sum, scratch) / float(width);
    dot = row_sum(dot, scratch) / float(width);
    for (long long d = threadIdx.x; d < width; d += BLOCK_SIZE) {
        float v = (float(x[xb + d * X_STEP]) - m) * r;
        float grad = float(g[gb + d * G_STEP]) * (gamma ? gamma[d] : scalar_gamma);
        dx[row * width + d] = INPUT((grad - avg - v * dot) * r);
    }
}

extern "C" __global__ void row_norm_params_partial(
    const INPUT* x, const GRADIENT* g, const float* mean, const float* inv,
    float* dg, float* db, long long width, long long rows,
    long long rows_per_chunk LAYOUT_ARGS) {
    long long d = (long long)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (d >= width) return;
    long long end = ((long long)blockIdx.y + 1) * rows_per_chunk;
    if (end > rows) end = rows;
    float gamma_sum = 0.f, beta_sum = 0.f;
    for (long long row = (long long)blockIdx.y * rows_per_chunk; row < end; ++row) {
        long long xb = X_ROW, gb = G_ROW;
        float grad = float(g[gb + d * G_STEP]);
        float m = CENTERED ? mean[row] : 0.f;
        gamma_sum += grad * ((float(x[xb + d * X_STEP]) - m) * inv[row]);
        if (CENTERED) beta_sum += grad;
    }
    long long out = (long long)blockIdx.y * width + d;
    dg[out] = gamma_sum;
    if (CENTERED) db[out] = beta_sum;
}

extern "C" __global__ void row_norm_params_finish(
    const float* partial_gamma, const float* partial_beta,
    float* dg, float* db, long long width, long long chunks) {
    long long d = (long long)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (d >= width) return;
    float gamma_sum = 0.f, beta_sum = 0.f;
    for (long long chunk = 0; chunk < chunks; ++chunk) {
        gamma_sum += partial_gamma[chunk * width + d];
        if (CENTERED) beta_sum += partial_beta[chunk * width + d];
    }
    dg[d] = gamma_sum;
    if (CENTERED) db[d] = beta_sum;
}
'''


@lru_cache(None)
def _row_norm_module(input_bf16, gradient_bf16, centered, ndim,
                     input_contiguous, gradient_contiguous):
    # Pass shape/strides by value, avoiding device metadata allocations and copies.
    # Specialize on rank and contiguity, never on shape, so variable token lengths
    # reuse compiled kernels. Negative and broadcast strides remain supported.
    sizes = [f'long long n{i}' for i in range(ndim - 1)]
    strides = [f'long long s{i}' for i in range(ndim - 1)]
    offset = ['long long result = 0;']
    for i in reversed(range(ndim - 1)):
        offset.append(f'result += (row % n{i}) * s{i}; row /= n{i};')
    signature = ', '.join(['long long row'] + sizes + strides)
    helper = f'__device__ long long row_offset({signature}) {{' + ''.join(offset) + 'return result;}'
    layout = sizes + [f'long long {prefix}{i}' for prefix in ('x', 'g') for i in range(ndim)]
    source = _ROW_NORM_SOURCE.replace('ROW_OFFSET_FUNCTION', helper)
    source = source.replace('LAYOUT_ARGS', ', ' + ', '.join(layout))
    for prefix, contiguous in (('x', input_contiguous), ('g', gradient_contiguous)):
        args = ['row'] + [f'n{i}' for i in range(ndim - 1)] + [f'{prefix}{i}' for i in range(ndim - 1)]
        source = source.replace(prefix.upper() + '_ROW',
                                'row * width' if contiguous else 'row_offset(' + ', '.join(args) + ')')
        source = source.replace(prefix.upper() + '_STEP', '1' if contiguous else f'{prefix}{ndim - 1}')
    source = source.replace('IS_CENTERED', str(int(centered)))
    source = source.replace('INPUT', '__nv_bfloat16' if input_bf16 else 'float')
    source = source.replace('GRADIENT', '__nv_bfloat16' if gradient_bf16 else 'float')
    return xp.RawModule(code=source, options=('--std=c++11',))


def _row_norm_layout(input, gradient=None, centered=False):
    if input.ndim < 1 or input.shape[-1] == 0:
        raise ValueError('Normalization kernels require a nonempty last dimension')
    if not use_kernels(input):
        raise TypeError('Normalization kernels require CUDA FP32 or BF16 inputs')
    gradient = input if gradient is None else gradient
    if gradient.shape != input.shape or not use_kernels(gradient):
        raise ValueError('Normalization gradient must match input shape and use FP32 or BF16')
    module = _row_norm_module(is_bf16(input), is_bf16(gradient), centered,
                              input.ndim, input.flags.c_contiguous, gradient.flags.c_contiguous)
    layout = (*map(np.int64, input.shape[:-1]),
              *_element_strides(input), *_element_strides(gradient))
    return module, input.shape[-1], input.size // input.shape[-1], layout


def _row_norm_scale(gamma, width):
    if isinstance(gamma, xp.ndarray):
        if gamma.shape != (width,) or gamma.dtype != np.dtype(FLOAT_TYPE):
            raise ValueError('Normalization scale must be an FP32 vector matching the last dimension')
        return xp.ascontiguousarray(gamma), _ONE
    return np.uint64(0), _float_scalar(gamma)


def _row_norm_forward(input, gamma, beta, eps):
    centered = beta is not None
    module, width, rows, layout = _row_norm_layout(input, centered=centered)
    scale, scalar = _row_norm_scale(gamma, width)
    if centered:
        if beta.shape != (width,) or beta.dtype != np.dtype(FLOAT_TYPE):
            raise ValueError('LayerNorm bias must be an FP32 vector matching the last dimension')
        beta = xp.ascontiguousarray(beta)
    output = xp.empty(input.shape, dtype=input.dtype)
    inv = xp.empty(input.shape[:-1] + (1,), dtype=FLOAT_TYPE)
    mean = xp.empty_like(inv) if centered else None
    if rows:
        module.get_function('row_norm_forward')((rows,), (_ROW_NORM_BLOCK_SIZE,),
            (input, scale, beta if centered else np.uint64(0), output,
             mean if centered else np.uint64(0), inv, np.int64(width),
             _float_scalar(eps), scalar, *layout))
    return output, mean, inv


def _row_norm_backward(gradient, input, gamma, mean, inv, with_scale):
    centered = mean is not None
    module, width, rows, layout = _row_norm_layout(input, gradient, centered)
    scale, scalar = _row_norm_scale(gamma, width)
    mean_arg = mean if centered else np.uint64(0)
    dx = xp.empty(input.shape, dtype=input.dtype)
    dg = xp.empty((width,), dtype=FLOAT_TYPE) if with_scale else None
    db = xp.empty_like(dg) if centered else None
    if not rows:
        if dg is not None:
            dg.fill(0)
        if db is not None:
            db.fill(0)
        return dx, dg, db
    module.get_function('row_norm_backward')((rows,), (_ROW_NORM_BLOCK_SIZE,),
        (input, gradient, scale, mean_arg, inv, dx, np.int64(width), scalar, *layout))
    if with_scale:
        chunks = min((rows + 127) // 128, _ROW_NORM_MAX_PARTIALS)
        partial_gamma = xp.empty((chunks, width), dtype=FLOAT_TYPE) if chunks > 1 else dg
        partial_beta = (xp.empty_like(partial_gamma) if chunks > 1 else db) if centered else np.uint64(0)
        blocks = (width + _ROW_NORM_BLOCK_SIZE - 1) // _ROW_NORM_BLOCK_SIZE
        module.get_function('row_norm_params_partial')((blocks, chunks), (_ROW_NORM_BLOCK_SIZE,),
            (input, gradient, mean_arg, inv, partial_gamma, partial_beta,
             np.int64(width), np.int64(rows), np.int64((rows + chunks - 1) // chunks), *layout))
        if chunks > 1:
            module.get_function('row_norm_params_finish')((blocks,), (_ROW_NORM_BLOCK_SIZE,),
                (partial_gamma, partial_beta, dg, db if centered else np.uint64(0),
                 np.int64(width), np.int64(chunks)))
    return dx, dg, db


def layer_norm_forward(input, gamma, beta, eps):
    """Fused row statistics and affine output; save FP32 mean/inverse deviation."""
    return _row_norm_forward(input, gamma, beta, eps)


def layer_norm_backward(gradient, input, gamma, mean, inv_std):
    """Fused input gradients plus a combined scale/bias parameter reduction."""
    return _row_norm_backward(gradient, input, gamma, mean, inv_std, True)


def rms_norm_forward(input, gamma, eps):
    """Fused second moment and scaling, with no centering or bias."""
    output, _, inv = _row_norm_forward(input, gamma, None, eps)
    return output, inv


def rms_norm_backward(gradient, input, gamma, inv_rms, with_scale=True):
    """Fused input gradients; skip parameter reduction for scale-free RMSNorm."""
    dx, dg, _ = _row_norm_backward(gradient, input, gamma, None, inv_rms, with_scale)
    return dx, dg


# activations

_ACTIVATIONS = {
    'relu': ('fmaxf(v,0.f)', 'v > 0.f ? 1.f : 0.f'),
    'gelu': ('0.5f*v*(1.f+erff(v*0.7071067811865475f))',
             '0.5f*(1.f+erff(v*0.7071067811865475f)) + v*expf(-0.5f*v*v)*0.3989422804014327f'),
    'gelutanh': ('0.5f*v*(1.f+tanhf(0.7978845608028654f*(v+0.044715f*v*v*v)))',
                 '0.5f*(1.f+t) + 0.5f*v*(1.f-t*t)*0.7978845608028654f*(1.f+0.134145f*v*v)'),
    'silu': ('v*s', 's+v*s*(1.f-s)'),
    'softcap': ('cap*tanhf(v/cap)', '1.f-tanhf(v/cap)*tanhf(v/cap)'),
}


def activation(input, kind, gradient = None, cap = _ONE):
    expression = _ACTIVATIONS[kind][gradient is not None]
    preamble = 'float v = float(x);'
    if kind == 'gelutanh' and gradient is not None:
        preamble += 'float t = tanhf(0.7978845608028654f*(v+0.044715f*v*v*v));'
    if kind == 'silu':
        preamble += 'float s = (1.f+tanhf(v*0.5f))*0.5f;'
    args = 'T x, float32 cap'
    values = [input, _float_scalar(cap)]
    if gradient is not None:
        args += ', G g'
        values.append(gradient)
        expression = '(' + expression + ')*float(g)'
    return _elementwise(args, 'T y', preamble + 'y = T(' + expression + ');',
                        kind + ('_backward' if gradient is not None else '_forward'))(*values)


# softmax and attention probabilities

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
    for (long long j = threadIdx.x; j < width; j += blockDim.x) {
        long long k = kstart + j;
        float v = float(x[row * width + j]) * scale;
        if (causal && (k > q || (window > 0 && q - k >= window))) v += -1.e9f;
        mx = fmaxf(mx, v);
    }
    scratch[threadIdx.x] = mx;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (threadIdx.x < s)
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + s]);
        __syncthreads();
    }
    mx = scratch[0];
    __syncthreads();

    float sum = 0.f;
    for (long long j = threadIdx.x; j < width; j += blockDim.x) {
        long long k = kstart + j;
        float v = float(x[row * width + j]) * scale;
        if (causal && (k > q || (window > 0 && q - k >= window))) v += -1.e9f;
        sum += expf(v - mx);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
        __syncthreads();
    }
    sum = scratch[0];
    for (long long j = threadIdx.x; j < width; j += blockDim.x) {
        long long k = kstart + j, idx = row * width + j;
        float v = float(x[idx]) * scale;
        if (causal && (k > q || (window > 0 && q - k >= window))) v += -1.e9f;
        float p = expf(v - mx) / sum;
        if (save) saved[idx] = p;
        y[idx] = OUTPUT(keep ? (keep[idx] ? p * dropout_scale : 0.f) : p);
    }
}

extern "C" __global__ void softmax_backward(
    const INPUT* g, const float* p, const bool* keep, OUTPUT* dx,
    long long width, float dropout_scale, float scale) {
    __shared__ float scratch[256];
    long long row = blockIdx.x;
    float dot = 0.f;
    for (long long j = threadIdx.x; j < width; j += blockDim.x) {
        long long idx = row * width + j;
        float v = float(g[idx]);
        if (keep) v = keep[idx] ? v * dropout_scale : 0.f;
        dot += v * p[idx];
    }
    scratch[threadIdx.x] = dot;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
        __syncthreads();
    }
    dot = scratch[0];
    for (long long j = threadIdx.x; j < width; j += blockDim.x) {
        long long idx = row * width + j;
        float v = float(g[idx]);
        if (keep) v = keep[idx] ? v * dropout_scale : 0.f;
        dx[idx] = OUTPUT(p[idx] * (v - dot) * scale);
    }
}
'''


@lru_cache(None)
def _softmax_module(input_bf16, output_bf16):
    source = _SOFTMAX_SOURCE.replace('INPUT', '__nv_bfloat16' if input_bf16 else 'float')
    source = source.replace('OUTPUT', '__nv_bfloat16' if output_bf16 else 'float')
    return xp.RawModule(code = source, options = ('--std=c++11',))


def softmax(input, dtype, *, scale = _ONE, mask = None, dropout_scale = _ONE,
            save = False, causal = False, qstart = 0, kstart = 0, window = None):
    input = xp.ascontiguousarray(input)
    output = xp.empty(input.shape, dtype = dtype)
    saved_probabilities = xp.empty(input.shape if save else (0,), dtype = FLOAT_TYPE)
    kernel_module = _softmax_module(is_bf16(input), np.dtype(dtype) == AMP_TYPE)
    forward_kernel = kernel_module.get_function('softmax_forward')
    width = input.shape[-1]
    num_rows = input.size // width
    if num_rows:
        arguments = (
            input, output, saved_probabilities,
            mask if mask is not None else np.uint64(0),
            np.int64(width), np.int64(input.shape[-2] if input.ndim > 1 else 1),
            np.int64(qstart), np.int64(kstart), np.int64(window or 0),
            np.int32(causal), _float_scalar(scale), _float_scalar(dropout_scale),
            np.int32(save))
        forward_kernel((num_rows,), (256,), arguments)
    return output, saved_probabilities if save else None


def softmax_backward(gradient, probabilities, dtype, *, mask = None,
                     dropout_scale = _ONE, scale = _ONE):
    gradient = xp.ascontiguousarray(gradient)
    output = xp.empty(gradient.shape, dtype = dtype)
    kernel_module = _softmax_module(is_bf16(gradient), np.dtype(dtype) == AMP_TYPE)
    backward_kernel = kernel_module.get_function('softmax_backward')
    width = gradient.shape[-1]
    if gradient.size:
        arguments = (
            gradient, probabilities, mask if mask is not None else np.uint64(0),
            output, np.int64(width), _float_scalar(dropout_scale), _float_scalar(scale))
        backward_kernel((gradient.size // width,), (256,), arguments)
    return output


# rotary position embeddings

def rope(input, cos, sin, rotary_dim, backward = False, dtype = None):
    """Index paired dimensions directly, including strided head-split inputs."""
    dtype = input.dtype if dtype is None else dtype
    output = xp.empty(input.shape, dtype = dtype)
    width = input.shape[-1]
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
    ''', 'rope')(input, cos, sin, np.int64(width), np.int64(rotary_dim),
                np.int64(input.shape[-2]), _NEGATIVE_ONE if backward else _ONE, output)
    return output


# token and sinusoidal embeddings

def scatter_embedding(destination, indices, gradient, factor):
    """Scale FP32/BF16 gradients in registers, then FP32 atomicAdd."""
    if not use_kernels(gradient):
        xp.add.at(destination, indices, gradient * _array_scalar(factor, gradient.dtype))
        return
    _elementwise('T g, raw I ids, int64 width, int64 vocab, float32 factor',
                 'raw float32 dest',
                 'long long row = ((ids[i / width] % vocab) + vocab) % vocab; '
                 'atomicAdd(&dest[row*width + i % width], float(g)*factor);',
                 'embedding_scatter')(gradient, indices, np.int64(gradient.shape[-1]),
                                      np.int64(destination.shape[0]), _float_scalar(factor), destination)


def sinusoidal(length, width, dtype):
    output = xp.empty((length, width), dtype = dtype)
    _elementwise('int64 width', 'T y', r'''
        long long d = i % width;
        float angle = float(i / width) / powf(10000.f, float(2*(d/2))/float(width));
        y = T(d % 2 ? cosf(angle) : sinf(angle));
    ''', 'sinusoidal')(np.int64(width), output)
    return output
