"""Compare fused row normalization against the previous generic CUDA kernels.

XP_RUNTIME=CUDA python benchmarks/row_norm.py --dtype both
Includes allocation/dispatch, excludes layer parameter-gradient accumulation.
"""
import argparse
import os
from pathlib import Path
import sys
import numpy as np

os.environ.setdefault('XP_RUNTIME', 'CUDA')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import xp, FLOAT_TYPE, AMP_TYPE
import kernel_ops as ops
from kernel_ops import _reduce, _elementwise, _centered_affine, _float_scalar, _ONE


# Previous implementations retained as the performance baseline.
def old_layer_norm_forward(input, gamma, beta, eps):
    """Normalize the last dimension; cache its mean and inverse deviation."""
    count = FLOAT_TYPE(input.shape[-1])
    mean = _reduce('T x', 'float(x)', 'layer_norm_sum')(
        input, axis = -1, keepdims = True) / count
    variance = _reduce('T x, float32 mean', '(float(x)-mean)*(float(x)-mean)',
                      'layer_norm_variance')(input, mean, axis = -1, keepdims = True) / count
    inv_std = _ONE / xp.sqrt(variance + _float_scalar(eps))
    return _centered_affine(input, gamma, beta, mean, inv_std), mean, inv_std


def old_layer_norm_backward(gradient, input, gamma, mean, inv_std):
    """LayerNorm has both centered-input and mean-gradient corrections."""
    count = FLOAT_TYPE(input.shape[-1])
    param_axes = tuple(range(input.ndim - 1))
    args = 'G g, T x, float32 mean, float32 inv'
    gamma_grads = _reduce(args, 'float(g)*(float(x)-mean)*inv', 'layer_norm_dgamma')(
        gradient, input, mean, inv_std, axis = param_axes)
    beta_grads = _reduce('G g', 'float(g)', 'layer_norm_dbeta')(gradient, axis = param_axes)
    args += ', float32 gamma'
    dot = _reduce(args, 'float(g)*gamma*(float(x)-mean)*inv', 'layer_norm_dot')(
        gradient, input, mean, inv_std, gamma, axis = -1, keepdims = True) / count
    avg = _reduce('G g, float32 gamma', 'float(g)*gamma', 'layer_norm_avg')(
        gradient, gamma, axis = -1, keepdims = True) / count
    input_gradient = xp.empty_like(input)
    _elementwise(args + ', float32 dot, float32 avg', 'T dx',
                 'dx = T((float(g)*gamma - avg - (float(x)-mean)*inv*dot)*inv);',
                 'layer_norm_backward')(gradient, input, mean, inv_std, gamma, dot, avg, input_gradient)
    return input_gradient, gamma_grads, beta_grads


# RMS normalization

def old_rms_norm_forward(input, gamma, eps):
    """RMSNorm needs a second moment only, with no mean subtraction or bias."""
    count = FLOAT_TYPE(input.shape[-1])
    second_moment = _reduce('T x', 'float(x)*float(x)', 'rms_norm_square_sum')(
        input, axis = -1, keepdims = True) / count
    inv_rms = _ONE / xp.sqrt(second_moment + _float_scalar(eps))
    output = xp.empty_like(input)
    _elementwise('T x, float32 inv, float32 gamma', 'T y',
                 'y = T(float(x)*inv*gamma);', 'rms_norm_forward')(input, inv_rms, gamma, output)
    return output, inv_rms


def old_rms_norm_backward(gradient, input, gamma, inv_rms, with_scale = True):
    """Only the projection onto normalized input is subtracted in RMSNorm."""
    count = FLOAT_TYPE(input.shape[-1])
    args = 'G g, T x, float32 inv'
    gamma_grads = None
    if with_scale:
        gamma_grads = _reduce(args, 'float(g)*float(x)*inv', 'rms_norm_dgamma')(
            gradient, input, inv_rms, axis = tuple(range(input.ndim - 1)))
    args += ', float32 gamma'
    dot = _reduce(args, 'float(g)*gamma*float(x)*inv', 'rms_norm_dot')(
        gradient, input, inv_rms, gamma, axis = -1, keepdims = True) / count
    input_gradient = xp.empty_like(input)
    _elementwise(args + ', float32 dot', 'T dx',
                 'dx = T((float(g)*gamma - float(x)*inv*dot)*inv);',
                 'rms_norm_backward')(gradient, input, inv_rms, gamma, dot, input_gradient)
    return input_gradient, gamma_grads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16', 'both'], default='float32')
    parser.add_argument('--repeat', type=int, default=30)
    args = parser.parse_args()
    if xp is np:
        parser.error('This benchmark requires XP_RUNTIME=CUDA')
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    from cupyx.profiler import benchmark

    # Decode, small/large token batches, odd widths, and strided Q/K heads.
    shapes = [(1, 4096), (16, 4096), (1024, 768), (2048, 4096),
              (259, 1025), (2, 8, 1, 128, 128)]
    dtypes = ['float32', 'bfloat16'] if args.dtype == 'both' else [args.dtype]
    device = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
    print(f"GPU: {device['name']}; CuPy: {xp.__version__}")
    print('GPU milliseconds after warm-up; old=generic kernels, new=fused row kernels')
    print(f"{'dtype':<9} {'norm':<12} {'shape':<25} {'stage':<9} {'old ms':>10} {'new ms':>10} {'speedup':>9}")
    for dtype_name in dtypes:
        dtype = AMP_TYPE if dtype_name == 'bfloat16' else np.dtype('float32')
        if dtype_name == 'bfloat16' and int(xp.cuda.Device().compute_capability) < 80:
            parser.error('BF16 requires SM80+ and the patched CuPy build')
        xp.empty(0, dtype=dtype)
        for shape in shapes:
            rng = xp.random.default_rng(42)
            x = rng.standard_normal(shape, dtype=xp.float32).astype(dtype)
            g = rng.standard_normal(shape, dtype=xp.float32).astype(dtype)
            if len(shape) == 5:
                x, g = x.swapaxes(1, 3), g.swapaxes(1, 3)
            gamma = xp.ones(x.shape[-1], dtype=xp.float32)
            beta = xp.zeros_like(gamma)
            eps = np.float32(1e-5)
            for kind in ('layer', 'rms', 'rms_unscaled'):
                if kind == 'layer':
                    def old_forward():
                        return old_layer_norm_forward(x, gamma, beta, eps)
                    def new_forward():
                        return ops.layer_norm_forward(x, gamma, beta, eps)
                    def old_backward(saved):
                        return old_layer_norm_backward(g, x, gamma, saved[1], saved[2])
                    def new_backward(saved):
                        return ops.layer_norm_backward(g, x, gamma, saved[1], saved[2])
                else:
                    scaled = kind == 'rms'
                    scale = gamma if scaled else np.float32(1)
                    def old_forward():
                        return old_rms_norm_forward(x, scale, eps)
                    def new_forward():
                        return ops.rms_norm_forward(x, scale, eps)
                    def old_backward(saved):
                        return old_rms_norm_backward(g, x, scale, saved[1], scaled)
                    def new_backward(saved):
                        return ops.rms_norm_backward(g, x, scale, saved[1], scaled)
                old_saved, new_saved = old_forward(), new_forward()
                stages = [('forward', old_forward, new_forward),
                          ('backward', lambda: old_backward(old_saved), lambda: new_backward(new_saved)),
                          ('train', lambda: old_backward(old_forward()), lambda: new_backward(new_forward()))]
                for stage, old, new in stages:
                    times = [float(benchmark(fn, n_repeat=args.repeat, n_warmup=5,
                                             max_duration=3).gpu_times.mean()) * 1000
                             for fn in (old, new)]
                    print(f'{dtype_name:<9} {kind:<12} {str(x.shape):<25} {stage:<9} '
                          f'{times[0]:10.3f} {times[1]:10.3f} {times[0]/times[1]:8.2f}x', flush=True)


if __name__ == '__main__':
    main()
