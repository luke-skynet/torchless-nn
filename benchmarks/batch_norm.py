"""Compare split BatchNorm reductions with the previous shared kernels.

XP_RUNTIME=CUDA python benchmarks/batch_norm.py --dtype both
Use --all-shapes for all 20 BatchNorm inputs in the DenseNet demo.
Times include allocation/dispatch but exclude running-stat/gradient accumulation,
which are unchanged. GPU timings use CUDA events after warm-up.
"""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('XP_RUNTIME', 'CUDA')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16', 'both'], default='float32')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--repeat', type=int, default=30)
    parser.add_argument('--all-shapes', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.repeat < 1:
        parser.error('batch-size and repeat must be positive')

    import numpy as np
    from backend import xp, AMP_TYPE
    import kernel_ops as ops
    if xp is np:
        parser.error('This benchmark requires XP_RUNTIME=CUDA')
    from cupyx.profiler import benchmark

    # Historical generic BatchNorm kernels, retained only for this comparison.
    def generic_forward(x, gamma, beta, eps, stats=None):
        axes = (0, 2, 3)
        count = np.float32(x.shape[0] * x.shape[2] * x.shape[3])
        if stats is None:
            mean = ops._reduce('T x', 'float(x)', 'norm_sum')(
                x, axis=axes, keepdims=True) / count
            variance = ops._reduce('T x, float32 mean',
                '(float(x)-mean)*(float(x)-mean)', 'norm_variance')(
                    x, mean, axis=axes, keepdims=True) / count
        else:
            mean, variance = stats
        inv = np.float32(1) / xp.sqrt(variance + eps)
        return ops._centered_affine(x, gamma, beta, mean, inv), mean, variance, inv

    def generic_backward(g, x, gamma, mean, inv):
        axes = (0, 2, 3)
        count = np.float32(x.shape[0] * x.shape[2] * x.shape[3])
        params = 'G g, T x, float32 mean, float32 inv'
        dg = ops._reduce(params, 'float(g)*(float(x)-mean)*inv', 'norm_dgamma')(
            g, x, mean, inv, axis=axes, keepdims=True)
        db = ops._reduce('G g', 'float(g)', 'norm_dbeta')(g, axis=axes, keepdims=True)
        params += ', float32 gamma'
        dot = ops._reduce(params, 'float(g)*gamma*(float(x)-mean)*inv', 'norm_dot')(
            g, x, mean, inv, gamma, axis=axes, keepdims=True) / count
        avg = ops._reduce('G g, float32 gamma', 'float(g)*gamma', 'norm_avg')(
            g, gamma, axis=axes, keepdims=True) / count
        dx = xp.empty_like(x)
        ops._elementwise(params + ', float32 dot, float32 avg', 'T dx',
            'dx = T((float(g)*gamma - avg - (float(x)-mean)*inv*dot)*inv);',
            'norm_backward')(g, x, mean, inv, gamma, dot, avg, dx)
        return dx, dg, db

    shapes = [(32, 32), (128, 16), (160, 8)]
    if args.all_shapes:
        shapes = ([(c, 32) for c in range(32, 129, 16)]
                  + [(c, 16) for c in range(64, 161, 16)]
                  + [(c, 8) for c in range(80, 161, 16)])
    dtypes = ['float32', 'bfloat16'] if args.dtype == 'both' else [args.dtype]
    device = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
    print(f"GPU: {device['name']}; CuPy: {xp.__version__}")
    print('GPU milliseconds; old=generic shared reductions, new=split NCHW reductions')
    print(f"{'dtype':<9} {'N,C,H,W':<22} {'stage':<10} {'old ms':>10} {'new ms':>10} {'speedup':>9}")
    for dtype_name in dtypes:
        dtype = np.dtype('float32') if dtype_name == 'float32' else AMP_TYPE
        # Fail clearly if the requested BF16 build/device is unavailable.
        if dtype_name == 'bfloat16' and int(xp.cuda.Device().compute_capability) < 80:
            parser.error('BF16 requires SM80+')
        xp.empty(0, dtype=dtype)
        for channels, size in shapes:
            shape = (args.batch_size, channels, size, size)
            rng = xp.random.default_rng(42)
            x = rng.standard_normal(shape, dtype=xp.float32).astype(dtype, copy=False)
            g = rng.standard_normal(shape, dtype=xp.float32).astype(dtype, copy=False)
            gamma = xp.ones((1, channels, 1, 1), dtype=xp.float32)
            beta = xp.zeros_like(gamma)
            eps = np.float32(1e-5)

            def old_forward():
                return generic_forward(x, gamma, beta, eps)

            def new_forward():
                return ops.batch_norm_forward(x, gamma, beta, eps)

            def old_backward(saved):
                return generic_backward(g, x, gamma, saved[1], saved[3])

            def new_backward(saved):
                return ops.batch_norm_backward(g, x, gamma, saved[1], saved[3])

            old_saved, new_saved = old_forward(), new_forward()
            stages = [
                ('forward', old_forward, new_forward),
                ('backward', lambda: old_backward(old_saved), lambda: new_backward(new_saved)),
                ('train', lambda: old_backward(old_forward()), lambda: new_backward(new_forward())),
                ('eval', lambda: generic_forward(x, gamma, beta, eps, stats=old_saved[1:3]),
                 lambda: ops.batch_norm_forward(x, gamma, beta, eps, statistics=old_saved[1:3])),
            ]
            for stage, old, new in stages:
                times = [float(benchmark(fn, n_repeat=args.repeat, n_warmup=5,
                                         max_duration=3).gpu_times.mean()) * 1000
                         for fn in (old, new)]
                print(f'{dtype_name:<9} {str(shape):<22} {stage:<10} '
                      f'{times[0]:10.3f} {times[1]:10.3f} {times[0]/times[1]:8.2f}x', flush=True)


if __name__ == '__main__':
    main()
