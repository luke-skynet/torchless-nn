"""Shared array backend; select TORCHLESS_BACKEND=numpy before importing the library."""
import os
import numpy as np

os.environ['CUPY_TF32'] = "1"
os.environ['CUPY_ACCELERATORS'] = "cub,cutensor"

name = os.environ.get('TORCHLESS_BACKEND', 'cupy')
if name == 'numpy':
    xp = np
elif name == 'cupy':
    import cupy as xp
    from cupy.cuda import cublas
    
    cublas_handle = xp.cuda.Device().cublas_handle
    cublas.setMathMode(cublas_handle, cublas.CUBLAS_TENSOR_OP_MATH)
else:
    raise ValueError('TORCHLESS_BACKEND must be numpy or cupy')

FLOAT_TYPE = xp.float32 # (TF32 enabled)


def to_numpy(array):
    """Return a host array from either backend."""
    return np.asarray(array) if xp is np else xp.asnumpy(array)
