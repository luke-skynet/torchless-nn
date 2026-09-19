import os

runtime = os.environ.get('XP_RUNTIME', 'CUDA')

assert runtime == "CUDA" or runtime == "CPU"

if runtime == "CUDA":

    os.environ['CUPY_TF32'] = "1"
    os.environ['CUPY_ACCELERATORS'] = "cub,cutensor"
    
    import cupy
    from cupy.cuda import cublas
    
    cublas_handle = cupy.cuda.Device().cublas_handle
    cublas.setMathMode(cublas_handle, cublas.CUBLAS_TENSOR_OP_MATH)
    
    xp = cupy

else:
    import numpy as np
    xp = np

FLOAT_TYPE = xp.float32 # (TF32 enabled for cupy)


# tensor initialization with float type
rng = xp.random.default_rng()

def init_random_tensor(size):
    return rng.standard_normal(size, dtype = FLOAT_TYPE)

def init_zeros_tensor(size):
    return xp.zeros(size, dtype = FLOAT_TYPE)