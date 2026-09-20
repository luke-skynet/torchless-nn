"""Shared array backend: array module, float type, tensor initializers, and the
inference_mode / empty_weights construction contexts.

Select XP_RUNTIME=CPU before importing the library."""
import os
import numpy as np

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
    xp = np

FLOAT_TYPE = xp.float32 # (TF32 enabled for cupy)


def to_numpy(array):
    """Return a host array from either backend."""
    return np.asarray(array) if xp is np else xp.asnumpy(array)


# inference mode: build a model with no training state at all

INFERENCE_MODE = False


class inference_mode:

    """Build and run a model without any of the state that only backward needs.

    Layers constructed inside this context allocate no gradient, moment or variance
    buffers - three extra full size copies of every parameter - and Network._forward
    drops each layer's cached activations as soon as the next layer has consumed them.
    For a Gemma 4 31B configuration that is the difference between ~474 GiB and
    ~115 GiB.

    Construction has to happen inside the context, because the buffers are allocated
    in each layer's __init__:

        with inference_mode():
            model = gemma_gpt(**config)

        model.predict(tokens)          # fine, inside or outside the context

    Layers remember how they were built, so prediction behaves correctly either way.
    Calling backward() on such a model raises, which is the intent - there is nowhere
    for a gradient to accumulate.
    """

    def __init__(self, enabled = True):
        self.enabled = enabled

    def __enter__(self):
        global INFERENCE_MODE
        self.previous  = INFERENCE_MODE
        INFERENCE_MODE = self.enabled
        return self

    def __exit__(self, *exception):
        global INFERENCE_MODE
        INFERENCE_MODE = self.previous
        return False


# tensor initialization with float type

def init_random_tensor(size):
    rng = xp.random.default_rng()
    return rng.standard_normal(size, dtype = FLOAT_TYPE)

def init_zeros_tensor(size):
    return xp.zeros(size, dtype = FLOAT_TYPE)


# Checkpoint construction allocates weight storage without random initialization.

EMPTY_WEIGHTS = False

class empty_weights:
    def __enter__(self):
        global EMPTY_WEIGHTS
        if not INFERENCE_MODE:
            raise RuntimeError("empty_weights requires inference_mode")
        self.previous = EMPTY_WEIGHTS
        EMPTY_WEIGHTS = True
        return self

    def __exit__(self, *exception):
        global EMPTY_WEIGHTS
        EMPTY_WEIGHTS = self.previous


def init_weight_tensor(size, scale = 1.0):
    if EMPTY_WEIGHTS:
        return xp.empty(size, dtype = FLOAT_TYPE)
    return init_random_tensor(size) / scale
