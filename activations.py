from backend import xp
import math
import numpy as np
from utils import Layer

if xp is not np:
    from cupyx.scipy.special import erf


def _erf(input):
    """Elementwise erf, evaluated on the input's device."""
    if isinstance(input, np.ndarray):
        return np.vectorize(math.erf, otypes=[input.dtype])(input)
    with input.device:
        return erf(input)


class ReLU(Layer):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        self.input = input
        self.output = xp.maximum(input, 0)
        return self.output

    def backward(self, gradient):
        return xp.heaviside(self.input, 0) * gradient


class GeLU(Layer):

    """Exact, erf-based GELU."""

    CACHED = ("input", "output", "cdf")

    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, input):
        self.input = input
        self.cdf = 0.5 * (1 + _erf(input / 2**0.5))
        self.output = input * self.cdf
        return self.output

    def backward(self, gradient):
        return (self.cdf + self.input * xp.exp(-0.5 * self.input**2) / (2 * xp.pi)**0.5) * gradient


class GeLUTanh(Layer):

    """Tanh approximation of GELU, used by Gemma."""

    CACHED = ("input", "output", "tanh")

    def __init__(self):
        super(GeLUTanh, self).__init__()

    def forward(self, input):
        self.input = input
        self.tanh = xp.tanh((2/xp.pi)**0.5 * (self.input + 0.044715*self.input**3))
        self.output = 0.5 * self.input * (1 + self.tanh)
        return self.output

    def backward(self, gradient):
        return (0.5 * (1 + self.tanh) + \
                0.5 * self.input * (1 - self.tanh**2) * \
               (2/xp.pi)**0.5 * (1 + 0.134145 * self.input**2)) * gradient


class SiLU(Layer):

    CACHED = ("input", "output", "sigmoid")

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, input):
        self.input = input
        self.sigmoid = (1 + xp.tanh(self.input / 2)) / 2
        self.output = self.input * self.sigmoid
        return self.output

    def backward(self, gradient):
        return (self.sigmoid + self.output - self.sigmoid * self.output) * gradient


class SoftMax(Layer):

    """Softmax over the last axis.

    fused_loss=True (the default) is the arrangement the rest of the library assumes:
    this layer sits last, CrossEntropy.gradients already returns the gradient with
    respect to this layer's *input* for the softmax + cross entropy pair, and backward
    passes it straight through. That is only correct in that position - used anywhere
    else it silently drops the softmax Jacobian, so Network rejects a fused SoftMax
    that is not the final layer.

    fused_loss=False computes the real Jacobian, y * (g - sum(g * y)), and can be used
    anywhere in a network.
    """

    def __init__(self, temperature = 1.0, fused_loss = True):
        super(SoftMax, self).__init__()
        self.temperature = temperature
        self.fused_loss  = fused_loss

    def forward(self, input):
        self.input = input

        normalization = xp.max(self.input, axis = -1, keepdims = True)
        exponent = xp.exp((self.input - normalization) / self.temperature)

        self.output = exponent / xp.sum(exponent, axis = -1, keepdims=True)
        return self.output

    def backward(self, gradient):

        if not self.fused_loss:
            gradient = self.output * (gradient - (gradient * self.output).sum(axis = -1, keepdims = True))

        # temperature divides the logits on the way in, so it divides the gradient on
        # the way out. At the default of 1.0 this is the untouched gradient.
        return gradient if self.temperature == 1.0 else gradient / self.temperature
