import kernel_ops as ops
from backend import xp, FLOAT_TYPE
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
        self._zero = FLOAT_TYPE(0)

    def forward(self, input):
        self.input = input
        if ops.use_kernels(input):
            self.output = ops.activation(input, 'relu')
            return self.output
        self.output = xp.maximum(input, self._zero)
        return self.output

    def backward(self, gradient):
        if ops.use_kernels(self.input):
            return ops.activation(self.input, 'relu', gradient)
        return xp.heaviside(self.input, self._zero) * gradient


class GeLU(Layer):

    """Exact, erf-based GELU."""

    CACHED = ("input", "output", "cdf")

    def __init__(self):
        super(GeLU, self).__init__()
        self._half = FLOAT_TYPE(0.5)
        self._one = FLOAT_TYPE(1)
        self._sqrt_two = FLOAT_TYPE(2**0.5)
        self._negative_half = FLOAT_TYPE(-0.5)
        self._two = FLOAT_TYPE(2)
        self._sqrt_two_pi = FLOAT_TYPE((2 * xp.pi)**0.5)

    def forward(self, input):
        self.input = input
        if ops.use_kernels(input):
            self.output = ops.activation(input, 'gelu')
            return self.output
        self.cdf = self._half * (self._one + _erf(input / self._sqrt_two))
        self.output = input * self.cdf
        return self.output

    def backward(self, gradient):
        if ops.use_kernels(self.input):
            return ops.activation(self.input, 'gelu', gradient)
        return (self.cdf + self.input * xp.exp(self._negative_half * self.input**self._two) / self._sqrt_two_pi) * gradient


class GeLUTanh(Layer):

    """Tanh approximation of GELU, used by Gemma."""

    CACHED = ("input", "output", "tanh")

    def __init__(self):
        super(GeLUTanh, self).__init__()
        self._half = FLOAT_TYPE(0.5)
        self._one = FLOAT_TYPE(1)
        self._two = FLOAT_TYPE(2)
        self._three = FLOAT_TYPE(3)
        self._sqrt_two_over_pi = FLOAT_TYPE((2/xp.pi)**0.5)
        self._cubic_coefficient = FLOAT_TYPE(0.044715)
        self._cubic_derivative = FLOAT_TYPE(0.134145)

    def forward(self, input):
        self.input = input
        if ops.use_kernels(input):
            self.output = ops.activation(input, 'gelutanh')
            return self.output
        self.tanh = xp.tanh(self._sqrt_two_over_pi * (self.input + self._cubic_coefficient*self.input**self._three))
        self.output = self._half * self.input * (self._one + self.tanh)
        return self.output

    def backward(self, gradient):
        if ops.use_kernels(self.input):
            return ops.activation(self.input, 'gelutanh', gradient)
        return (self._half * (self._one + self.tanh) + \
                self._half * self.input * (self._one - self.tanh**self._two) * \
               self._sqrt_two_over_pi * (self._one + self._cubic_derivative * self.input**self._two)) * gradient


class SiLU(Layer):

    CACHED = ("input", "output", "sigmoid")

    def __init__(self):
        super(SiLU, self).__init__()
        self._one = FLOAT_TYPE(1)
        self._two = FLOAT_TYPE(2)

    def forward(self, input):
        self.input = input
        if ops.use_kernels(input):
            self.output = ops.activation(input, 'silu')
            return self.output
        self.sigmoid = (self._one + xp.tanh(self.input / self._two)) / self._two
        self.output = self.input * self.sigmoid
        return self.output

    def backward(self, gradient):
        if ops.use_kernels(self.input):
            return ops.activation(self.input, 'silu', gradient)
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

    CACHED = ("input", "output", "probabilities")

    def __init__(self, temperature = 1.0, fused_loss = True):
        super(SoftMax, self).__init__()
        self.temperature = temperature
        self.fused_loss  = fused_loss

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = FLOAT_TYPE(value)
        self._inverse_temperature = FLOAT_TYPE(1) / self._temperature

    def forward(self, input):
        self.input = input
        if ops.use_kernels(input):
            self.output, self.probabilities = ops.softmax(
                input, FLOAT_TYPE if self.fused_loss else input.dtype,
                scale=self._inverse_temperature,
                save=not self.inference_only and not self.fused_loss)
            return self.output

        normalization = xp.max(self.input, axis = -1, keepdims = True)
        exponent = xp.exp((self.input - normalization) / self.temperature)

        self.output = exponent / xp.sum(exponent, axis = -1, keepdims=True)
        return self.output

    def backward(self, gradient):
        if ops.use_kernels(self.input):
            if self.fused_loss:
                return ops.scale(gradient, self._inverse_temperature, self.input.dtype)
            return ops.softmax_backward(gradient, self.probabilities,
                                        self.input.dtype, scale=self._inverse_temperature)

        if not self.fused_loss:
            gradient = self.output * (gradient - (gradient * self.output).sum(axis = -1, keepdims = True))

        # temperature divides the logits on the way in, so it divides the gradient on
        # the way out. At the default of 1.0 this is the untouched gradient.
        return gradient if self.temperature == 1.0 else gradient / self.temperature
