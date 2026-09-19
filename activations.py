from backend import xp, FLOAT_TYPE
from utils import Layer


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

    def __init__(self):
        super(GeLU, self).__init__()

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

    def __init__(self, temperature = 1.0):
        super(SoftMax, self).__init__()
        self.temperature = temperature

    def forward(self, input):
        self.input = input

        normalization = xp.max(self.input, axis = -1, keepdims = True) 
        exponent = xp.exp((self.input - normalization) / self.temperature)

        self.output = exponent / xp.sum(exponent, axis = -1, keepdims=True)
        return self.output

    def backward(self, gradient):
        return gradient # handled by Cross Entropy Loss