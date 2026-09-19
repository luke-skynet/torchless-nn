from backend import xp, FLOAT_TYPE, init_random_tensor, init_zeros_tensor

import numpy as np
import cv2

# layer interface and residual layer wrapper

class Layer:

    def __init__(self):

        self.input = None
        self.output = None

        self.parameters = []
        self.gradients = []
        self.moments = []
        self.variances = []

        self.eval_mode = False
        self.is_embed_layer_back = False

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError

    def set_eval(self, eval_mode):
        self.eval_mode = eval_mode
        
    def register_param(self, parameter):
        
        gradient = init_zeros_tensor(parameter.shape)
        moment = init_zeros_tensor(parameter.shape)
        variance = init_zeros_tensor(parameter.shape)
        
        self.parameters.append(parameter)
        self.gradients.append(gradient)
        self.moments.append(moment)
        self.variances.append(variance)
        
        return gradient

    def zero_grad(self):
        for grad in self.gradients:
            grad.fill(0)
    
    def zero_adam(self):
        for moment, variance in zip(self.moments, self.variances):
            moment.fill(0)
            variance.fill(0)
            

class Residual(Layer):
    
    def __init__(self, layers:list[Layer], mode = "add", concat_axis = 1):
        super(Residual, self).__init__()
        
        self.layers = layers
        self.mode = mode
        assert self.mode in {"add", "concat"}
        self.concat_axis = concat_axis
        
        self.parameters = []
        self.gradients  = []
        self.moments    = []
        self.variances  = []
        
        for layer in layers:
            self.parameters = self.parameters + layer.parameters
            self.gradients  = self.gradients  + layer.gradients
            self.moments    = self.moments    + layer.moments
            self.variances  = self.variances  + layer.variances
        
    def forward(self, input):
        
        self.input = input
        x = input
        
        for layer in self.layers:
            x = layer.forward(x)
        
        if self.mode == "add":
            self.output = self.input + x
        elif self.mode == "concat":
            self.output = xp.concatenate((self.input, x), axis = self.concat_axis)
        return self.output
    
    def backward(self, gradient):
        
        gradient, nabla = (gradient, gradient) if self.mode == "add" else xp.array_split(gradient, 
                                                                                           (self.input.shape[self.concat_axis], ),
                                                                                           axis = self.concat_axis)
        
        for layer in reversed(self.layers):
            nabla = layer.backward(nabla)
            
        gradient += nabla
        return gradient

    def set_eval(self, eval_mode):
        for layer in self.layers:
            layer.set_eval(eval_mode)
            

# crude augments from scratch

def random_flip(data):
    indices = np.random.choice(np.arange(0, len(data)), size = len(data) // 2)
    data[indices] = np.flip(data[indices], axis = 3)
    return data
    
def random_shift(data):
    
    x_shifts = np.random.choice(np.arange(0,9),  size = len(data))
    y_shifts = np.random.choice(np.arange(0,9),  size = len(data))
    
    x_res = data.shape[2]
    y_res = data.shape[3]

    for i, (img, dx, dy) in enumerate(zip(data, x_shifts, y_shifts)):
        cropped = np.pad(img, ((0,0), (4,4), (4,4)))
        cropped = cropped[:, dx:x_res+dx, dy:y_res+dy]
        data[i] = cropped
    
    return data


def random_rotate(data):
    
    x_res = data.shape[2]
    y_res = data.shape[3]
    
    mid = (x_res // 2, y_res // 2)

    angles = np.random.randint(-15, 16, len(data))
    
    for i, (img, angle) in enumerate(zip(data, angles)):
        img = img.transpose(1, 2, 0)
        matrix  = cv2.getRotationMatrix2D(mid, angle, 1.0)
        data[i] = cv2.warpAffine(img, matrix, (x_res, y_res)).transpose(2, 0, 1)
    
    return data


def augment_images(data):
    x = data.copy()
    return random_shift(random_rotate(random_flip(x)))