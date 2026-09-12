import cupy

from utils import Layer, FLOAT_TYPE, init_random_tensor, init_zeros_tensor

class Convolution(Layer):

    def __init__(self, input_dim: tuple, num_kernels:int, kernel_size: tuple, padding = (0, 0)):
        super(Convolution, self).__init__()

        self.kernel_size =  kernel_size
        self.output_dims = (input_dim[1] + 2*padding[0] + 1 - self.kernel_size[0],
                            input_dim[2] + 2*padding[1] + 1 - self.kernel_size[1])
        
        self.pad_h, self.pad_w = padding
        self.padded_input = None

        self.weights = init_random_tensor((num_kernels, input_dim[0], kernel_size[0], kernel_size[1]))
        self.weights = self.weights / (input_dim[0] * kernel_size[0] * kernel_size[1])**0.5
        self.bias    = init_zeros_tensor(num_kernels).reshape(1, num_kernels, 1, 1)
        
        self.weight_grads   = self.register_param(self.weights)
        self.bias_grads     = self.register_param(self.bias)


    def forward(self, input):

        self.input = input
        
        self.padded_input = cupy.pad(self.input, ((0, 0), (0, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w)))

        self.output = cupy.lib.stride_tricks.sliding_window_view(self.padded_input, self.kernel_size, (2, 3))
        self.output = cupy.einsum("nchwkl,ockl->nohw", self.output, self.weights)
        self.output = self.output + self.bias

        return self.output

    def backward(self, gradient):

        conv_in = cupy.lib.stride_tricks.sliding_window_view(self.padded_input, self.output_dims, (2, 3))
        self.weight_grads += cupy.einsum("ncklhw,nohw->ockl", conv_in, gradient)

        self.bias_grads += cupy.sum(gradient, axis = (0, 2, 3), keepdims = True)

        grad_pad_h = self.kernel_size[0] - 1
        grad_pad_w = self.kernel_size[1] - 1

        flipped_weights = cupy.flip(self.weights, axis=(2, 3))

        gradient = cupy.pad(gradient, ((0, 0), (0, 0), (grad_pad_h, grad_pad_h), (grad_pad_w, grad_pad_w)))
        gradient = cupy.lib.stride_tricks.sliding_window_view(gradient, self.kernel_size, (2, 3))
        gradient = cupy.einsum("nohwkl,ockl->nchw", gradient, flipped_weights)
        gradient = gradient[:, :, self.pad_h:gradient.shape[2]-self.pad_h, self.pad_w:gradient.shape[3]-self.pad_w]
        
        return gradient


class BatchNorm(Layer):
    def __init__(self, num_channels):
        super(BatchNorm, self).__init__()
        
        self.channels = num_channels
        self.eps = 1e-5
        
        self.momentum = 0.1
        
        self.running_mean = init_zeros_tensor((1, num_channels, 1, 1))
        self.running_var  = init_zeros_tensor((1, num_channels, 1, 1))

        self.gamma = init_zeros_tensor((1, num_channels, 1, 1)) + 1
        self.beta  = init_zeros_tensor((1, num_channels, 1, 1))

        self.gamma_grads = self.register_param(self.gamma)
        self.beta_grads  = self.register_param(self.beta)
        
        self.mean = None
        self.var = None
        self.std = None
        self.centered = None
        self.normed = None
    
    def forward(self, input):
        
        self.input = input
        
        if self.eval_mode is False:
            self.mean = cupy.mean(input, axis = (0, 2, 3), keepdims = True)
            self.var  = cupy.var(input, axis = (0, 2, 3), keepdims = True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * self.var
        else:
            self.mean = self.running_mean
            self.var = self.running_var
            
        self.centered = self.input - self.mean
        
        self.std = (self.var + self.eps)**0.5
        self.normed = self.centered / self.std
        
        self.output = self.gamma * self.normed + self.beta
        return self.output
            
    def backward(self, gradient):
        
        self.gamma_grads += cupy.sum(gradient * self.normed, axis = (0, 2, 3), keepdims = True)
        self.beta_grads  += cupy.sum(gradient,               axis = (0, 2, 3), keepdims = True)
        
        gradient = gradient * self.gamma
        
        gradient_normed = (gradient - cupy.mean(gradient, axis = (0, 2, 3), keepdims = True)) / self.std
        
        return gradient_normed - self.centered * (cupy.mean(gradient * self.centered, axis = (0, 2, 3), keepdims = True) / self.std**3)
    

class MaxPool(Layer):

    def __init__(self, pool_height, pool_width):
        super(MaxPool, self).__init__()

        self.pool_h, self.pool_w = (pool_height, pool_width)
        self.batch_size, self.channels, self.in_h, self.in_w = 0,0,0,0
        self.mask = None

    def forward(self, input):

        self.input = input
        self.batch_size, self.channels, self.in_h, self.in_w = self.input.shape

        out_h = self.in_h // self.pool_h
        out_w = self.in_w // self.pool_w

        view = self.input[:, :, :out_h*self.pool_h, :out_w*self.pool_w]
        view = view.reshape(self.batch_size, self.channels, out_h, self.pool_h, out_w, self.pool_w)
        
        self.output = view.max(axis=(3, 5), keepdims = True)

        self.mask   = self.output == view
        self.output = self.output.reshape(self.batch_size, self.channels, out_h, out_w)

        return self.output

    def backward(self, gradient):
        gradient = gradient[:, :, :, cupy.newaxis, :, cupy.newaxis]
        gradient = gradient * self.mask
        gradient = gradient.reshape(self.batch_size, self.channels, self.in_h, self.in_w)
        return gradient
    

class AveragePool(Layer):

    def __init__(self, pool_height, pool_width):
        super(AveragePool, self).__init__()

        self.pool_h, self.pool_w = (pool_height, pool_width)
        self.batch_size, self.channels, self.in_h, self.in_w = 0,0,0,0
        self.mask = None

    def forward(self, input):

        self.input = input
        self.batch_size, self.channels, self.in_h, self.in_w = self.input.shape

        out_h = self.in_h // self.pool_h
        out_w = self.in_w // self.pool_w

        view = self.input[:, :, :out_h*self.pool_h, :out_w*self.pool_w]
        view = view.reshape(self.batch_size, self.channels, out_h, self.pool_h, out_w, self.pool_w)
        
        self.output = view.mean(axis=(3, 5), keepdims = True)

        self.mask   = cupy.ones(view.shape, dtype = FLOAT_TYPE) / (self.pool_h * self.pool_w)
        self.output = self.output.reshape(self.batch_size, self.channels, out_h, out_w)
        return self.output

    def backward(self, gradient):
        gradient = gradient[:, :, :, cupy.newaxis, :, cupy.newaxis]
        gradient = gradient * self.mask
        gradient = gradient.reshape(self.batch_size, self.channels, self.in_h, self.in_w)
        return gradient


class Flatten(Layer):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        self.input  = input
        self.output = self.input.reshape((self.input.shape[0], -1))
        return self.output

    def backward(self, gradient):
        return gradient.reshape(self.input.shape)
    
    
class Dense(Layer):

    def __init__(self, input_size, output_size):
        super(Dense, self).__init__()

        self.weights = init_random_tensor((input_size, output_size)) / input_size**0.5
        self.bias    = init_zeros_tensor(output_size)

        self.weight_grads = self.register_param(self.weights)
        self.bias_grads   = self.register_param(self.bias)

    def forward(self, input):
        self.input  = input
        self.output = input @ self.weights + self.bias
        return self.output

    def backward(self, gradient):
        self.weight_grads += self.input.transpose() @ gradient
        self.bias_grads   += cupy.sum(gradient, axis = 0)
        return gradient @ self.weights.transpose()


class Dropout(Layer):
    def __init__(self, dropout_rate):
        super(Dropout, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.dropout_rng  = cupy.random.default_rng()
        self.dropout_neurons = None
    
    def forward(self, input):
        self.input = input
        if self.dropout_rate > 0.0 and self.eval_mode is False:
            self.dropout_neurons = self.dropout_rng.random(self.input.shape, dtype = FLOAT_TYPE) >= self.dropout_rate
            self.dropout_neurons = self.dropout_neurons / FLOAT_TYPE(1 - self.dropout_rate)
            self.output = self.input * self.dropout_neurons
        else:
            self.output = self.input
        return self.output
        
    def backward(self, gradient):
        if self.dropout_rate > 0.0 and self.eval_mode is False:
            return gradient * self.dropout_neurons
        return gradient


class MultiHeadAttention(Layer):

    def __init__(self, embedding_dim, context_length, num_heads, decoder = False,
                 attn_dropout_rate = 0.0, res_dropout_rate = 0.0):
        super(MultiHeadAttention, self).__init__()

        self.embedding_dim  = embedding_dim
        self.context_length = context_length

        self.num_heads = num_heads
        self.heads_dim = embedding_dim // num_heads

        self.is_decoder = decoder
        if self.is_decoder:
            self.mask = cupy.tril(cupy.ones((context_length, context_length)))
            self.mask = cupy.where(self.mask == 0, -1e9, 0.0).astype(FLOAT_TYPE, copy = False)

        self.query = None
        self.key_t = None
        self.value = None
        
        self.softmax = None
        self.dropped_softmax = None
        self.heads_out = None
        
        self.attn_dropout = Dropout(attn_dropout_rate)
        self.res_dropout  = Dropout(res_dropout_rate)  
        
        self.qkv_weights = init_random_tensor((embedding_dim, 3*embedding_dim)) / embedding_dim**0.5
        self.qkv_bias    = init_zeros_tensor(3*embedding_dim)
        
        self.out_weights = init_random_tensor((embedding_dim,   embedding_dim)) / embedding_dim**0.5
        self.out_bias    = init_zeros_tensor(embedding_dim)

        self.qkv_weight_grads = self.register_param(self.qkv_weights)
        self.qkv_bias_grads   = self.register_param(self.qkv_bias)
        
        self.out_weight_grads = self.register_param(self.out_weights)
        self.out_bias_grads   = self.register_param(self.out_bias)
 
    def forward(self, input):

        self.input = input
        B, T, C    = self.input.shape

        qkv = self.input @ self.qkv_weights + self.qkv_bias
        self.query, self.key_t, self.value = cupy.split(qkv, 3, axis = 2)

        self.query = self.query.reshape((B, T, self.num_heads, self.heads_dim)).transpose(0, 2, 1, 3)
        self.key_t = self.key_t.reshape((B, T, self.num_heads, self.heads_dim)).transpose(0, 2, 3, 1)
        self.value = self.value.reshape((B, T, self.num_heads, self.heads_dim)).transpose(0, 2, 1, 3)

        attends = (self.query @ self.key_t) / self.heads_dim**.5

        if self.is_decoder:
            attends += self.mask[:T,:T]

        normalization = cupy.max(attends, axis = -1, keepdims = True)
        exponent      = cupy.exp(attends - normalization)
        self.softmax  = exponent / cupy.sum(exponent, axis = -1, keepdims=True)
        self.dropped_softmax = self.attn_dropout.forward(self.softmax)

        self.heads_out = self.dropped_softmax @ self.value
        self.heads_out = self.heads_out.transpose(0, 2, 1, 3).reshape(B, T, C)

        self.output = self.heads_out @ self.out_weights + self.out_bias
        return self.res_dropout.forward(self.output)

    def backward(self, gradient):

        B, T, C = gradient.shape
        gradient = self.res_dropout.backward(gradient)
        
        self.out_weight_grads += cupy.tensordot(self.heads_out.transpose(2, 0, 1), gradient, 2)
        self.out_bias_grads   += cupy.sum(gradient, axis = (0, 1))
        
        gradient = gradient @ self.out_weights.transpose()
        gradient = gradient.reshape((B, T, self.num_heads, self.heads_dim)).transpose(0, 2, 1, 3)

        value_grads = self.dropped_softmax.transpose(0, 1, 3, 2) @ gradient
        gradient = gradient @ self.value.transpose(0, 1, 3, 2)
        gradient = self.attn_dropout.backward(gradient)

        gradient = self.softmax * (gradient - (gradient * self.softmax).sum(axis = -1, keepdims=True))
        gradient = gradient / self.heads_dim**.5

        key_t_grads = self.query.transpose(0, 1, 3, 2) @ gradient
        query_grads = gradient @ self.key_t.transpose(0, 1, 3, 2)

        query_grads = query_grads.transpose(0, 2, 1, 3).reshape(B, T, C)
        key_t_grads = key_t_grads.transpose(0, 3, 1, 2).reshape(B, T, C)
        value_grads = value_grads.transpose(0, 2, 1, 3).reshape(B, T, C)

        gradient = cupy.concatenate((query_grads, key_t_grads, value_grads), axis = 2)

        self.qkv_weight_grads += cupy.tensordot(self.input.transpose(2, 0, 1), gradient, 2)
        self.qkv_bias_grads   += cupy.sum(gradient, axis = (0, 1))
        
        return gradient @ self.qkv_weights.transpose()

    def set_eval(self, eval_mode):
        self.attn_dropout.set_eval(eval_mode)
        self.res_dropout.set_eval(eval_mode)


class TransformerFeedForward(Layer):

    def __init__(self, num_channels, activation, res_dropout_rate = 0.0):
        super(TransformerFeedForward, self).__init__()
        
        self.activation:Layer = activation()
        self.dropout = Dropout(res_dropout_rate)
        
        self.hidden_output = None

        self.weights1 = init_random_tensor((  num_channels, 4*num_channels)) / num_channels**0.5
        self.bias1    = init_zeros_tensor(4*num_channels)
        
        self.weights2 = init_random_tensor((4*num_channels,   num_channels)) / num_channels**0.5
        self.bias2    = init_zeros_tensor(num_channels)

        self.weight_grads1 = self.register_param(self.weights1)
        self.bias_grads1   = self.register_param(self.bias1)

        self.weight_grads2 = self.register_param(self.weights2)
        self.bias_grads2   = self.register_param(self.bias2)


    def forward(self, input):

        self.input = input
        self.hidden_output = self.activation.forward(self.input @ self.weights1 + self.bias1)
        self.output = self.dropout.forward(self.hidden_output @ self.weights2 + self.bias2)

        return self.output

    def backward(self, gradient):
        
        B, T, C = gradient.shape
        
        gradient = self.dropout.backward(gradient)
        self.weight_grads2 += cupy.tensordot(self.hidden_output.transpose(2, 0, 1), gradient, 2)
        self.bias_grads2   += cupy.sum(gradient, axis = (0, 1))
        
        gradient = self.activation.backward(gradient @ self.weights2.transpose())
        self.weight_grads1 += cupy.tensordot(self.input.transpose(2, 0, 1), gradient,  2)
        self.bias_grads1   += cupy.sum(gradient, axis = (0, 1))

        return gradient @ self.weights1.transpose()

    def set_eval(self, eval_mode):
        self.dropout.set_eval(eval_mode)

# TODO: Add RMSnorm
class LayerNorm(Layer):

    def __init__(self, num_channels):
        super(LayerNorm, self).__init__()

        self.channels = num_channels
        self.eps = 1e-5

        self.gamma = init_zeros_tensor(num_channels) + 1
        self.beta  = init_zeros_tensor(num_channels)

        self.gamma_grads  = self.register_param(self.gamma)
        self.beta_grads   = self.register_param(self.beta)

        self.mean = None
        self.var = None
        self.std = None
        self.centered = None
        self.normed = None

    def forward(self, input):

        self.input = input

        self.mean = cupy.mean(input, axis=-1, keepdims=True)
        self.var  = cupy.var(input, axis=-1, keepdims=True)

        self.centered = input - self.mean

        self.std = (self.var + self.eps)**0.5
        self.normed = self.centered / self.std

        self.output = self.gamma * self.normed + self.beta
        return self.output

    def backward(self, gradient):
        
        B, T, C = gradient.shape

        self.gamma_grads += cupy.sum(gradient * self.normed, axis=(0, 1))
        self.beta_grads  += cupy.sum(gradient, axis=(0, 1))

        gradient = gradient * self.gamma
        
        gradient_normed = (gradient - cupy.mean(gradient, axis = -1, keepdims = True)) / self.std
        
        return gradient_normed - self.centered * (cupy.mean(gradient * self.centered, axis = -1, keepdims = True) / self.std**3)


class TransformerBlock(Layer):

    def __init__(self, embed_dim, context_length, num_heads, activation, decoder = False, attn_dropout_rate = 0.0, res_dropout_rate = 0.0):
        super(TransformerBlock, self).__init__()

        self.pre_attn_norm = LayerNorm(embed_dim)
        self.attn_block = MultiHeadAttention(embed_dim, context_length, num_heads, decoder = decoder, 
                                             attn_dropout_rate = attn_dropout_rate,
                                             res_dropout_rate = res_dropout_rate)

        self.pre_ffn_norm = LayerNorm(embed_dim)
        self.ffn = TransformerFeedForward(embed_dim, activation, 
                                          res_dropout_rate = res_dropout_rate)

        self.parameters = [*self.pre_attn_norm.parameters, *self.attn_block.parameters, *self.pre_ffn_norm.parameters, *self.ffn.parameters]
        self.gradients  = [*self.pre_attn_norm.gradients,  *self.attn_block.gradients,  *self.pre_ffn_norm.gradients,  *self.ffn.gradients]
        self.moments    = [*self.pre_attn_norm.moments,    *self.attn_block.moments,    *self.pre_ffn_norm.moments,    *self.ffn.moments]
        self.variances  = [*self.pre_attn_norm.variances,  *self.attn_block.variances,  *self.pre_ffn_norm.variances,  *self.ffn.variances]

    def forward(self,input):
        self.input  = input
        self.output = self.input  + self.attn_block.forward(self.pre_attn_norm.forward(self.input))
        self.output = self.output + self.ffn.forward(self.pre_ffn_norm.forward(self.output))
        return self.output

    def backward(self, gradient):
        gradient += self.pre_ffn_norm.backward(self.ffn.backward(gradient))
        gradient += self.pre_attn_norm.backward(self.attn_block.backward(gradient))
        return gradient

    def set_eval(self, eval_mode):
        self.attn_block.set_eval(eval_mode)
        self.ffn.set_eval(eval_mode)