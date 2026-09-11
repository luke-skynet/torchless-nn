import cupy
import cupyx

from utils import Layer, FLOAT_TYPE, init_random_tensor, init_zeros_tensor
from layers import Dropout

class VitProjector(Layer):

    def __init__(self, input_size:tuple, patch_size:tuple, embedding_dim, num_registers = 0):
        super(VitProjector, self).__init__()

        self.batch_size = 0
        self.channels, self.height, self.width = input_size
        self.patch_size = patch_size

        self.patches_height = self.height // self.patch_size[0]
        self.patches_width  = self.width  // self.patch_size[1]

        self.sequence_length = self.patches_height * self.patches_width
        self.token_dim = self.patch_size[0] * self.patch_size[1] * self.channels
        self.embedding_dim = embedding_dim
        self.cls_reg_size = 1 + num_registers

        self.tokens = None
        self.embeddings = None

        self.projection            = init_random_tensor((self.token_dim,    self.embedding_dim)) / self.token_dim**0.5
        self.cls_reg_tokens        = init_random_tensor((self.cls_reg_size, self.embedding_dim)) / self.embedding_dim**0.5
        self.positional_embeddings = init_zeros_tensor((self.cls_reg_size + self.sequence_length, self.embedding_dim))

        self.projection_grads              = self.register_param(self.projection)
        self.cls_reg_token_grads           = self.register_param(self.cls_reg_tokens)
        self.positional_embeddings_grads   = self.register_param(self.positional_embeddings)

    def forward(self, input):

        self.input = input
        self.batch_size = self.input.shape[0]

        reshape = self.input.reshape((self.batch_size,
                                      self.channels,
                                      self.patches_height, self.patch_size[0],
                                      self.patches_width, self.patch_size[1]))

        permute = reshape.transpose(0,2,4,3,5,1)
        self.tokens = permute.reshape(self.batch_size,
                                      self.sequence_length,
                                      self.token_dim)

        self.embeddings = self.tokens @ self.projection

        class_token_batch = cupy.tile(self.cls_reg_tokens, (self.batch_size, 1, 1))
        self.embeddings = cupy.concatenate([class_token_batch, self.embeddings], axis = 1)

        self.output = self.embeddings + self.positional_embeddings
        return self.output

    def backward(self, gradient):
        
        self.positional_embeddings_grads += gradient.sum(axis = 0)

        self.cls_reg_token_grads += gradient[:,:self.cls_reg_size,:].sum(axis = 0)
        gradient = gradient[:,self.cls_reg_size:,:]

        self.projection_grads += cupy.tensordot(self.tokens.transpose(2, 0, 1), gradient, 2)
        gradient = gradient @ self.projection.transpose()

        gradient = gradient.reshape((self.batch_size,
                                     self.patches_height,  self.patches_width,
                                     self.patch_size[0], self.patch_size[1], self.channels))

        gradient = gradient.transpose(0, 5, 1, 3, 2, 4)
        gradient = gradient.reshape((self.batch_size, self.channels, self.height, self.width))
        return gradient


class VitMLPHead(Layer):

    def __init__(self, in_channels, out_channels):
        super(VitMLPHead, self).__init__()

        self.weights = init_random_tensor((in_channels, out_channels)) / in_channels**0.5
        self.bias    = init_zeros_tensor(out_channels)

        self.class_tokens = None

        self.weight_grads = self.register_param(self.weights)
        self.bias_grads   = self.register_param(self.bias)

    def forward(self, input):

        self.input = input
        self.class_tokens = input[:,0,:]
        self.output = self.class_tokens @ self.weights + self.bias

        return self.output

    def backward(self, gradient):

        self.weight_grads += self.class_tokens.transpose() @ gradient
        self.bias_grads += cupy.sum(gradient, axis = 0)

        gradient = gradient @ self.weights.transpose()
        gradient = gradient[:,cupy.newaxis,:]

        return cupy.concatenate((gradient, init_zeros_tensor(self.input.shape)[:,:-1,:]), axis = 1)


class GPTEmbeddingTable:
    def __init__(self, vocab_size, embed_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.table         = init_random_tensor((vocab_size, embed_size))
        self.table_grads   = init_zeros_tensor(self.table.shape)
        self.table_moments = init_zeros_tensor(self.table.shape)
        self.table_vars    = init_zeros_tensor(self.table.shape)


class GPTEmbedFront(Layer):

    def __init__(self, embedding_table:GPTEmbeddingTable, context_length, positional_embedding = "learned"):    
        super(GPTEmbedFront, self).__init__()
        assert positional_embedding == "learned" or positional_embedding == "sinusoidal"
        
        self.positional_embedding = positional_embedding
        self.pos_embedding_table = None

        if self.positional_embedding == "sinusoidal":
            pos = cupy.arange(context_length)[:, None]
            i   = cupy.arange(embedding_table.embed_size)[None, :]

            self.pos_embedding_table = pos / 10000**(2 * (i // 2) / embedding_table.embed_size)
            self.pos_embedding_table[:, 0::2] = cupy.sin(self.pos_embedding_table[:, 0::2])
            self.pos_embedding_table[:, 1::2] = cupy.cos(self.pos_embedding_table[:, 1::2])
            self.pos_embedding_table = self.pos_embedding_table.astype(FLOAT_TYPE, copy = False)
        else:
            self.pos_embedding_table = init_zeros_tensor((context_length, embedding_table.embed_size))
            self.pos_embedding_table_grads = self.register_param(self.pos_embedding_table)
    
        self.embedding_table = embedding_table
        self.embeddings = None
        
        self.parameters.append(self.embedding_table.table)
        self.gradients.append(self.embedding_table.table_grads)
        self.moments.append(self.embedding_table.table_moments)
        self.variances.append(self.embedding_table.table_vars)

    def forward(self, input):
        self.input = input
        self.embeddings = self.embedding_table.table[input]
        self.output = self.embeddings + self.pos_embedding_table[:self.input.shape[1],:]
        
        return self.output

    def backward(self, gradient):
        
        B, T, C = gradient.shape
        
        cupyx.scatter_add(self.embedding_table.table_grads, self.input, gradient)
        
        if self.positional_embedding == "learned":
            self.pos_embedding_table_grads[:T] += gradient.sum(axis = 0)
            
        return None


class GPTEmbedBack(Layer):

    def __init__(self, embedding_table:GPTEmbeddingTable):
        super(GPTEmbedBack, self).__init__()
        self.embedding_table = embedding_table
        # parameter updates handled in GPTEmbedFront
    
    def forward(self, input):
        self.input = input
        self.output = self.input @ self.embedding_table.table.transpose()
        return self.output

    def backward(self, gradient):
        self.embedding_table.table_grads += cupy.tensordot(self.input.transpose(2, 0, 1), gradient, 2).transpose()
        return gradient @ self.embedding_table.table