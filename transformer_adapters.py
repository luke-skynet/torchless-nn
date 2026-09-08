import cupy

from utils import Layer, FLOAT_TYPE, init_random_tensor, init_zeros_tensor, scatter_add

class VitProjector(Layer):

    CACHED = ("input", "output", "tokens", "embeddings")

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

        
        
        self.projection_grads            = self.register(self.projection)
        self.cls_reg_token_grads         = self.register(self.cls_reg_tokens)
        self.positional_embeddings_grads = self.register(self.positional_embeddings)

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

        self.projection_grads += cupy.tensordot(self.tokens.transpose(2, 0, 1), gradient, 2) / gradient.shape[1]
        gradient = gradient @ self.projection.transpose()

        gradient = gradient.reshape((self.batch_size,
                                     self.patches_height,  self.patches_width,
                                     self.patch_size[0], self.patch_size[1], self.channels))

        gradient = gradient.transpose(0, 5, 1, 3, 2, 4)
        gradient = gradient.reshape((self.batch_size, self.channels, self.height, self.width))
        return gradient

class VitMLPHead(Layer):

    CACHED = ("input", "output", "class_tokens")

    def __init__(self, in_channels, out_channels):
        super(VitMLPHead, self).__init__()

        self.weights = init_random_tensor((in_channels, out_channels)) / in_channels**0.5
        self.bias    = init_zeros_tensor(out_channels)

        self.class_tokens = None

        self.weight_grads = self.register(self.weights)
        self.bias_grads   = self.register(self.bias)

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

class GPTEmbedFront(Layer):

    """Token embedding lookup, with optional absolute sinusoidal positions.

    positional = "none" is what a RoPE model wants: position enters inside attention
    instead, so nothing is added here.

    scale_embeddings multiplies by sqrt(embedding_dim), which is what Gemma does to
    put the embedding scale on the same footing as the residual stream.
    """

    def __init__(self, table, context_length, positional = "sinusoidal", scale_embeddings = False):
        super(GPTEmbedFront, self).__init__()

        assert positional in {"sinusoidal", "none"}

        self.table         = table

        self.scale = table.shape[1]**0.5 if scale_embeddings else 1.0

        self.positional_encoding = None
        if positional == "sinusoidal":
            pos = cupy.arange(context_length)[:, None]
            i   = cupy.arange(table.shape[1])[None, :]

            self.positional_encoding = pos / 10000**(2 * (i // 2) / table.shape[1])
            self.positional_encoding[:, 0::2] = cupy.sin(self.positional_encoding[:, 0::2])
            self.positional_encoding[:, 1::2] = cupy.cos(self.positional_encoding[:, 1::2])
            self.positional_encoding = self.positional_encoding.astype(FLOAT_TYPE, copy = False)

        self.table_grads = self.register(self.table)

    def forward(self, input):
        self.input  = input
        self.output = self.table[self.input]

        # while generating, this forward continues a sequence rather than starting one,
        # so the encodings have to be read from where the tokens actually sit
        position = self.cache.position if self.cache is not None else 0

        if self.scale != 1.0:
            self.output = self.output * self.scale
        if self.positional_encoding is not None:
            self.output = self.output + self.positional_encoding[position : position + self.input.shape[1],:]

        if self.cache is not None:
            self.cache.position = position + self.input.shape[1]

        return self.output

    def backward(self, gradient):
        scatter_add(self.table_grads, self.input, gradient * (self.scale / gradient.shape[1]))
        return None # nothing upstream of the token ids to receive a gradient


class GPTEmbedBack(Layer):

    """Unembedding, sharing the table with GPTEmbedFront.

    While a cache is active this returns only the final position's logits. That is the
    one place incremental decoding changes what a forward means, and it is not an
    optimization that can be skipped: the logits are the largest tensor in the model,
    (batch, sequence, 262144) running to 1 GiB per 1024 tokens at Gemma's vocabulary.
    Unembedding a whole prefill chunk to sample one token from the last row of it is
    the most expensive mistake available here.
    """

    def __init__(self, table):
        super(GPTEmbedBack, self).__init__()

        self.table         = table
        self.table_grads = self.register(self.table)

    def forward(self, input):
        if self.cache is not None:
            input = input[:, -1:, :]
        self.input = input
        self.output = self.input @ self.table.transpose()
        return self.output

    def backward(self, gradient):
        self.table_grads += cupy.tensordot(self.input.transpose(2, 0, 1), gradient, 2).transpose() / gradient.shape[1]
        return gradient @ self.table