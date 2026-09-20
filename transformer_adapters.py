from backend import xp, FLOAT_TYPE, init_random_tensor, init_zeros_tensor, init_weight_tensor

from utils import Layer

class VitProjector(Layer):

    CACHED = ("input", "output", "tokens", "embeddings")

    def __init__(self, input_size:tuple, patch_size:tuple, embedding_dim, num_registers = 0, cls_token = True):
        super(VitProjector, self).__init__()

        self.batch_size = 0
        self.channels, self.height, self.width = input_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_registers = num_registers
        self.cls_token = cls_token

        self.patches_height = self.height // self.patch_size[0]
        self.patches_width  = self.width  // self.patch_size[1]

        self.sequence_length = self.patches_height * self.patches_width
        self.token_dim = self.patch_size[0] * self.patch_size[1] * self.channels

        self.cls_size = int(self.cls_token)
        self.cls_reg_size = self.cls_size + self.num_registers
        self.total_length = self.cls_reg_size + self.sequence_length

        self.tokens = None
        self.embeddings = None

        self.projection_weights    = init_weight_tensor((self.token_dim, self.embedding_dim), self.token_dim**0.5)
        self.projection_bias       = init_zeros_tensor(self.embedding_dim)

        if self.cls_reg_size:
            self.cls_reg_tokens = init_weight_tensor((self.cls_reg_size, self.embedding_dim), self.embedding_dim**0.5)

        # Register rows stay frozen at zero, allowing one positional add for all tokens.
        self.positional_encodings  = init_zeros_tensor((self.total_length, self.embedding_dim))

        self.projection_grads           = self.register(self.projection_weights)
        self.projection_bias_grads      = self.register(self.projection_bias)

        if self.cls_reg_size:
            self.cls_reg_token_grads    = self.register(self.cls_reg_tokens)

        self.positional_encoding_grads  = self.register(self.positional_encodings)

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

        self.embeddings = xp.tensordot(self.tokens, self.projection_weights, axes = 1) + self.projection_bias

        if self.cls_reg_size:
            cls_reg_batch = xp.broadcast_to(self.cls_reg_tokens,
                                             (self.batch_size,
                                              self.cls_reg_size,
                                              self.embedding_dim))

            self.output = xp.concatenate([cls_reg_batch, self.embeddings], axis = 1)

        else:
            self.output = self.embeddings

        self.output += self.positional_encodings

        return self.output

    def backward(self, gradient):

        self.positional_encoding_grads += gradient.sum(axis = 0)

        if self.num_registers:
            # These positional rows are padding, not trainable register positions.
            register_start = self.cls_size
            register_end   = self.cls_size + self.num_registers

            self.positional_encoding_grads[register_start:register_end] = 0

        if self.cls_reg_size:
            self.cls_reg_token_grads += gradient[:, :self.cls_reg_size].sum(axis = 0)
            gradient = gradient[:, self.cls_reg_size:]

        
        self.projection_grads += xp.tensordot(self.tokens.transpose(2,0,1), gradient, 2)
        self.projection_bias_grads += gradient.sum(axis = (0,1))

        gradient = xp.tensordot(gradient, self.projection_weights.T, axes = 1)

        gradient = gradient.reshape((self.batch_size,
                                     self.patches_height, self.patches_width,
                                     self.patch_size[0], self.patch_size[1], self.channels))

        gradient = gradient.transpose(0,5,1,3,2,4)
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
        self.bias_grads += xp.sum(gradient, axis = 0)

        gradient = gradient @ self.weights.transpose()
        gradient = gradient[:,xp.newaxis,:]

        return xp.concatenate((gradient, init_zeros_tensor(self.input.shape)[:,:-1,:]), axis = 1)

class GPTEmbeddingTable(Layer):

    """One weight and optimizer state shared by a front/back embedding pair.

    The front exposes the state to Network; the back only accumulates into it.
    Existing arrays are wrapped without copying. Construction honors inference_mode
    and empty_weights through the same helpers as other layers.
    """

    def __init__(self, vocab_size, embed_size, table = None, scale = 1.0):
        super(GPTEmbeddingTable, self).__init__()
        if table is not None and table.shape != (vocab_size, embed_size):
            raise ValueError("embedding table shape must match vocab_size and embed_size")
        self.table = (init_weight_tensor((vocab_size, embed_size), scale)
                      if table is None else table)
        self.table_grads = self.register(self.table)


class GPTEmbedFront(Layer):

    """Token embedding lookup, with sinusoidal, learned, or no absolute positions.

    positional = "none" is what a RoPE model wants: position enters inside attention
    instead, so nothing is added here.

    scale_embeddings multiplies by sqrt(embedding_dim), which is what Gemma does to
    put the embedding scale on the same footing as the residual stream.
    """

    def __init__(self, table: GPTEmbeddingTable, context_length, positional = "sinusoidal", scale_embeddings = False):
        super(GPTEmbedFront, self).__init__()

        assert positional in {"sinusoidal", "learned", "none"}

        if not isinstance(table, GPTEmbeddingTable):
            raise TypeError("GPTEmbedFront requires a GPTEmbeddingTable")
        if table.inference_only != self.inference_only:
            raise ValueError("shared embedding table and layer must use the same inference mode")
        
        self.embedding_table = table
        self.table = table.table
        self.table_grads = table.table_grads
        
        # Register embedding table once here
        self.parameters = list(table.parameters)
        self.gradients = list(table.gradients)
        self.moments = list(table.moments)
        self.variances = list(table.variances)

        self.scale = self.table.shape[1]**0.5 if scale_embeddings else 1.0

        self.positional_encoding = None
        self.positional_encoding_grads = None
        if positional == "sinusoidal":
            pos = xp.arange(context_length)[:, None]
            i   = xp.arange(self.table.shape[1])[None, :]

            self.positional_encoding = pos / 10000**(2 * (i // 2) / self.table.shape[1])
            self.positional_encoding[:, 0::2] = xp.sin(self.positional_encoding[:, 0::2])
            self.positional_encoding[:, 1::2] = xp.cos(self.positional_encoding[:, 1::2])
            self.positional_encoding = self.positional_encoding.astype(FLOAT_TYPE, copy = False)
        elif positional == "learned":
            self.positional_encoding = init_zeros_tensor((context_length, self.table.shape[1]))
            self.positional_encoding_grads = self.register(self.positional_encoding)

    def forward(self, input):
        self.input  = input
        self.output = self.table[self.input]

        # while generating, this forward continues a sequence rather than starting one,
        # so the encodings have to be read from where the tokens actually sit
        position = self.cache.position if self.cache is not None else 0

        if self.scale != 1.0:
            self.output = self.output * self.scale
            
        if self.positional_encoding is not None:
            if position + self.input.shape[1] > self.positional_encoding.shape[0]:
                raise ValueError("input exceeds the absolute position table's context length")
            self.output = self.output + self.positional_encoding[position : position + self.input.shape[1],:]

        if self.cache is not None:
            self.cache.position = position + self.input.shape[1]

        return self.output

    def backward(self, gradient):
        
        if self.cache is not None:
            raise RuntimeError("backward() while an embedding cache is active is unsupported")
        
        xp.add.at(self.table_grads, self.input, gradient * self.scale)
        
        if self.positional_encoding_grads is not None:
            self.positional_encoding_grads[:self.input.shape[1]] += gradient.sum(axis=0)
            
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

    def __init__(self, table: GPTEmbeddingTable):
        super(GPTEmbedBack, self).__init__()

        if not isinstance(table, GPTEmbeddingTable):
            raise TypeError("GPTEmbedBack requires a GPTEmbeddingTable")
        if table.inference_only != self.inference_only:
            raise ValueError("shared embedding table and layer must use the same inference mode")
        self.embedding_table = table
        self.table = table.table
        self.table_grads = table.table_grads
        # The front owns registration, zeroing, and optimizer updates.

    def forward(self, input):
        if self.cache is not None:
            input = input[:, -1:, :]
        self.input = input
        self.output = xp.tensordot(self.input, self.table.transpose(), axes = 1)
        return self.output

    def backward(self, gradient):
        self.table_grads += xp.tensordot(self.input.transpose(2, 0, 1), gradient, 2).transpose()
        return xp.tensordot(gradient, self.table, axes = 1)
