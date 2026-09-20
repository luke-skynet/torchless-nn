import backend
from backend import xp, init_zeros_tensor
import numpy as np


# key/value cache: state that survives across forwards during incremental decoding

class Cache:

    """Per layer state for incremental decoding.

    Every cached layer needs the same one thing: the absolute position of the next
    token, so that RoPE and the sinusoidal encodings rotate it to where it actually
    sits rather than to zero. Attention additionally needs somewhere to keep the keys
    and values it has already computed, so it calls allocate(); the rest of the layers
    carry an empty one, which makes `self.cache is not None` a uniform signal that
    generation is running.

    Layers stay in lockstep with no coordination between them, because every layer
    sees the same tokens on the same forward.

    The store is contiguous and in absolute order, which is what lets the block mask
    machinery work unchanged. A sliding window layer keeps `window` entries plus one
    step of headroom and compacts when it overflows: that copy is `window` entries
    once per `step` tokens, against the `window` entries every single token already
    reads, so it amortizes to well under a percent. The alternative, a ring buffer,
    copies nothing but rotates the key axis, which would make the cached masks depend
    on position instead of on geometry alone.
    """

    def __init__(self):

        self.position = 0   # absolute index of the next token
        self.base     = 0   # absolute index of entry 0 of the store
        self.fill     = 0   # entries currently held
        self.capacity = 0

        self.keys   = None
        self.values = None

    def allocate(self, batch_size, num_kv_heads, head_dim, max_length, window, step):

        """Reserve the store. A sliding layer never reads back more than its window."""

        self.capacity = max_length if window is None else min(max_length, window + step)

        shape = (batch_size, num_kv_heads, 1, self.capacity, head_dim)
        self.keys   = init_zeros_tensor(shape)
        self.values = init_zeros_tensor(shape)

    def append(self, key, value):

        """Store this forward's keys and values, and return the whole visible span.

        The span starts at absolute position self.base and is contiguous, so a caller
        holding absolute key bounds indexes it at [k0 - base : k1 - base].
        """

        length = key.shape[-2]

        assert length <= self.capacity, (f"{length} new tokens do not fit a cache of "
                                         f"{self.capacity}: lower the prefill step")

        if self.fill + length > self.capacity:
            # drop the oldest entries the window has already moved past, keeping the
            # buffer contiguous. Nothing still reachable is lost: a query can never
            # look further back than `window`, and the headroom is what guarantees
            # `window` entries survive every compaction
            keep = self.capacity - length

            self.keys  [:, :, :, :keep] = self.keys  [:, :, :, self.fill - keep : self.fill]
            self.values[:, :, :, :keep] = self.values[:, :, :, self.fill - keep : self.fill]

            self.base += self.fill - keep
            self.fill  = keep

        self.keys  [:, :, :, self.fill : self.fill + length] = key
        self.values[:, :, :, self.fill : self.fill + length] = value
        self.fill += length

        return self.keys[:, :, :, :self.fill], self.values[:, :, :, :self.fill]


# layer interface and residual layer wrapper

class Layer:

    # activations stashed by forward for backward to consume. Listed per layer so
    # clear_cache knows what is safe to drop: it must never touch anything the next
    # forward still needs, such as BatchNorm's running statistics or a cached mask.
    CACHED = ("input", "output")

    def __init__(self):

        self.input = None
        self.output = None

        self.parameters = []
        self.gradients = []
        self.moments = []
        self.variances = []

        self.eval_mode = False
        self.inference_only = backend.INFERENCE_MODE

        # incremental decoding state, None whenever the model is not generating.
        # Deliberately absent from CACHED: it has to survive clear_cache, the same way
        # BatchNorm's running statistics do
        self.cache = None

    def register(self, parameter):

        """Register a parameter and allocate its optimizer buffers.

        Returns the gradient buffer, so layers can keep a named handle on it. Under
        inference mode nothing is allocated and None is returned: the four lists stay
        parallel because they all stay empty, which quietly makes zero_grad and the
        optimizer step no-ops, and makes backward fail loudly on the None.
        """

        self.parameters.append(parameter)

        if self.inference_only:
            return None

        gradient = init_zeros_tensor(parameter.shape)

        self.gradients.append(gradient)
        self.moments.append(init_zeros_tensor(parameter.shape))
        self.variances.append(init_zeros_tensor(parameter.shape))

        return gradient

    def clear_cache(self):
        for name in self.CACHED:
            setattr(self, name, None)

    def start_cache(self, batch_size, max_length, step = 1):

        """Begin incremental decoding.

        Every layer gets a Cache, so that `self.cache is not None` is a uniform signal
        that generation is running; only the layers with something to keep across
        forwards allocate anything into it. Composite layers recurse, the way
        clear_cache and set_eval do.
        """

        self.cache = Cache()

    def stop_cache(self):
        self.cache = None

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError

    def set_eval(self, eval_mode):
        self.eval_mode = eval_mode

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

    def clear_cache(self):
        super(Residual, self).clear_cache()
        for layer in self.layers:
            layer.clear_cache()

    def start_cache(self, batch_size, max_length, step = 1):
        super(Residual, self).start_cache(batch_size, max_length, step)
        for layer in self.layers:
            layer.start_cache(batch_size, max_length, step)

    def stop_cache(self):
        super(Residual, self).stop_cache()
        for layer in self.layers:
            layer.stop_cache()
            

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
    import cv2
    
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
