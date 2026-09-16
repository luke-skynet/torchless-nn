"""Adam updates: one CUDA launch across tensors, or a NumPy reference path."""
import numpy as np

from backend import xp


_CHUNK_SIZE = 4096
_CUDA_SOURCE = r'''
// NVRTC provides sqrt directly; avoid depending on host math headers.

template <typename T>
__device__ void update_chunk(const unsigned long long* info,
                             unsigned long long start,
                             double learning_rate, double weight_decay,
                             double num_samples, double correction1,
                             double correction2, double epsilon) {
    T* param = reinterpret_cast<T*>(info[0]);
    T* grad = reinterpret_cast<T*>(info[1]);
    T* moment = reinterpret_cast<T*>(info[2]);
    T* variance = reinterpret_cast<T*>(info[3]);
    const unsigned long long stop = start + 4096 < info[4] ? start + 4096 : info[4];
    const T decay = info[5] ? T(weight_decay) : T(0);

    for (unsigned long long i = start + threadIdx.x; i < stop; i += blockDim.x) {
        const T g = grad[i] / T(num_samples);
        const T m = T(0.9) * moment[i] + T(1.0 - 0.9) * g;
        const T v = T(0.999) * variance[i] + T(1.0 - 0.999) * (g * g);
        const T m_hat = m / T(correction1);
        const T v_hat = v / T(correction2);
        const T p = param[i];
        param[i] = p - T(learning_rate) * (m_hat / (sqrt(v_hat) + T(epsilon)) + decay * p);
        moment[i] = m;
        variance[i] = v;
        grad[i] = T(0);
    }
}

extern "C" __global__ void adam_step(
    const unsigned long long* tensors, const unsigned long long* chunks,
    double learning_rate, double weight_decay, double num_samples,
    double correction1, double correction2, double epsilon) {
    const unsigned long long* info = tensors + chunks[blockIdx.x * 2] * 7;
    const unsigned long long start = chunks[blockIdx.x * 2 + 1];
    if (info[6] == 4)
        update_chunk<float>(info, start, learning_rate, weight_decay, num_samples,
                            correction1, correction2, epsilon);
    else
        update_chunk<double>(info, start, learning_rate, weight_decay, num_samples,
                             correction1, correction2, epsilon);
}
'''


def _array_key(array):
    if not isinstance(array, xp.ndarray):
        raise TypeError('Adam arrays must belong to the selected backend')
    pointer = array.ctypes.data if xp is np else array.data.ptr
    device = None if xp is np else array.device.id
    return (pointer, array.shape, array.strides, array.dtype.str, device)


def _storage_bounds(array):
    """Conservative byte bounds, including strided NumPy views."""
    start = _array_key(array)[0]
    lower = sum(min(0, (n - 1) * s) for n, s in zip(array.shape, array.strides))
    upper = sum(max(0, (n - 1) * s) for n, s in zip(array.shape, array.strides))
    return start + lower, start + upper + array.itemsize


class Adam:
    """Update and clear gradients, retaining the layers' existing array storage.

    Entries are (parameter, gradient, moment, variance, decay_enabled). CUDA
    supports contiguous float32 and float64 tensors, including both in one launch.
    Metadata is rebuilt when storage, shape, dtype, or decay policy changes.
    """

    def __init__(self):
        self._signature = None
        self.entries = []
        self._kernel = None

    def _prepare(self, entries):
        signature = tuple((tuple(_array_key(a) for a in entry[:4]), bool(entry[4]))
                          for entry in entries)
        if signature == self._signature:
            return

        unique = []
        seen = {}
        storage = []
        devices = set()
        for entry, (keys, decay) in zip(entries, signature):
            param, grad, moment, variance, _ = entry
            if len(keys) != 4:
                raise ValueError('Adam requires a parameter and three state arrays')
            if keys[0] in seen:
                if seen[keys[0]] != (keys, decay):
                    raise ValueError('Shared parameters must share Adam state and decay policy')
                continue
            seen[keys[0]] = (keys, decay)
            for array in (param, grad, moment, variance):
                if array.shape != param.shape or array.dtype != param.dtype:
                    raise ValueError('Adam state must match parameter shape and dtype')
                if array.dtype not in (np.dtype('float32'), np.dtype('float64')):
                    raise ValueError('Adam requires float32 or float64 arrays')
                if xp is not np and not array.flags.c_contiguous:
                    raise ValueError('CUDA Adam requires C-contiguous arrays')
                if xp is np and not array.flags.writeable:
                    raise ValueError('Adam arrays must be writable')
                devices.add(_array_key(array)[4])
                if array.size:
                    storage.append(_storage_bounds(array))
            unique.append((param, grad, moment, variance, decay))

        if len(devices) > 1:
            raise ValueError('Adam arrays must be on a single device')
        storage.sort()
        if any(right[0] < left[1] for left, right in zip(storage, storage[1:])):
            raise ValueError('Adam arrays must not overlap except for identical shared entries')

        if xp is not np:
            device = next(iter(devices), xp.cuda.runtime.getDevice())
            tensors = []
            chunks = []
            for index, (p, g, m, v, decay) in enumerate(unique):
                tensors.append([p.data.ptr, g.data.ptr, m.data.ptr, v.data.ptr,
                                p.size, int(decay), p.itemsize])
                chunks.extend((index, start) for start in range(0, p.size, _CHUNK_SIZE))
            with xp.cuda.Device(device):
                self._tensors = xp.asarray(np.asarray(tensors, dtype=np.uint64).reshape(-1, 7))
                self._chunks = xp.asarray(np.asarray(chunks, dtype=np.uint64).reshape(-1, 2))
                self._metadata_ready = xp.cuda.Event()
                self._metadata_ready.record()
                if self._kernel is None:
                    self._kernel = xp.RawKernel(_CUDA_SOURCE, 'adam_step', options=('--fmad=false',))
            self._device = device
            self._num_chunks = len(chunks)

        # Keep strong references to every allocation represented by a raw pointer.
        self.entries = unique
        self._signature = signature

    def step(self, entries, learning_rate, weight_decay, t, num_samples, eps=1e-7):
        if t < 1 or num_samples <= 0 or eps < 0:
            raise ValueError('Adam requires t >= 1, num_samples > 0, and eps >= 0')
        self._prepare(list(entries))
        if xp is np:
            self._reference_step(learning_rate, weight_decay, t, num_samples, eps)
        elif self._num_chunks:
            with xp.cuda.Device(self._device):
                xp.cuda.get_current_stream().wait_event(self._metadata_ready)
                self._kernel((self._num_chunks,), (256,), (
                    self._tensors, self._chunks,
                    np.float64(learning_rate), np.float64(weight_decay), np.float64(num_samples),
                    np.float64(1 - 0.9**t), np.float64(1 - 0.999**t), np.float64(eps)))

    def _reference_step(self, learning_rate, weight_decay, t, num_samples, eps):
        beta1, beta2 = 0.9, 0.999
        for param, grad, moment, variance, decay in self.entries:
            grad /= num_samples
            moment *= beta1
            moment += (1 - beta1) * grad
            grad *= grad
            grad *= (1 - beta2)
            variance *= beta2
            variance += grad
            mom_hat = moment / (1 - beta1**t)
            var_hat = variance / (1 - beta2**t)
            lmda = weight_decay if decay else 0.0
            param -= learning_rate * (mom_hat / (var_hat**0.5 + eps) + lmda * param)
            grad.fill(0)
