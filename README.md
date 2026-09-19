# Torchless Neural Nets

Convolutional Neural Nets, Vision Transformers, and GPT's all implemented from scratch in Cuda Numpy (CuPy).
Without torch's tensor or autograd engine, this library implements its own autograd engine with hand derived reverse accumulation / backpropagation calculations.
This library is designed to run on an Nvidia GPU with tensor cores. To run everything on CPU, simply set the `XP_RUNTIME` environmental variable to `CPU` and the backend will load regular NumPy instead of CuPy.

## Modules

* **activations.py** - ReLU, GeLU Approx, SiLU (Swish), and Softmax activations.
* **backend.py** - CuPy or NumPy configuration selector, and device TF32/FP32 tensor initializer functions.
* **layers.py** - Convolution, BatchNorm, MaxPool, AveragePool, Flatten, Dense, Dropout, and Transformer (LayerNorm, Attention, Feed Forward) layers.
* **network.py** - Network framework class with Cross Entropy loss criterion and AdamW optimization.
* **transformer_adapters.py** - ViT image to tokens embedding, ViT MLP classification head, GPT embedding, and GPT token prediction layers.
* **utils** - Layer interface, Residual Layer wrapper, basic image augmentation functions.