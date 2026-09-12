# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Torchless Neural Nets is a deep learning library implementing CNNs, Vision Transformers, and GPTs from scratch using CuPy (CUDA-accelerated NumPy). All gradient calculations are hand-derived with reverse accumulation—no PyTorch autograd engine is used. The library runs on Nvidia GPUs with tensor core support but is CPU-compatible by replacing `cupy.` with `np.`.

## Development Environment

**Primary workflow:** Development is done in Jupyter notebooks (see demo.ipynb)

**Enable TF32 and CuPy accelerators:**
```python
import os
os.environ['CUPY_TF32'] = "1"
os.environ['CUPY_ACCELERATORS'] = "cub,cutensor"

from cupy.cuda import cublas
cublas_handle = cupy.cuda.Device().cublas_handle
cublas.setMathMode(cublas_handle, cublas.CUBLAS_TENSOR_OP_MATH)
```

**Key dependencies:** CuPy, NumPy, tqdm, seaborn, matplotlib, opencv-python (cv2)

**Datasets location:** All datasets live in `datasets/` directory (CIFAR-10, shakespeare.txt, lotr.txt)

## Architecture Overview

### Layer Interface (utils.py)

All layers inherit from the `Layer` base class which defines:
- `forward()` / `backward()` methods for forward/backward propagation
- `parameters`, `gradients`, `moments`, `variances` lists for AdamW optimizer
- `eval_mode` flag for train/test behavior (affects Dropout, BatchNorm)
- `zero_grad()` / `zero_adam()` for clearing accumulators

**Important:** Every layer must maintain these four lists in parallel order for the optimizer to work correctly. Use `self.register(parameter)` rather than building the lists by hand — it appends to all four and returns the gradient buffer for the layer to keep a named handle on:

```python
self.weights = init_random_tensor((input_size, output_size)) / input_size**0.5
self.bias    = init_zeros_tensor(output_size)

self.weight_grads = self.register(self.weights)
self.bias_grads   = self.register(self.bias)
```

- `CACHED` names the activations a layer stashes in `forward` for `backward` to consume, and `clear_cache()` drops them. It must never list anything the *next* forward still needs (BatchNorm's running statistics, a cached positional encoding, the KV `cache`). Composite layers override `clear_cache()` to recurse into the layers they own
- `inference_only` records whether the layer was built inside `inference_mode()`
- `cache` holds incremental decoding state and is `None` outside generation; `start_cache()` / `stop_cache()` bracket it, and composite layers recurse the way `clear_cache` does

### Network Framework (network.py)

`Network` class wraps a list of layers and provides:
- `train()` method with AdamW optimizer and gradient accumulation via `batches_per_step`
- `evaluate()` for test set evaluation
- `predict()` for inference
- `CrossEntropy` loss combines softmax + cross-entropy for numerical stability. It builds **no one-hot**: each label selects one entry per row, so `gradients()` subtracts 1 at those positions in a copy of the logits and `loss()` gathers them. The identity matrix it used to hold was `num_classes` squared — 256 GiB at a 262144 token vocabulary — and indexing that expanded to a full `(batch, sequence, vocab)` array on every call

**Weight decay handling:** AdamW applies weight decay to all parameters EXCEPT:
- 1D tensors (biases, normalization parameters)
- Parameters in `VitProjector`, `VitMLPHead`, `GPTEmbedFront`, `GPTEmbedBack` (see network.py:69-71)

### Core Layers (layers.py)

**Convolution:** Uses `cupy.lib.stride_tricks.sliding_window_view` for efficient im2col-style convolution with Einstein summation (`einsum`)

**BatchNorm:** Tracks `running_mean` and `running_var` with momentum=0.1. Uses eval_mode flag to switch between training stats and running stats.

**MaxPool:** Reshape-based pooling, with a boolean mask routing the gradient to the argmax

**AveragePool:** Reshape-based pooling. The backward pass divides by the window size and broadcasts — it does **not** build a mask. It previously multiplied by a full input-sized array holding a single constant, which additionally promoted every downstream gradient to float64 (`cupy.ones()` has no dtype by default), silently breaking the FP32/TF32 invariant

**Dense:** Standard fully-connected layer with matrix multiplication

**Dropout:** Uses RNG with inverted dropout (scale by 1/(1-p) during training)

**MultiHeadAttention:**
- Fuses QKV projection into single matrix multiply **when called with default arguments**, so existing checkpoints keep loading. Any of `num_kv_heads`, `head_dim` or `kv_shared` switches to separate Q/K/V projections
- Heads are laid out `(batch, kv_heads, group, sequence, head_dim)` so one KV head broadcasts across the query heads sharing it (GQA) with no materialized repeat; plain MHA is the `group=1` case
- `head_dim` is decoupled from `embedding_dim // num_heads` (Gemma 4 31B uses 5376 embed with 256-wide heads)
- `qk_norm=True` applies RMSNorm per head to Q and K before attention
- `rope=` applies rotary embeddings to Q and K; `sliding_window=` restricts the causal mask to a band
- `kv_shared=True` is Gemma's `attention_k_eq_v`: one projection serves as both key and value. The value branches off **before** QK norm and RoPE, since those condition attention logits and must not touch values
- `chunk_size` sets the query block size. Queries are always processed in a loop over blocks; `chunk_size=None` means one block, which is exactly the unchunked computation (bit-identical, not merely equivalent). There is only one code path — small models simply run one iteration
- Chunking bounds peak memory by the block size rather than the sequence length, and for a sliding-window layer it also **skips computing** the scores the mask would discard. That is a real FLOP saving: at T=8192 with window 1024 it is ~5.7x less work, at T=65536 ~43x. Even plain causal layers save ~2x by skipping the upper triangle
- Keep `chunk_size >= 256` in practice. CuPy dispatch is asynchronous, so Python loop overhead hides under GPU execution — but only while each block carries enough work. Below that you go dispatch-bound and the GPU idles between blocks
- Bidirectional (`decoder=False`) layers get no saving from chunking, since every query needs every key; leave them at one block
- Masks are never materialized at `(T, T)`. `_BLOCK_MASKS` caches them by *block geometry* (query-key offset, block shape, window) at module scope, so all layers share them and the cache does not grow with sequence length. A model declared at Gemma's 262144 context would otherwise have allocated a 275 GB mask per layer. A block where every key is visible caches **`None`** rather than an array of zeros, and the caller skips the add — during single token decode that is *every* block, sliding and global alike, so generation's hot path adds no mask at all
- The block loop runs on **absolute** positions, so `start_cache()` makes it serve incremental decoding with no second code path: uncached, the first query sits at 0 and keys span `[0, T)`, which is exactly the previous behaviour. Keys and values go into the cache already QK-normed and rotated, since both depend only on the token and its position. `backward` raises while a cache is active
- Attention gradients use `cupy.tensordot(..., 2) / T` pattern to average over sequence dimension

**GatedFeedForward:** Implements SwiGLU-style gating (activation(x @ W1) * (x @ W2)). `multiplier` sets the hidden expansion (default 4, which is also Gemma 4's 5376 -> 21504). Passing `GeLU` as the activation makes this GeGLU, exactly Gemma's FFN — the library's GeLU is the tanh approximation, matching `gelu_pytorch_tanh`

**LayerNorm:** Per-token normalization for transformers (normalizes over embedding dimension)

**RMSNorm:** Rescale-only normalization (no mean subtraction, no beta) used by Gemma-style models. Written shape-agnostically over the last axis so the same layer serves `(batch, sequence, channels)` activations and the `(batch, kv_heads, group, sequence, head_dim)` tensors QK norm operates on. Gemma checkpoints store this weight as `(gamma - 1)`

**RotaryEmbedding:** RoPE for the head-split Q and K tensors. Holds no parameters (just cached cos/sin), so one instance is shared across every layer that uses it. `partial_rotary_factor < 1` gives Gemma 4's p-RoPE, rotating only the leading fraction of each head's dimensions

**Softcap:** `cap * tanh(x / cap)`, Gemma's final logit cap of 30. Sits between the unembedding and SoftMax and composes with the fused softmax+cross-entropy gradient for free, since `SoftMax.backward` is the identity

**TransformerBlock:** Pre-norm architecture (norm before attention/FFN) with residual connections. `norm=RMSNorm` swaps the normalization; `post_norm=True` adds Gemma's sandwich arrangement, a second norm on each branch output before it rejoins the residual stream:

```
h = x + post_attn_norm(attn(pre_attn_norm(x)))
y = h + post_ffn_norm(ffn(pre_ffn_norm(h)))
```

Attention keywords (`num_kv_heads`, `head_dim`, `qk_norm`, `rope`, `sliding_window`, `kv_shared`) pass straight through to MultiHeadAttention.

**gemma_layer_types(num_layers, pattern=6):** Gemma's local/global interleave — every `pattern`-th layer attends globally and the last layer always does. At 60 layers this reproduces Gemma 4 31B's `layer_types` exactly

### Activations (activations.py)

- **ReLU:** Basic rectified linear
- **GeLU:** Approximation using tanh
- **SiLU (Swish):** x * sigmoid(x) approximated with tanh
- **SoftMax:** Numerically stable with max normalization. Temperature parameter for generation.

**SoftMax fusion:** `SoftMax(fused_loss=True)` (the default) returns its gradient unchanged, because `CrossEntropy.gradients` already produced the gradient with respect to the softmax *input*. That is only correct as the **final** layer — anywhere else it silently drops the softmax Jacobian, so `Network.__init__` raises if a fused SoftMax is not last. Pass `SoftMax(fused_loss=False)` to get the real Jacobian, `y * (g - sum(g*y))`, usable anywhere. Both paths divide by `temperature`.

### Transformer Adapters (transformer_adapters.py)

**VitProjector:** Converts images to patch embeddings
- Reshapes input into patches via tensor reshaping/transpose
- Projects patches to embedding dimension
- Prepends CLS token + optional register tokens
- Adds learned positional embeddings

**VitMLPHead:** Extracts CLS token (position 0) and applies linear projection for classification

**GPTEmbedFront:** Token embedding layer. Looks up rows by fancy indexing and accumulates gradients with `scatter_add` (the previous one-hot matmul cost a `(batch, sequence, vocab)` intermediate — fine at char-level vocabs, fatal at Gemma's 262144). `positional="sinusoidal"` (default) adds **absolute sinusoidal** position encodings; `positional="none"` is what a RoPE model wants, since position then enters inside attention. `scale_embeddings=True` multiplies by `sqrt(embedding_dim)` as Gemma does. `backward()` returns `None`: nothing upstream of token ids can receive a gradient

**GPTEmbedBack:** Unembedding layer that shares weights with GPTEmbedFront via tied embedding table

**Weight tying:** GPT models should pass the same embedding table to both GPTEmbedFront and GPTEmbedBack for parameter sharing.

### Gemma Assembly (gemma.py)

`gemma_gpt(...)` assembles a Gemma 4 shaped decoder: scaled embeddings with no absolute positions, interleaved sliding/global blocks with RMSNorm sandwich norms, QK norm, GQA, p-RoPE on global layers, shared key/value projections on global layers, a tied unembedding, logit softcap, softmax.

`GEMMA_4_31B` and `GEMMA_4_12B` hold the two published text configurations. `gemma_parameter_count` returns 30.70B and 11.91B without allocating anything, against the ~30.72B implied by the 31B checkpoint's 62,546,177,752 bf16 bytes once the ~0.55B vision tower is set aside, and the 12B checkpoint's 11,959,730,224 tensors less its projector. The 12B is 48 layers of 3840 with 16 heads over 8 KV heads, and its global layers cut to a single shared key/value head.

**This assembles the architecture, not the weights.** `gemma_gpt(**GEMMA_4_31B)` allocates **473.93 GiB** in float32 and OOMs after ~11 of 60 layers on an 80 GB card. Under `inference_mode()` that drops to **115.11 GiB** (4.12x), which is parameters plus 0.75 GiB of RoPE tables. Getting to one GPU still needs fp16 or int8 on top, plus a safetensors loader and a KV cache.

The 12B is the one that fits as written: **189.43 GiB** to train, but **45.11 GiB** under `inference_mode()`, which leaves ~34 GiB on an 80 GB card. Watch the logits there, `(batch, sequence, 262144)` at 1 GiB per 1024 tokens, and again for the softmax output. Halving to bf16 would put it at 22.93 GiB (parameters halved, RoPE tables left in fp32). CuPy v14 made that reachable, bf16 via `ml_dtypes.bfloat16` (numpy 2.1.2+, not CUDA 12.1), but nothing in the library speaks it yet: `FLOAT_TYPE` is a single float32 global and mixed precision wants per-tensor control. Measured on 14.2.0 by `bf16_smoke.py`:

- **2D matmul works, batched matmul does not.** `a @ b` at exactly two dimensions is fine in all of `@`, `matmul`, `dot`, `tensordot` (~0.3% max relative error, which is just bf16 rounding). Anything with leading batch axes raises `TypeError: data type 'E' not understood`, and `einsum` fails at any rank. That splits this codebase cleanly: every parameterized matmul is `(B, T, C) @ (C, out)` and reshapes to `(B*T, C)` as a free view, while attention's `q @ k.T` and `attends @ v` (layers.py:553, 567) are genuinely batched over `(batch, kv_heads, group)` and have no 2D form
- Reductions are safe. CuPy widens the accumulator internally, unlike numpy: summing 3840 bf16 values is 0.09% off, 262144 of them 0.11%. The axis reduction RMSNorm actually uses is 0.30%, improving to 0.03% with an explicit `dtype = cupy.float32` - worth passing, but not the correctness emergency a numpy-only measurement suggests
- bf16 carries fp32's exponent range, so the `x*x` in RMSNorm has headroom fp16 does not. It pays 8 mantissa bits to fp16's 11
- `cupy.tanh` returns float32 from a bf16 input. Softcap sits on the `(batch, sequence, 262144)` logits, the largest tensor in the model, so that one needs an explicit `.astype`
- cuRAND will not generate bf16, so `init_random_tensor` and `Dropout` have to generate fp32 and cast. `cupy.add.at` has no bf16 either, so `scatter_add` and therefore training are out - this is an inference-only path

Both configs carry 0.75 GiB of RoPE tables, since those scale with `context_length` and head width rather than model size.

### Utilities (utils.py)

**Residual wrapper:** Combines layers with residual connections (mode="add") or concatenation (mode="concat")

**Image augmentations:** `augment_images()` chains random_flip, random_rotate, random_shift using NumPy/CV2

**inference_mode():** context manager for building a model with no training state. Layers constructed inside it allocate no gradient/moment/variance buffers, and `Network._forward` calls `clear_cache()` on each layer as soon as the next one has consumed its output. Construction must happen inside the context, since the buffers are allocated in `__init__`; prediction works inside or outside it, because layers remember how they were built. `train()` raises on such a model

**Cache:** per layer state for incremental decoding, held on `Layer.cache` and `None` whenever the model is not generating. Deliberately **not** in `CACHED`, since it has to survive `clear_cache()` the way BatchNorm's running statistics do. Carries the absolute `position` of the next token — which is all most layers need, so that RoPE and the sinusoidal encodings read from where the tokens actually sit — plus, for attention, the key/value store itself. Every layer gets one, so `self.cache is not None` is a uniform signal that generation is running; only attention calls `allocate()`.

The store is **contiguous and in absolute order**, which is what lets `_block_mask` work unchanged. A sliding layer holds `window + step` entries and compacts when it overflows: one copy of `window` entries per `step` tokens, against the `window` entries every token already reads, so it amortizes to well under a percent. A ring buffer copies nothing but rotates the key axis, which would make `_BLOCK_MASKS` depend on position rather than geometry — the reason it was not used.

`Layer.start_cache(batch_size, max_length, step)` / `stop_cache()` bracket generation; composite layers (`Residual`, `TransformerBlock`) recurse the way `clear_cache` and `set_eval` do. `MultiHeadAttention.backward` raises while a cache is active: the store holds keys from earlier forwards with no gradient path back.

**scatter_add:** in-place `target[indices] += values` summing repeated indices, via `cupyx.scatter_add` with a numpy fallback. Used for embedding table gradients

**Tensor initialization:**
- `init_random_tensor()`: Gaussian initialization scaled by fan-in
- `init_zeros_tensor()`: Zero initialization
- All tensors use `FLOAT_TYPE = cupy.float32` for TF32 tensor core compatibility

## Common Patterns

### Building a CNN
```python
from network import Network, CrossEntropy
from layers import Convolution, BatchNorm, MaxPool, Flatten, Dense
from activations import SiLU, SoftMax
from utils import Residual

model = Network([
    Convolution((3,32,32), 32, (3,3), padding=(1,1)),
    BatchNorm(32),
    SiLU(),
    MaxPool(2,2),
    Flatten(),
    Dense(32*16*16, 10),
    SoftMax()
])
```

### Building a Vision Transformer
```python
from transformer_adapters import VitProjector, VitMLPHead

model = Network([
    VitProjector((3,32,32), patch_size=(4,4), embedding_dim=384, num_registers=15),
    TransformerBlock(384, context_length=0, num_heads=6, SiLU, dropout_rate=0.2),
    # ... more transformer blocks
    LayerNorm(384),
    VitMLPHead(384, 128),
    SiLU(),
    Dense(128, 10),
    SoftMax()
])
```

**Note:** ViT uses `context_length=0` since image patches have no causal structure.

### Building a GPT
```python
from transformer_adapters import GPTEmbedFront, GPTEmbedBack

embedding_table = init_random_tensor((vocab_size, embed_size))

model = Network([
    GPTEmbedFront(embedding_table, context_length=1024),
    TransformerBlock(384, context_length=1024, num_heads=6, SiLU, decoder=True, dropout_rate=0.2),
    # ... more transformer blocks
    LayerNorm(384),
    GPTEmbedBack(embedding_table),  # Tied weights
    SoftMax()
])
```

**Critical:** Set `decoder=True` in TransformerBlock for autoregressive masking.

### Building a Gemma 4 style GPT
```python
from gemma import gemma_gpt, GEMMA_4_31B, gemma_parameter_count

model = gemma_gpt(
    vocab_size=vocab_size, context_length=1024,
    num_layers=12, embed_dim=384, ffn_multiplier=4,
    num_heads=6, head_dim=64, num_kv_heads=2,     # GQA, 3 query heads per kv head
    global_head_dim=128, num_global_kv_heads=1,   # global layers widen heads, share K=V
    sliding_window=256, pattern=6,                # 5 sliding : 1 global, last layer global
    global_partial_rotary=0.25,                   # p-RoPE on global layers only
)
```

`gemma_parameter_count(**GEMMA_4_31B)` gives the real model's 30.70B text params for comparison.

### Inference Only
```python
from utils import inference_mode

with inference_mode():                 # must wrap CONSTRUCTION, not just the forward
    model = gemma_gpt(**config)

model.predict(tokens)                  # works inside or outside the context
```

Skips three full extra copies of every parameter and frees each layer's activations as the forward advances. `model.train(...)` on such a model raises rather than silently no-opping.

### Generation
```python
with inference_mode():
    model = gemma_gpt(**config)

tokens = model.generate(prompt, max_new_tokens=256, temperature=0.8, top_k=50, step=256)
```

`prompt` is `(batch, length)` of ids, or a single `(length,)` sequence; every row must be the same length, since one cache position is shared across the batch. Returns only the generated ids, not the prompt.

The prompt is run in chunks of `step` and then tokens come out one at a time, each forward costing one token of work instead of re-reading the whole sequence. For a 1024 token prompt and 256 sampled tokens that is ~1280 token-forwards against the ~295000 an uncached loop does — **~230x**.

- `step` bounds a sliding layer's store at `window + step`, trading cache headroom against compaction frequency. It composes with each attention layer's own `chunk_size`, which independently bounds the score matrix: a `chunk_size` below `step` simply runs several blocks per forward
- `temperature` is applied by the final `SoftMax` — the knob that layer already has — and restored afterwards. Sampling is temperature and `top_k`, no nucleus
- **While a cache is active, `GPTEmbedBack` returns only the final position's logits.** This is the one place a cache changes what a forward means, and it is not optional: the logits are the largest tensor in the model, `(batch, sequence, 262144)` at 1 GiB per 1024 tokens, so unembedding a whole prefill chunk to sample one row of it is the most expensive mistake available here

Cache size, fp32, at `step=256` — the window bound is the whole game, since sliding layers become a **constant** and only the global layers scale:

| | T=8192 | T=32768 | T=131072 | T=262144 |
|---|---|---|---|---|
| 12B sliding (40 layers) | 0.78 | 0.78 | 0.78 | 0.78 GiB |
| 12B global (8 layers) | 0.25 | 1.00 | 4.00 | 8.00 GiB |
| 31B sliding (50 layers) | 1.95 | 1.95 | 1.95 | 1.95 GiB |
| 31B global (10 layers) | 1.25 | 5.00 | 20.00 | 40.00 GiB |

Unbounded, the sliding cache at full context would be 168 GiB (12B) and 440 GiB (31B). The 12B under `inference_mode()` is 45.11 GiB, so 32k context costs 1.78 GiB of cache on top and even 131k fits at 4.78 GiB.

### Training
```python
criterion = CrossEntropy(num_classes=10)

model.train(
    criterion,
    train_data, train_labels,
    test_data=test_data, test_labels=test_labels,
    augments=augment_images,  # Optional augmentation function
    epochs=20,
    batch_size=512,
    batches_per_step=1,  # Gradient accumulation: effective_batch = batch_size * batches_per_step
    learning_rate=0.001,
    weight_decay=0.01
)
```

### Allocation Discipline

Large intermediates are the thing to watch, especially anything that scales with vocabulary. Fixed instances, kept here as the pattern to avoid:

- an identity matrix indexed to build one-hots (`num_classes` squared)
- one-hot expansion to `(batch, sequence, vocab)` when a gather over `(batch, sequence)` does
- a full-sized array holding a single constant, where a scalar works — and which upcast the dtype as well
- `cupy.ones(...)` / `cupy.zeros(...)` without `dtype = FLOAT_TYPE`, which silently gives float64

The AdamW step in `_update` still allocates ~9 full-sized temporaries per parameter tensor. The variance update reuses the gradient buffer as scratch (bitwise identical, and the gradient is zeroed right after the step). Folding the bias correction into the learning rate would remove two more, but is not bit-identical, so it has been left alone.

### Gradient Accumulation Pattern

The optimizer step occurs every `batches_per_step` batches. Gradients are scaled by `num_samples = batch_size * batches_per_step` (see network.py:66).

### DenseNet-Style Residual Blocks
```python
def dense_layer(input_size, growth_rate, dropout_rate):
    return Residual([
        BatchNorm(input_size[0]),
        SiLU(),
        Dropout(dropout_rate),
        Convolution(input_size, growth_rate, (3,3), padding=(1,1)),
    ], mode="concat")
```

## Critical Implementation Details

### Gradient Tensor Dimension Handling

**Attention/FFN layers** use `cupy.tensordot(x.transpose(2,0,1), grad, 2) / T` to:
1. Average gradients over sequence length T
2. Accumulate gradients for weight matrices

**Convolution layers** use `einsum` with explicit dimension labels for clarity

### Data Flow Through Transformers

Input shape progression for ViT:
- Image: `(B, C, H, W)`
- Patches: `(B, num_patches, patch_dim)`
- Embeddings: `(B, num_patches + cls_tokens, embed_dim)`
- After attention/FFN: `(B, seq_len, embed_dim)` preserved
- Classification head extracts: `(B, embed_dim)` from position 0

Input shape progression for GPT:
- Tokens: `(B, T)` uint8 indices
- After GPTEmbedFront: `(B, T, C)` float32 embeddings
- After transformer blocks: `(B, T, C)` preserved
- After GPTEmbedBack: `(B, T, vocab_size)` logits
- After SoftMax: `(B, T, vocab_size)` probabilities

### CUDA Memory and Precision

**Float precision:** Everything runs in FP32, but TF32 tensor cores are enabled via environment variables and cuBLAS math mode. This gives ~8x speedup over FP32 on Ampere+ GPUs with minimal accuracy loss.

**Batch size tuning:** Start with batch_size=512 for CIFAR-10 on CNNs, batch_size=16 for GPT with context_length=1024

## Model Checkpointing

See demo.ipynb cells 14-16 for save/load functions:
```python
def save_weights(model, path):
    lst = []
    for layer in model.layers:
        dct = {"name": type(layer).__name__, "parameters": []}
        for parameter in layer.parameters:
            dct["parameters"].append(parameter.get())  # Transfer from GPU
        lst.append(dct)
    with open(path, "wb") as handle:
        pickle.dump(lst, handle)

def load_weights(model, path):
    model._zero_grad()
    with open(path, "rb") as handle:
        lst = pickle.load(handle)
    for layer, dct in zip(model.layers, lst):
        for parameter, arr in zip(layer.parameters, dct["parameters"]):
            parameter *= 0
            parameter += cupy.array(arr, dtype=cupy.float32)
```

## Testing

`python test_gemma.py` gradient-checks every layer against finite differences. It substitutes numpy for cupy and switches `FLOAT_TYPE` to float64, because central differences are far too noisy to validate a backward pass in float32.

Run it after touching any `backward()`. Every gradient here is hand derived, so a wrong backward pass is silent: the model still trains, just towards the wrong thing.

It also checks the KV cache the same way it checks chunking — by equivalence. `cache_equivalent` requires incremental decode to reproduce a single full forward, and `positionwise_cache_equivalent` walks a whole model, comparing the distribution each cached forward returns against the full forward's at that position. The latter is what pins down `GPTEmbedBack`'s last-position slice and `GPTEmbedFront`'s position offset; the gemma stack alone catches neither, since it has no absolute encodings and a prefill ending on a single token hides the slice. When adding cache tests, make the prefill chunks longer than one token and keep at least one model with `positional="sinusoidal"`.

`python bf16_smoke.py` is a separate probe, and needs a real GPU rather than the numpy substitution: it runs every cupy call the Gemma forward path makes in bf16 and reports which ones work. It checks two things beyond whether an op raises - whether the result stays bf16 (an op that silently upcasts to float32 doubles memory with no error to notice) and what a long reduction costs without an fp32 accumulator. Run it before touching `FLOAT_TYPE`.

Two traps the harness encodes, both of which produce misleading passes:
- The upstream gradient must not be drawn from the same seed as the input. If they coincide, a normalization layer's Jacobian analytically cancels to ~0 and the check becomes vacuous.
- Compare gradients relative to *tensor scale*, not per element. Normalization gradients sum to zero along the normalized axis, so individual entries legitimately sit near zero and a per-element ratio explodes on noise.

## TODOs in Codebase

Done: GQA + RoPE in MultiHeadAttention, and RMSNorm.

Done: chunked attention, so sliding windows now save memory and compute rather than only masking.

Done: KV cache and an incremental decode loop, via `Cache` and `Network.generate`.

Remaining, if pushing further towards a runnable Gemma 4:
- Recompute-in-backward. Chunking bounds *peak* memory everywhere, but *retained* memory only for sliding layers (O(T*window)); global layers still hold O(T^2) of softmax across the loop. Dropping `self.softmax` and recomputing each block in backward would bound those too, at roughly +25-30% attention FLOPs. This is the switch that pays at short context, where chunking alone does nothing
- Loading real checkpoints: safetensors reader and the vision tower. bf16 no longer needs converting, since CuPy v14 reads `ml_dtypes.bfloat16` directly and a checkpoint tensor can go to the device as it is