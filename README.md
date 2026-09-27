# Torchless Neural Nets

Convolutional Neural Nets, Vision Transformers, and GPT's all implemented from scratch in Cuda Numpy (CuPy).
Without torch's tensor or autograd engine, this library hand derives all gradient calculations with reverse accumulation.
All model operations use the shared `xp` backend in `backend.py`. CuPy is the default.
To use NumPy on CPU, set `XP_RUNTIME=CPU` before importing any library modules
(or set `os.environ['XP_RUNTIME'] = 'CPU'` at the start of a notebook).
Use `from backend import xp` when creating input arrays. Select one backend per process.
NumPy execution does not require CuPy. Install CuPy separately from the shared
`requirements.txt`; this avoids replacing an installed patched build.

## BF16 precision

Set `XP_PRECISION=bfloat16` before importing the library to use BF16 weights,
activations, residuals, and KV caches. The default remains `float32`. BF16 requires
an NVIDIA GPU with compute capability 8.0 or later and the BF16-enabled CuPy build
from [luke-skynet/cupy, batched-16-fix](https://github.com/luke-skynet/cupy/tree/batched-16-fix).
Install that build separately; `requirements.txt` does not install or replace CuPy.
For FP32-only CUDA execution, install the appropriate stock CuPy wheel separately.

Training and inference use the same forward precision. Norm parameters and running
statistics stay FP32. Kernels for normalization, RoPE, activation functions,
softmax, dropout, and embedding-gradient scatter convert values in registers,
without full-sized FP32 casting buffers. Norm statistics and training attention
probabilities are intentionally retained in FP32. Final loss/sampling probabilities
are FP32. Contractions use BF16 operands with FP32 internal accumulation and BF16
outputs; accumulated parameter gradients are FP32, but the contraction output has
already rounded to BF16. Loss scaling is not used.

Training keeps FP32 master weights, gradients, and Adam moments. Adam updates the
masters and refreshes BF16 compute weights in the same CUDA launch. Inference
allocates none of that state. BF16 training therefore uses 18 bytes per ordinary
parameter before activations (2 compute + 4 master + 4 gradient + 8 moments), versus
16 for FP32 training; its savings are in activations and computation. BF16 inference
uses two bytes per ordinary weight, with FP32 norm parameters and RoPE tables.

```python
# Set XP_RUNTIME=CUDA XP_PRECISION=bfloat16 before starting Python.
from backend import inference_mode
from checkpoint import load_checkpoint

model, report = load_checkpoint('/models/gemma-4-12B-it')
```

```bash
python generate.py --checkpoint /models/gemma-4-12B-it --dtype bfloat16 \
  --prompt "Explain grouped-query attention." --max-new-tokens 128
```

BF16 checkpoints copy their original BF16 bits directly to BF16 destinations,
with bounded contiguous staging for transposed weights. Norm tensors expand to
FP32. F16/F32 source checkpoints remain supported, and training loads preserve
source precision in the FP32 master. `--inspect --dtype bfloat16` estimates mixed
storage without CUDA; actual loading reports destination array sizes.

Parameters are represented by `Parameter` objects: `.data` is compute storage;
`.master`, `.grad`, `.moment`, and `.variance` are optional training state.
`Layer.register(array, dtype=None)` returns `(compute_array, gradient_array)`.
Custom layers should retain both returned arrays and combine child `.parameters`
lists when composing layers. Tied embeddings share one parameter object.

Validation:

```bash
XP_RUNTIME=CPU python -m pytest -q
XP_RUNTIME=CUDA XP_PRECISION=bfloat16 python -m pytest -q
```

The CUDA suite executes the fused kernels, BF16 GEMMs/einsum, Adam, checkpoint
loading, cached decoding, and a short Gemma training run. GPU correctness and
throughput must be measured on the target CUDA system; CPU checks do not establish
GPU correctness or speedups.

## Attention

Attention supports both plain multi-head and grouped query (GQA), rotary and partial-rotary position embeddings, per-head QK normalization, sliding window attention, and Gemma's shared key/value projection. Calling `MultiHeadAttention` with default arguments keeps the original fused QKV path and parameter layout, so existing checkpoints still load.

Queries are processed in blocks of `chunk_size` (default: one block, identical to computing the whole sequence at once). Setting it bounds peak memory by the block size instead of the sequence length, and makes a sliding window an actual saving rather than just a mask — the scores outside the window are never computed instead of being computed and discarded.

## Feedforward

`TransformerBlock` accepts `attn_bias=True` and `ffn_bias=True` independently.
The underlying `MultiHeadAttention` and `TransformerFeedForward` layers each use
`bias=True`. Biases default to disabled, preserving Gemma's parameter layout.

`attn_dropout_rate` drops attention probabilities, with a separate mask retained
for every query chunk during training. `res_dropout_rate` applies after the
attention output projection and, on `TransformerBlock`, after the FFN projection.
An explicit `output_dropout_rate` overrides the FFN rate. All rates default to zero;
`set_eval(True)` disables dropout throughout the block.

```python
block = TransformerBlock(768, 197, 12, GeLU,
                         attn_bias=True, ffn_bias=True,
                         attn_dropout_rate=0.1, res_dropout_rate=0.1)
```

`TransformerFeedForward` supports a plain activation MLP (`glu=False`, the default)
or GLU gating (`glu=True`). `TransformerBlock` uses the same default and flag. Both modes
use `ffn_multiplier` on the block (`multiplier` on the layer) to set hidden width.
`hidden_dropout_rate` applies after activation/gating; `output_dropout_rate` applies after
the down projection, before any post-FFN norm and residual addition. Hidden and
output dropout can be configured independently.

```python
block = TransformerBlock(768, 197, 12, GeLU, glu=False,
                         hidden_dropout_rate=0.1, output_dropout_rate=0.1)
```

`GatedFeedForward` remains an alias for existing imports and also defaults to
`glu=False`. Pass `glu=True` to preserve the three-matrix checkpoint layout; ungated
layers omit the gate matrix entirely. Gemma explicitly enables gating.

Pass `activation=GeLU` for exact, erf-based GELU or `activation=GeLUTanh` for the
tanh approximation. Both classes provide their own backward pass. Gemma defaults
to `GeLUTanh` to match its checkpoints.
Exact `GeLU` uses vectorized `math.erf` on the CPU and `cupyx.scipy.special.erf`
on CUDA, keeping GPU inputs on the GPU. `GeLUTanh` also runs entirely on the selected backend.

Training layers accumulate gradients of the summed loss without averaging over
patches, tokens, or attention heads. Each optimizer step divides by the actual
number of labels accumulated: images for classification, tokens for language
modeling. The final partial accumulation group is applied at the end of each epoch.

On CUDA, `optimizer.py` updates all registered parameters and clears their gradients
in one kernel launch per optimizer step. It uses the existing arrays, supports
C-contiguous float32 and float64 tensors (including both in the same model), and
caches pointer metadata until storage or decay rules change. Each parameter's Adam
state must match its master weight's shape and dtype. Identical shared entries are updated once;
conflicting optimizer state and overlapping storage are rejected. NumPy uses the
reference update equations and also clears gradients during the update.

Gemma uses `GPTEmbeddingTable` to share the token weight, accumulated gradient,
and Adam state between lookup and output projection. The input layer registers
the state once; both layers retain `.table` access for checkpoint loading.
`gemma_gpt(embedding_table=...)` accepts an existing array or a shared table.
Construct a shared table in the same inference mode as its embedding layers.
For other tied GPT models, pass one `GPTEmbeddingTable(vocab_size, embed_size)`
to both `GPTEmbedFront` and `GPTEmbedBack`; both layers require this shared-table
object and reject raw arrays. To reuse an existing array, wrap it with
`GPTEmbeddingTable(vocab_size, embed_size, table=array)` first.

`GPTEmbedFront(..., positional="learned")` adds a trainable, initially zero
absolute position table, available as `.positional_encoding`. Its gradients sum
over the batch, and cached decoding reads positions at the current cache offset.
Inputs beyond the position table's context length raise `ValueError`.
The existing `"sinusoidal"` default and Gemma's `"none"` mode are preserved.

```python
table = GPTEmbeddingTable(vocab_size, embed_size)
front = GPTEmbedFront(table, context_length, positional="learned")
back = GPTEmbedBack(table)
```

## Numerical stability

Normalization epsilons are configurable: `BatchNorm` and `LayerNorm` default to
`eps=1e-5`, while `RMSNorm` and attention's Q/K/V norms default to `eps=1e-6`.
`TransformerBlock(..., eps=...)` passes that value to every nested norm; omitting
it preserves each norm's default. `gemma_gpt(rms_norm_eps=...)` also configures the
final norm. Checkpoint loading honors its stored value, with
`default_rms_norm_eps=1e-6` available as a fallback when the field is absent.

Loss and optimizer stabilization are separate: `CrossEntropy(..., eps=1e-7)`
controls the log offset, and `Network.train(..., adam_eps=1e-7)` controls Adam's
denominator on every update, including the final partial accumulation group.

## Inference Mode

```python
from backend import inference_mode

with inference_mode():
    model = gemma_gpt(**config)
```

Layers built inside the context allocate no gradient, moment or variance buffers, and each layer's cached activations are released as soon as the next layer has consumed them. With the default FP32 precision, the Gemma 4 12B configuration holds about 45.86 GiB of weights and full-context RoPE tables; KV caches, activations, and CUDA workspace are additional. The 31B configuration requires about 115.86 GiB before those extras.

## Modules

* **activations.py** - ReLU, exact GeLU, GeLUTanh approximation, SiLU (Swish), and Softmax activations.
* **layers.py** - Convolution, BatchNorm, MaxPool, AveragePool, Flatten, Dense, Dropout, and Transformer (LayerNorm, RMSNorm, Attention, Rotary Embeddings, Gated Feed Forward, Logit Softcap) layers.
* **gemma.py** - Gemma 4 style model assembly: grouped query attention, QK norm, sandwich norms, interleaved sliding window and global attention, and p-RoPE.
* **network.py** - Network framework class with Cross Entropy loss criterion and AdamW optimization.
* **precision_ops.py** - Fused CUDA kernels at BF16/FP32 computation boundaries.
* **optimizer.py** - Fused CUDA AdamW updates and the NumPy reference implementation.
* **transformer_adapters.py** - ViT image to tokens embedding, ViT MLP classification head, GPT embedding and GPT prediction layers.
* **backend.py** - Array backend selection (CuPy or NumPy), model precision selection, tensor initializers, and the `inference_mode` and `empty_weights` construction contexts.
* **utils.py** - Layer interface, Residual Layer wrapper, the incremental decoding Cache, and basic image augmentation functions.

## Running Gemma 4 12B

Text-only FP32 and BF16 inference is implemented for the dense Gemma 4 Unified architecture.
It loads the original BF16/F16/F32 safetensors checkpoint, including tied embeddings,
normalization scales, and per-layer scalars. Quantized checkpoints, multimodal inputs,
MoE, and cross-layer KV sharing are rejected or unsupported.

On a Linux NVIDIA VM with an appropriate CUDA 12 driver, create an environment and
install `requirements.txt` and a CuPy build matching your CUDA version. Use the
patched CuPy build for BF16, or a stock wheel for FP32. PyTorch is not required for
loading or generation. OpenCV is only needed for the image augmentation examples.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
hf download google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 --local-dir /models/gemma-4-12B-it
python generate.py --checkpoint /models/gemma-4-12B-it --inspect
python generate.py --checkpoint /models/gemma-4-12B-it \
  --prompt "Explain grouped-query attention." \
  --max-new-tokens 128 --context-length 8192 --report run.json
```

The pinned revision is the checkpoint manifest/tokenizer verified during development.
The download is about 24 GB of weights plus tokenizer/configuration files. The CLI
reads local files only. `--inspect` validates the architecture and all tensor headers
without CUDA or weight allocation; it does not scan weight values for finiteness.
Actual loading also checks values and verifies an optional duplicate tied LM head.

The default uses the checkpoint chat template and greedy decoding. Use
`--raw-prompt` for base-model completion, or `--temperature 0.8 --top-k 50 --seed 0`
for sampling. Stop IDs come from the generation configuration and tokenizer. The
context must cover the prompt plus requested generation; no silent truncation occurs.
`--prefill-step` and `--chunk-size` default to 256. `--tf32` explicitly enables TF32;
leave it off when comparing against FP32 reference results.

For CPU generation, add `--backend numpy` to the generation command (or set
`XP_RUNTIME=CPU`). CPU execution supports checkpoint loading and generation;
`--tf32` requires CuPy. Reports use `peak_xp_used_bytes` and `peak_xp_reserved_bytes`
for CUDA allocator peaks; both are `null` on CPU, where memory usage is not measured.

Generated text goes to stdout. Tensor accounting and synchronized load/prefill/decode
timings go to stderr, and optionally to `--report`. Reported memory peaks are CuPy
allocator high-water marks; CUDA context and external-library workspaces are excluded.
Decode throughput includes sampling the first token from the prefill result.

The loader validates all names/shapes before model allocation, skips random weight
initialization, and copies bounded row chunks (normally at most 16 MiB of FP32 output
per chunk). It memory-maps source tensors. FP32 destinations expand BF16 on the CPU; BF16
destinations receive the original bits.
It never creates a second complete model on the GPU. The original checkpoint's
vision/audio tensors are explicitly listed as skipped in the report.

At 8192 allocated context, the 12B weights and RoPE tables occupy about 44.41 GiB;
KV caches and inference intermediates are additional. The full 262144-context tables
raise the baseline to 45.86 GiB. Actual 12B GPU peak memory and speed still need to be
measured on the VM.
