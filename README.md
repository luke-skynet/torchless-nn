# Torchless Neural Nets

Convolutional Neural Nets, Vision Transformers, and GPT's all implemented from scratch in Cuda Numpy (CuPy).
Without torch's tensor or autograd engine, this library hand derives all gradient calculations with reverse accumulation.
All model operations use the shared `xp` backend in `backend.py`. CuPy is the default.
To use NumPy on CPU, set `TORCHLESS_BACKEND=numpy` before importing any library modules
(or set `os.environ['TORCHLESS_BACKEND'] = 'numpy'` at the start of a notebook).
Use `from backend import xp` when creating input arrays. Select one backend per process.
NumPy execution does not require CuPy; for CPU-only runtime installation, omit the
`cupy-cuda12x` line from `requirements.txt`.

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
state must match its shape and dtype. Identical shared entries are updated once;
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
from utils import inference_mode

with inference_mode():
    model = gemma_gpt(**config)
```

Layers built inside the context allocate no gradient, moment or variance buffers, and each layer's cached activations are released as soon as the next layer has consumed them. The Gemma 4 12B configuration holds about 45.86 GiB of FP32 weights and full-context RoPE tables; KV caches, activations, and CUDA workspace are additional. The 31B configuration requires about 115.86 GiB before those extras.

## Modules

* **activations.py** - ReLU, exact GeLU, GeLUTanh approximation, SiLU (Swish), and Softmax activations.
* **layers.py** - Convolution, BatchNorm, MaxPool, AveragePool, Flatten, Dense, Dropout, and Transformer (LayerNorm, RMSNorm, Attention, Rotary Embeddings, Gated Feed Forward, Logit Softcap) layers.
* **gemma.py** - Gemma 4 style model assembly: grouped query attention, QK norm, sandwich norms, interleaved sliding window and global attention, and p-RoPE.
* **network.py** - Network framework class with Cross Entropy loss criterion and AdamW optimization.
* **optimizer.py** - Fused CUDA AdamW updates and the NumPy reference implementation.
* **transformer_adapters.py** - ViT image to tokens embedding, ViT MLP classification head, GPT embedding and GPT prediction layers.
* **test_gemma.py** - Finite difference gradient checks for every layer, runnable on CPU (```python test_gemma.py```).
* **utils** - Layer interface, Residual Layer wrapper, basic image augmentation functions, and tensor initializers to keep all parameters in FP32/TF32.

## Running Gemma 4 12B

Text-only FP32 inference is implemented for the dense Gemma 4 Unified architecture.
It loads the original BF16/F16/F32 safetensors checkpoint, including tied embeddings,
normalization scales, and per-layer scalars. Quantized checkpoints, multimodal inputs,
MoE, and cross-layer KV sharing are rejected or unsupported.

On a Linux NVIDIA VM with an appropriate CUDA 12 driver, create an environment and
install `requirements.txt` (the CuPy wheel is for CUDA 12; choose the corresponding
CuPy wheel if your VM uses another CUDA major version). PyTorch is not required for
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
`TORCHLESS_BACKEND=numpy`). CPU execution supports checkpoint loading and generation;
`--tf32` requires CuPy. Reports use `peak_xp_used_bytes` and `peak_xp_reserved_bytes`
for CUDA allocator peaks; both are `null` on CPU, where memory usage is not measured.

Generated text goes to stdout. Tensor accounting and synchronized load/prefill/decode
timings go to stderr, and optionally to `--report`. Reported memory peaks are CuPy
allocator high-water marks; CUDA context and external-library workspaces are excluded.
Decode throughput includes sampling the first token from the prefill result.

The loader validates all names/shapes before model allocation, skips random weight
initialization, and copies bounded row chunks (normally at most 16 MiB of FP32 output
per chunk). It memory-maps source tensors and performs BF16 conversion on the CPU.
It never creates a second complete model on the GPU. The original checkpoint's
vision/audio tensors are explicitly listed as skipped in the report.

At 8192 allocated context, the 12B weights and RoPE tables occupy about 44.41 GiB;
KV caches and inference intermediates are additional. The full 262144-context tables
raise the baseline to 45.86 GiB. Actual 12B GPU peak memory and speed still need to be
measured on the VM.

## Reference and regression tests

Use a separate CPU environment with `requirements-test.txt`:

```bash
python -m pip install -r requirements-test.txt
python test_gemma.py
python -m pytest tests -q
```

The tests select the NumPy backend on CPU. They compare a deterministic tiny Gemma
against Transformers 5.16.1 in FP32, including embeddings, attention, FFNs, norms,
decoder outputs, logits, and cached decoding across sliding-window boundaries.
They also round-trip sharded BF16, FP16 and FP32 weights, test malformed checkpoints,
and compare text-to-token-to-generation results with reference greedy decoding.
The official 12B tensor manifest is checked without allocating its weights.

For actual CUDA parity, install both requirement sets and run:

```bash
CUPY_TF32=0 TORCHLESS_TEST_DEVICE=cuda python -m pytest tests -q
```

The CUDA suite requires a working NVIDIA GPU and a CUDA-enabled PyTorch installation.
`tests/test_optimizer.py` checks multi-step Adam parity, shared storage, metadata
replacement, frozen ViT positions, and (on CUDA) a single launch per update.
CPU parity has been verified; CUDA parity and a full 12B forward pass have not yet
been run. The source manifest in `tests/fixtures` contains only tensor names, shapes,
dtypes, configuration, and revision metadata, not checkpoint weights.
