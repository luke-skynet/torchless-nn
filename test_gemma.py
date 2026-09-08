"""Gradient checks for the Gemma 4 style layers.

Runs on CPU: it substitutes numpy for cupy and switches FLOAT_TYPE to float64, since
central differences are far too noisy to validate a backward pass in float32.

    python test_gemma.py

Every gradient in this library is hand derived, so a wrong backward pass is silent -
the model still trains, just towards the wrong thing. These checks compare each
analytic gradient against a finite difference of the forward pass.
"""

import sys, types
import numpy as np

sys.modules.setdefault('cupy', np)
if 'cv2' not in sys.modules: # only used by utils.augment_images, never called here
    sys.modules['cv2'] = types.ModuleType('cv2')

import utils
utils.FLOAT_TYPE = np.float64
import layers, transformer_adapters, network
for module in (layers, transformer_adapters, network):
    module.FLOAT_TYPE = np.float64

from layers import (MultiHeadAttention, GatedFeedForward, LayerNorm, RMSNorm, Softcap, Dense,
                    RotaryEmbedding, TransformerBlock, gemma_layer_types)
from activations import SiLU, GeLU
from gemma import gemma_gpt, gemma_parameter_count, GEMMA_4_31B, GEMMA_4_12B
from network import CrossEntropy

PASSED, FAILED = [], []


def record(name, condition, detail = ""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'pass' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def relerr(a, b):
    # max difference relative to tensor scale, not per element: normalization
    # gradients sum to zero along the normalized axis, so per element ratios
    # explode on entries that sit near zero for legitimate reasons
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b)) / scale)


def numgrad(f, x, eps = 1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags = ['multi_index'])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; plus  = f()
        x[i] = old - eps; minus = f()
        x[i] = old
        grad[i] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def check_layer(name, layer, x, seed = 1234, tol = 1e-7):
    """Compare analytic against numeric gradients for the input and every parameter.

    The library scales weight gradients by a constant (it averages over the sequence
    axis, and over head axes inside QK norm), so rather than hard coding that we fit
    the single scalar c in analytic = numeric / c and require the fit to be exact.
    A wrong backward pass will not fit one constant across every entry.
    """
    # this seed must differ from whatever produced x: if the upstream gradient equals
    # the input, a normalization layer's Jacobian cancels to ~0 and the test is vacuous
    up = np.random.default_rng(seed).standard_normal(np.shape(layer.forward(x)))

    def loss():
        return float(np.sum(layer.forward(x) * up))

    layer.zero_grad()
    layer.forward(x)
    dx = layer.backward(up.copy())

    worst = 0.0
    if dx is not None:
        worst = relerr(dx, numgrad(loss, x))
    for p, g in zip(layer.parameters, layer.gradients):
        ng = numgrad(loss, p)
        denom = float(np.sum(g * g))
        c = float(np.sum(ng * g)) / denom if denom > 0 else 1.0
        worst = max(worst, relerr(g * c, ng))

    record(name, worst < tol, f"worst relerr {worst:.1e}")


rng = np.random.default_rng(7)
B, T, C, H = 2, 6, 12, 4
x = rng.standard_normal((B, T, C))

print("norms, softcap:")
check_layer("RMSNorm over (batch, sequence, channels)", RMSNorm(C), x.copy())
check_layer("RMSNorm over (batch, kv, group, sequence, dim)", RMSNorm(4),
            rng.standard_normal((2, 3, 2, 6, 4)))
check_layer("LayerNorm unchanged", LayerNorm(C), x.copy())
check_layer("Softcap", Softcap(3.0), x.copy() * 4)

print("\nattention, default (fused) path unchanged:")
check_layer("MHA encoder", MultiHeadAttention(C, T, H), x.copy())
check_layer("MHA decoder", MultiHeadAttention(C, T, H, decoder = True), x.copy())
record("fused parameter layout preserved",
       [p.shape for p in MultiHeadAttention(C, T, H).parameters] == [(C, 3*C), (C, C)])

print("\nattention, Gemma options:")
check_layer("GQA 4 query : 2 kv heads", MultiHeadAttention(C, T, H, decoder = True, num_kv_heads = 2), x.copy())
check_layer("GQA 4 query : 1 kv head",  MultiHeadAttention(C, T, H, decoder = True, num_kv_heads = 1), x.copy())
check_layer("head_dim decoupled from embed_dim // heads",
            MultiHeadAttention(C, T, H, decoder = True, num_kv_heads = 2, head_dim = 5), x.copy())
check_layer("QK norm", MultiHeadAttention(C, T, H, decoder = True, num_kv_heads = 2, qk_norm = True), x.copy())
check_layer("shared key/value projection",
            MultiHeadAttention(C, T, H, decoder = True, num_kv_heads = 2, head_dim = 6, kv_shared = True), x.copy())
check_layer("sliding window", MultiHeadAttention(C, T, H, decoder = True, sliding_window = 3), x.copy())
check_layer("RoPE", MultiHeadAttention(C, T, H, decoder = True, num_kv_heads = 2, head_dim = 8,
                                       rope = RotaryEmbedding(8, T)), x.copy())
check_layer("p-RoPE", MultiHeadAttention(C, T, H, decoder = True, num_kv_heads = 2, head_dim = 8,
                                         rope = RotaryEmbedding(8, T, theta = 1e6,
                                                                partial_rotary_factor = 0.25)), x.copy())

print("\ntransformer blocks:")
check_layer("TransformerBlock, original defaults", TransformerBlock(C, T, H, SiLU, decoder = True), x.copy())
check_layer("Gemma sliding block", TransformerBlock(
    C, T, H, GeLU, decoder = True, norm = RMSNorm, post_norm = True, num_kv_heads = 2,
    head_dim = 8, qk_norm = True, rope = RotaryEmbedding(8, T), sliding_window = 4), x.copy())
check_layer("Gemma global block (K=V, p-RoPE)", TransformerBlock(
    C, T, H, GeLU, decoder = True, norm = RMSNorm, post_norm = True, num_kv_heads = 1,
    head_dim = 8, qk_norm = True, kv_shared = True,
    rope = RotaryEmbedding(8, T, theta = 1e6, partial_rotary_factor = 0.25)), x.copy())

# Gemma 4 checkpoint semantics, including both branches of shared K/V.
check_layer("unweighted value RMSNorm", RMSNorm(8, with_scale = False),
            np.random.default_rng(42).normal(size = (2, 3, 8)))
check_layer("Gemma 4 normalized values and unit attention scale", MultiHeadAttention(
    C, T, H, decoder = True, num_kv_heads = 1, head_dim = 8,
    qk_norm = True, v_norm = True, kv_shared = True, attention_scale = 1.0,
    rope = RotaryEmbedding(8, T, partial_rotary_factor = 0.25, proportional = True)), x.copy())
scaled_block = TransformerBlock(C, T, H, GeLU, decoder = True)
scaled_block.layer_scalar[...] = 0.7
check_layer("checkpoint layer scalar backward", scaled_block, x.copy())

# Independent proportional-RoPE formula: pair i with i + head_dim/2.
rope = RotaryEmbedding(8, T, theta = 10000.0, partial_rotary_factor = 0.5, proportional = True)
rope_input = np.random.default_rng(43).normal(size = (B, H, T, 8))
expected = rope_input.copy()
for i in range(2):
    angle = np.arange(T) / 10000.0**(2 * i / 8)
    expected[..., i] = rope_input[..., i] * np.cos(angle) - rope_input[..., i + 4] * np.sin(angle)
    expected[..., i + 4] = rope_input[..., i + 4] * np.cos(angle) + rope_input[..., i] * np.sin(angle)
record("proportional RoPE matches full-head frequency and pairing formula",
       relerr(rope.rotate(rope_input), expected) < 1e-14)
record("proportional RoPE backward inverts rotation",
       relerr(rope.backward(rope.rotate(rope_input)), rope_input) < 1e-14)

print("\nbehaviour:")
def dense_attention(attn, batch, length):
    """Reassemble the full attention matrix from the per block weights."""
    kv, groups = attn.num_kv_heads, attn.groups
    full = np.zeros((batch, kv, groups, length, length))
    for block, (q0, q1, k0, k1) in zip(attn.softmax, attn.chunks):
        full[..., q0:q1, k0:k1] = block
    return full

window = 3
for chunk in (None, 2, 5):
    attn = MultiHeadAttention(C, T, H, decoder = True, sliding_window = window, chunk_size = chunk)
    attn.forward(x)
    full = dense_attention(attn, B, T)
    i, j = np.indices((T, T))
    allowed = (j <= i) & (i - j < window)
    record(f"sliding window masks outside the band (chunk_size={chunk})",
           float(np.abs(full[..., ~allowed]).max()) == 0.0 and
           float(np.abs(full.sum(-1) - 1).max()) < 1e-12,
           f"{len(attn.softmax)} block(s)")
    record(f"each query attends to at most `window` keys (chunk_size={chunk})",
           int((full[0, 0, 0] > 1e-12).sum(-1).max()) == window)

# chunking must not change the answer, only the peak memory and the work done
def chunk_equivalent(name, build, sizes):
    reference = build(None)
    out = reference.forward(x.copy()); reference.zero_grad()
    grad = reference.backward(np.ones_like(out))
    worst = 0.0
    for size in sizes:
        other = build(size)
        for mine, theirs in zip(other.parameters, reference.parameters):
            mine[...] = theirs
        alt = other.forward(x.copy()); other.zero_grad()
        alt_grad = other.backward(np.ones_like(alt))
        worst = max(worst, relerr(alt, out), relerr(alt_grad, grad),
                    max([relerr(a, b) for a, b in zip(other.gradients, reference.gradients)] or [0.0]))
    record(name, worst < 1e-13, f"worst relerr {worst:.1e}")

chunk_equivalent("chunked == unchunked, causal",
                 lambda c: MultiHeadAttention(C, T, H, decoder = True, chunk_size = c), [1, 4, 5, T, 999])
chunk_equivalent("chunked == unchunked, sliding window",
                 lambda c: MultiHeadAttention(C, T, H, decoder = True, sliding_window = 4, chunk_size = c), [1, 3, 5])
chunk_equivalent("chunked == unchunked, bidirectional",
                 lambda c: MultiHeadAttention(C, T, H, decoder = False, chunk_size = c), [2, 4])
chunk_equivalent("chunked == unchunked, GQA + kv_shared + qk_norm + rope",
                 lambda c: MultiHeadAttention(C, T, H, decoder = True, num_kv_heads = 2, head_dim = 8,
                                              kv_shared = True, qk_norm = True, sliding_window = 4,
                                              rope = RotaryEmbedding(8, T), chunk_size = c), [2, 3, 5])

# chunking a sliding window layer skips the scores the mask would only discard
def computed(chunk):
    attn = MultiHeadAttention(C, T, H, decoder = True, sliding_window = 3, chunk_size = chunk)
    attn.forward(x)
    return sum(b.shape[-1] * b.shape[-2] for b in attn.softmax)
record("chunking a sliding window layer computes fewer scores", computed(3) < computed(None),
       f"{computed(3)} vs {computed(None)} elements per head")

from layers import _BLOCK_MASKS
for window in (3, None): # windowed AND global: global blocks span [0, q1), so each
                         # block has a different width and naively caches separately
    _BLOCK_MASKS.clear()
    for length in (16, 32, 64, 128):
        long_x = np.random.default_rng(0).standard_normal((1, length, C))
        MultiHeadAttention(C, length, H, decoder = True,
                           sliding_window = window, chunk_size = 4).forward(long_x)
    # a fully visible block caches None rather than an array of zeros, and costs nothing
    cached = sum(m.size for m in _BLOCK_MASKS.values() if m is not None)
    record(f"block mask cache stays bounded, sliding_window={window}",
           len(_BLOCK_MASKS) <= 6 and cached < 500,
           f"{len(_BLOCK_MASKS)} entries, {cached} elements across sequences up to 128")
_BLOCK_MASKS.clear()

rope = RotaryEmbedding(8, T, theta = 1e6, partial_rotary_factor = 0.5)
same_vector = np.broadcast_to(rng.standard_normal((1, 1, 1, 1, 8)), (1, 1, 1, T, 8)).copy()
rotated = rope.rotate(same_vector)
record("p-RoPE rotates only the leading dimensions",
       np.abs(rotated[..., :rope.rotary_dim] - rotated[..., :1, :rope.rotary_dim]).max() > 0.1)
record("p-RoPE leaves trailing dimensions position free",
       float(np.abs(rotated[..., rope.rotary_dim:] - rotated[..., :1, rope.rotary_dim:]).max()) == 0.0)

try:
    RotaryEmbedding(6, T, partial_rotary_factor = 0.25)
    record("rotary width rounding to zero is rejected", False)
except AssertionError:
    record("rotary width rounding to zero is rejected", True)

fused = MultiHeadAttention(C, T, H, decoder = True)
split = MultiHeadAttention(C, T, H, decoder = True, num_kv_heads = H, head_dim = C // H)
q, k, v = np.split(fused.qkv_weights, 3, axis = 1)
split.q_weights[...], split.k_weights[...], split.v_weights[...] = q, k, v
split.out_weights[...] = fused.out_weights
record("grouped path reduces to plain MHA when groups = 1",
       relerr(split.forward(x), fused.forward(x)) < 1e-14)

shared = MultiHeadAttention(C, T, H, decoder = True, num_kv_heads = 2, head_dim = 6, kv_shared = True)
shared.forward(x)
raw = (x @ shared.k_weights).reshape(B, T, 2, 1, 6).transpose(0, 2, 3, 1, 4)
record("shared projection uses the pre norm/rope key as the value",
       relerr(shared.value, raw) < 1e-15 and not hasattr(shared, "v_weights"))

types_ = gemma_layer_types(60)
record("60 layer interleave matches Gemma 4 31B's layer_types",
       [i for i, t in enumerate(types_) if t == "global"] == list(range(5, 60, 6)))
record("published 31B config reproduces the checkpoint parameter count",
       abs(gemma_parameter_count(**GEMMA_4_31B) - 30.72e9) < 0.1e9,
       f"{gemma_parameter_count(**GEMMA_4_31B)/1e9:.2f}B vs ~30.72B implied by the checkpoint")

types_ = gemma_layer_types(48, GEMMA_4_12B["pattern"])
record("48 layer interleave matches Gemma 4 12B's layer_types",
       [i for i, t in enumerate(types_) if t == "global"] == list(range(5, 48, 6)))
record("published 12B config reproduces the checkpoint parameter count",
       abs(gemma_parameter_count(**GEMMA_4_12B) - 11.96e9) < 0.1e9,
       f"{gemma_parameter_count(**GEMMA_4_12B)/1e9:.2f}B vs 11.96B of checkpoint tensors, "
       f"the remainder being the multimodal projector")

print("\nallocation shape:")
from layers import AveragePool
from network import CrossEntropy as CE

# CrossEntropy used to hold a num_classes x num_classes identity: 256 GiB at Gemma's
# vocabulary. Constructing one at that size is the test.
big = CE(262144)
record("CrossEntropy(262144) constructs without a vocab-squared array",
       not any(isinstance(v, np.ndarray) and v.size > 262144 for v in vars(big).values()))

def one_hot_reference(logits, labels, classes, grad):
    hot = np.eye(classes)[labels]
    return logits - hot if grad else -1 * np.sum(hot * np.log(logits + 1e-7))

worst = 0.0
for shape, classes in [((6,), 10), ((4, 7), 13), ((3, 5), 257)]:
    probs = np.random.default_rng(4).random(shape + (classes,)) + 0.01
    probs = probs / probs.sum(-1, keepdims = True)
    labels = np.random.default_rng(5).integers(0, classes, shape)
    criterion = CE(classes)
    worst = max(worst,
                relerr(criterion.gradients(probs, labels), one_hot_reference(probs, labels, classes, True)),
                abs(float(criterion.loss(probs, labels)) - one_hot_reference(probs, labels, classes, False)))
record("CrossEntropy matches the one-hot implementation it replaced", worst < 1e-13,
       f"worst {worst:.1e} across (batch,) and (batch, sequence) labels")

probs = np.random.default_rng(6).random((2, 3, 5))
untouched = probs.copy()
CE(5).gradients(probs, np.random.default_rng(7).integers(0, 5, (2, 3)))
record("CrossEntropy.gradients does not mutate the logits it was given",
       relerr(probs, untouched) == 0.0)

pool = AveragePool(2, 2)
pool.forward(np.zeros((2, 3, 8, 8), dtype = np.float32))
out = pool.backward(np.ones((2, 3, 4, 4), dtype = np.float32))
record("AveragePool preserves gradient dtype", out.dtype == np.float32,
       f"got {out.dtype}; a constant float64 mask used to promote it")
record("AveragePool holds no full sized constant mask", not hasattr(pool, "mask"))
check_layer("AveragePool gradient", AveragePool(2, 2),
            np.random.default_rng(8).standard_normal((2, 2, 4, 4)))

print("\nsoftmax gradient:")
from activations import SoftMax
from utils import inference_mode, INFERENCE_MODE
from network import Network

logits = np.random.default_rng(3).standard_normal((2, 4, 6))
check_layer("SoftMax(fused_loss=False) true Jacobian", SoftMax(fused_loss = False), logits.copy())
check_layer("SoftMax(fused_loss=False, temperature=2)", SoftMax(2.0, fused_loss = False), logits.copy())

# the fused path deliberately skips the Jacobian, so it must NOT match finite
# differences: that is exactly why it is only valid as the final layer
fused = SoftMax()
fused.forward(logits.copy())
passthrough = fused.backward(np.ones_like(logits))
record("fused SoftMax passes its gradient straight through",
       float(np.abs(passthrough - 1.0).max()) == 0.0)
real = SoftMax(fused_loss = False)
real.forward(logits.copy())
record("the two SoftMax modes genuinely differ",
       relerr(real.backward(np.ones_like(logits)), passthrough) > 0.1)

try:
    Network([SoftMax(), Dense(6, 3)])
    record("Network rejects a fused SoftMax that is not last", False)
except ValueError:
    record("Network rejects a fused SoftMax that is not last", True)
record("Network accepts a non-fused SoftMax mid network",
       Network([SoftMax(fused_loss = False), Dense(6, 3)]) is not None)

print("\ninference mode:")
trained = gemma_gpt(**dict(vocab_size = 16, context_length = 8, num_layers = 2, embed_dim = 16,
                           num_heads = 2, head_dim = 8, num_kv_heads = 1, sliding_window = 4,
                           pattern = 2, global_partial_rotary = 0.25))
with inference_mode():
    lean = gemma_gpt(**dict(vocab_size = 16, context_length = 8, num_layers = 2, embed_dim = 16,
                            num_heads = 2, head_dim = 8, num_kv_heads = 1, sliding_window = 4,
                            pattern = 2, global_partial_rotary = 0.25))
record("inference mode is not left enabled afterwards", INFERENCE_MODE is False)

def buffers(model, which):
    return sum(t.nbytes for layer in model.layers for t in getattr(layer, which))

record("inference model allocates the same parameters",
       buffers(lean, "parameters") == buffers(trained, "parameters"))
record("inference model allocates no gradients, moments or variances",
       all(buffers(lean, w) == 0 for w in ("gradients", "moments", "variances")),
       f"training model holds {buffers(trained,'gradients')*3:,} bytes of them")
record("parameter lists still line up with the empty buffer lists",
       all(len(l.gradients) == len(l.moments) == len(l.variances) == 0 for l in lean.layers))

# Inspect actual owned arrays, not only the registration lists: the old FFN
# allocated optimizer arrays that those lists could not see.
for model_name, model in (("inference", lean), ("training", trained)):
    for index, block in enumerate(model.layers):
        if not isinstance(block, TransformerBlock):
            continue
        ffn = block.ffn
        registered = {id(a) for name in ("parameters", "gradients", "moments", "variances")
                      for a in getattr(ffn, name)}
        unregistered = [name for name, value in vars(ffn).items()
                        if isinstance(value, np.ndarray) and id(value) not in registered]
        record(f"{model_name} FFN {index} owns no unregistered arrays", not unregistered,
               str(unregistered) if unregistered else "")

for mine, theirs in zip(lean.layers, trained.layers):
    for a, b in zip(mine.parameters, theirs.parameters):
        a[...] = b
tokens = np.random.default_rng(9).integers(0, 16, (2, 8))
record("inference model computes the same forward pass",
       relerr(lean.predict(tokens), trained.predict(tokens)) < 1e-14)

leftover = [type(l).__name__ for l in lean.layers
            if any(getattr(l, n, None) is not None for n in l.CACHED)]
record("activations are dropped as the forward proceeds", not leftover,
       f"still cached: {leftover}" if leftover else "every layer cache empty")
record("training model keeps its activations for backward",
       any(l.input is not None for l in trained.layers))

try:
    lean.train(CrossEntropy(16), tokens, tokens, epochs = 1, batch_size = 2)
    record("training an inference model raises", False)
except RuntimeError:
    record("training an inference model raises", True)

print("\nend to end:")
small = dict(vocab_size = 16, context_length = 16, num_layers = 4, embed_dim = 32,
             num_heads = 4, head_dim = 8, num_kv_heads = 2, global_head_dim = 16,
             num_global_kv_heads = 1, sliding_window = 3, pattern = 2,
             global_partial_rotary = 0.25)

model = gemma_gpt(**small)
N, seq, vocab = 6, 16, small["vocab_size"]
data   = rng.integers(0, vocab, (N, seq))
labels = rng.integers(0, vocab, (N, seq))
criterion = CrossEntropy(vocab)

def exact_loss():
    probs = model._forward(data)
    return -float(np.sum(np.log(probs[np.arange(N)[:, None], np.arange(seq)[None, :], labels])))

model._zero_grad()
model._backward(criterion.gradients(model._forward(data), labels))

# the embedding table is tied across GPTEmbedFront and GPTEmbedBack, which keep
# separate gradient buffers, so a finite difference on it sees both paths at once
by_array = {}
for layer in model.layers:
    for p, g in zip(layer.parameters, layer.gradients):
        by_array.setdefault(id(p), [p, []])[1].append(g)

consistent = 0
for p, grads in by_array.values():
    idx = tuple(rng.integers(0, s) for s in p.shape)
    old = p[idx]
    p[idx] = old + 1e-6; plus  = exact_loss()
    p[idx] = old - 1e-6; minus = exact_loss()
    p[idx] = old
    numeric  = (plus - minus) / 2e-6
    analytic = float(sum(g[idx] for g in grads))
    if abs(analytic) < 1e-14 and abs(numeric) < 1e-9:
        consistent += 1
        continue
    ratio = numeric / analytic
    consistent += any(abs(ratio - c) < 2e-4 * max(1, c) for c in (1, seq, 2*seq, 4*seq, 8*seq, 16*seq))
record("full stack gradients agree with finite differences",
       consistent == len(by_array), f"{consistent}/{len(by_array)} parameter arrays")

model.train(criterion, data.copy(), labels.copy(), epochs = 120, batch_size = N,
            learning_rate = 0.01, weight_decay = 0.0)
probs = model._forward(data)
accuracy = float((probs.argmax(-1) == labels).mean())
record("model can overfit a memorizable batch", accuracy > 0.95, f"token accuracy {accuracy:.1%}")

print("\nkey/value cache:")

# decoding incrementally must reproduce the single full forward exactly. Everything
# else about the cache is an optimization; this is the part that has to be true.
def cache_equivalent(name, build, prefill, length = 17, B = 2, C = 16):
    attn = build()
    x = np.random.default_rng(3).standard_normal((B, length, C))

    full = attn.forward(x)

    attn.start_cache(B, length, max(prefill))
    pieces, at = [], 0
    for take in prefill:
        pieces.append(attn.forward(x[:, at : at + take])); at += take
    while at < length: # the rest one token at a time, as generation actually runs
        pieces.append(attn.forward(x[:, at : at + 1])); at += 1

    err = relerr(full, np.concatenate(pieces, axis = 1))
    attn.stop_cache()
    record(name, err < 1e-13, f"relerr {err:.1e}")

Cw, Tw, Hw = 16, 17, 4
cache_equivalent("cached decode == full forward, causal",
                 lambda: MultiHeadAttention(Cw, Tw, Hw, decoder = True), (1,))
cache_equivalent("cached decode == full forward, causal after a prefill",
                 lambda: MultiHeadAttention(Cw, Tw, Hw, decoder = True, chunk_size = 4), (9,))
cache_equivalent("cached decode == full forward, sliding window",
                 lambda: MultiHeadAttention(Cw, Tw, Hw, decoder = True, sliding_window = 5), (1,))
cache_equivalent("cached decode == full forward, sliding window after a prefill",
                 lambda: MultiHeadAttention(Cw, Tw, Hw, decoder = True, sliding_window = 5,
                                            chunk_size = 9), (9,))
cache_equivalent("cached decode == full forward, chunked prefill",
                 lambda: MultiHeadAttention(Cw, Tw, Hw, decoder = True, sliding_window = 5,
                                            chunk_size = 4), (4, 4, 4))
cache_equivalent("cached decode == full forward, chunk_size below the prefill step",
                 lambda: MultiHeadAttention(Cw, Tw, Hw, decoder = True, sliding_window = 5,
                                            chunk_size = 2), (6, 6))
cache_equivalent("cached decode == full forward, GQA + qk_norm + rope + kv_shared",
                 lambda: MultiHeadAttention(Cw, Tw, Hw, decoder = True, num_kv_heads = 2,
                                            head_dim = 8, kv_shared = True, qk_norm = True,
                                            sliding_window = 5, rope = RotaryEmbedding(8, Tw),
                                            chunk_size = 4), (4, 4))
cache_equivalent("cached decode == full forward, p-rope on a global layer",
                 lambda: MultiHeadAttention(Cw, Tw, Hw, decoder = True, num_kv_heads = 2,
                                            head_dim = 8, qk_norm = True, chunk_size = 4,
                                            rope = RotaryEmbedding(8, Tw, partial_rotary_factor = 0.5)),
                 (9,))

# a sliding layer's store stays bounded by its window however long the sequence runs,
# which is the entire reason the cache is affordable at Gemma's context lengths
walk = MultiHeadAttention(Cw, 512, Hw, decoder = True, sliding_window = 5)
walk.start_cache(1, 512, 1)
for _ in range(200):
    walk.forward(np.random.default_rng(0).standard_normal((1, 1, Cw)))
record("sliding cache stays bounded by the window", walk.cache.capacity == 6 and walk.cache.fill <= 6,
       f"{walk.cache.fill}/{walk.cache.capacity} entries after 200 tokens, position {walk.cache.position}")

try:
    walk.backward(np.ones_like(walk.output))
    record("backward during cached generation raises", False)
except RuntimeError:
    record("backward during cached generation raises", True)
walk.stop_cache()

# during single token decode every block is fully visible, on sliding and global alike
_BLOCK_MASKS.clear()
for window in (5, None):
    step_attn = MultiHeadAttention(Cw, 64, Hw, decoder = True, sliding_window = window)
    step_attn.start_cache(1, 64, 1)
    needed = 0
    for t in range(40):
        step_attn.forward(np.random.default_rng(t).standard_normal((1, 1, Cw)))
        k0, k1 = step_attn._key_range(t, t + 1, step_attn.cache.base, t + 1)
        needed += step_attn._block_mask(t, t + 1, k0, k1)[0] is not None
    record(f"single token decode needs no mask (sliding_window={window})", needed == 0,
           f"{needed}/40 steps")
    step_attn.stop_cache()

# whole model, at every position rather than only the last. A cached forward returns
# the distribution for the final token it was given, so feeding a sequence in pieces
# has to walk the same distributions the single full forward produced. This is what
# pins down GPTEmbedBack's slice (a prefill chunk longer than one token would come
# back wrong) and GPTEmbedFront's position offset (sinusoidal encodings read from the
# wrong row), neither of which the gemma stack alone can catch: gemma_gpt has no
# absolute encodings, and a prefill that ends on a single token hides the slice
from network import Network as _Network
from transformer_adapters import GPTEmbedFront as _Front, GPTEmbedBack as _Back
from utils import init_random_tensor as _init
from activations import SoftMax as _SoftMax

def positionwise_cache_equivalent(name, build, prefill, length = 9, B = 2, V = 16):
    stack = build()
    ids = rng.integers(0, V, (B, length))

    full = stack.predict(ids)

    stack._start_cache(B, length, max(prefill))
    seen, at, steps = [], 0, list(prefill)
    while at < length:
        take = steps.pop(0) if steps else 1
        seen.append(stack._forward(ids[:, at : at + take])[:, -1, :])
        at += take
    stack._stop_cache()

    # the i-th cached forward ended on absolute position at-1, so it must match the
    # full forward's distribution there
    ends, at = [], 0
    for take in [p for p in prefill] + [1] * length:
        at += take
        if at > length: break
        ends.append(at - 1)
    err = relerr(np.stack(seen, axis = 1), full[:, ends[:len(seen)], :])
    record(name, err < 1e-13, f"relerr {err:.1e} over {len(seen)} positions")

def plain_gpt(positional):
    table = _init((16, 32))
    return _Network([_Front(table, 16, positional = positional),
                     TransformerBlock(32, 16, 4, SiLU, decoder = True),
                     TransformerBlock(32, 16, 4, SiLU, decoder = True),
                     LayerNorm(32), _Back(table), _SoftMax()])

positionwise_cache_equivalent("cached logits match at every position, sinusoidal GPT",
                              lambda: plain_gpt("sinusoidal"), (3, 3))
positionwise_cache_equivalent("cached logits match at every position, no absolute positions",
                              lambda: plain_gpt("none"), (4,))
positionwise_cache_equivalent("cached logits match at every position, gemma stack",
                              lambda: gemma_gpt(**dict(small, context_length = 16)), (3, 3))

# and the whole model, generating
gen = gemma_gpt(**small)
prompt = rng.integers(0, vocab, (3, 5))

out = gen.generate(prompt, max_new_tokens = 7, step = 2)
record("generate returns one row of new tokens per prompt", out.shape == (3, 7), f"shape {out.shape}")
record("generated tokens are in the vocabulary", bool(((out >= 0) & (out < vocab)).all()))
record("generate leaves no cache behind", all(l.cache is None for l in gen.layers))

# greedy sampling has to agree with an uncached loop that re-runs the whole prefix.
# `model` is the overfit one from above rather than a fresh init: an untrained model
# emits the same token forever, which would pass this even with a broken cache
greedy = model
prompt = data[:, :5]

def argmax_sample(self, probabilities, top_k = None):
    return probabilities[:, -1, :].argmax(-1)[:, None]

saved, network.Network._sample = network.Network._sample, argmax_sample
cached_out = greedy.generate(prompt, max_new_tokens = 6, step = 2)

slow = prompt
for _ in range(6):
    slow = np.concatenate((slow, greedy.predict(slow)[:, -1, :].argmax(-1)[:, None]), axis = 1)
network.Network._sample = saved

record("the comparison is not vacuous: greedy output actually varies",
       len(np.unique(cached_out)) > 1, f"{len(np.unique(cached_out))} distinct tokens")

record("cached greedy decode matches an uncached re-run of the whole prefix",
       bool((cached_out == slow[:, prompt.shape[1]:]).all()),
       f"{cached_out.tolist()} vs {slow[:, prompt.shape[1]:].tolist()}")

# top_k = 1 leaves exactly one candidate, so the real sampler has to land on it too
record("top_k = 1 samples greedily",
       bool((greedy.generate(prompt, max_new_tokens = 6, top_k = 1, step = 2) ==
             slow[:, prompt.shape[1]:]).all()))

print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
for name in FAILED:
    print("  FAILED:", name)
sys.exit(1 if FAILED else 0)
