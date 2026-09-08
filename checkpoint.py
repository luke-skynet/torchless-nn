"""Strict, text-only Gemma 4 Unified safetensors loading; no torch dependency."""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import struct

import numpy as np
from safetensors import safe_open

# Both original checkpoint and current Transformers multimodal naming.
TEXT_ONLY_EXCLUSIONS = (
    "model.embed_vision.", "model.embed_audio.", "model.vision_embedder.",
)


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple
    target: tuple
    transpose: bool = False


@dataclass(frozen=True)
class StoredTensor:
    path: Path
    offset: int
    shape: tuple
    dtype: str


def _validate_text_config(config):
    """Reject unsupported model variants and validate the required dimensions."""
    if config.get('model_type') not in ('gemma4_unified', 'gemma4_unified_text'):
        raise ValueError('Only Gemma 4 Unified text checkpoints are supported')
    text = config.get('text_config', config)
    unsupported = {
        'attention_bias': False, 'enable_moe_block': False,
        'num_kv_shared_layers': 0, 'use_double_wide_mlp': False,
        'hidden_size_per_layer_input': 0,
    }
    for key, default in unsupported.items():
        if text.get(key, default) != default:
            raise ValueError(f'Unsupported checkpoint setting: {key}={text[key]!r}')
    if text.get('quantization_config') or config.get('quantization_config'):
        raise ValueError('Quantized checkpoints are not supported; use BF16/F16/F32 safetensors')
    if text.get('use_bidirectional_attention') not in (None, 'vision'):
        raise ValueError('Only causal text attention is supported')
    if not text.get('tie_word_embeddings', config.get('tie_word_embeddings', True)):
        raise ValueError('Untied embeddings are not supported')
    if text.get('hidden_activation', 'gelu_pytorch_tanh') != 'gelu_pytorch_tanh':
        raise ValueError('Only gelu_pytorch_tanh is supported')
    required = ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                'num_attention_heads', 'num_key_value_heads', 'head_dim',
                'max_position_embeddings', 'sliding_window')
    for key in required:
        if not isinstance(text.get(key), int) or isinstance(text[key], bool) or text[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if text['intermediate_size'] % text['hidden_size']:
        raise ValueError('intermediate_size must be a multiple of hidden_size')
    return text


def _layer_types(text):
    count = text['num_hidden_layers']
    kinds = text.get('layer_types')
    if kinds is None:
        kinds = ['full_attention' if (i + 1) % 6 == 0 or i == count - 1 else
                 'sliding_attention' for i in range(count)]
    if len(kinds) != count or any(k not in ('sliding_attention', 'full_attention') for k in kinds):
        raise ValueError('Invalid layer_types')
    if kinds[-1] != 'full_attention':
        raise ValueError('The last Gemma layer must use full_attention')
    return kinds


def _global_heads(text, kinds):
    count = len(kinds)
    shared = text.get('attention_k_eq_v', False)
    overrides = text.get('per_layer_config', {})
    first_global = {}
    for key, override in overrides.items():
        if str(key).isdigit() and int(key) < count and kinds[int(key)] == 'full_attention':
            first_global = override
            break

    global_head_dim = text.get('global_head_dim', first_global.get('head_dim', 512))
    global_kv_heads = text['num_key_value_heads']
    if shared:
        global_kv_heads = (text.get('num_global_key_value_heads')
                           or first_global.get('num_key_value_heads')
                           or global_kv_heads)

    # HF may serialize per-layer overrides. Only the uniform global head overrides
    # represented by this assembler are supported.
    expected = {'head_dim': global_head_dim, 'num_key_value_heads': global_kv_heads}
    for key, override in overrides.items():
        index = int(key)
        if not 0 <= index < count or kinds[index] != 'full_attention' or any(
                k not in expected or v != expected[k] for k, v in override.items()):
            raise ValueError(f'Unsupported per_layer_config override: {key}: {override}')
    for width in (text['head_dim'], global_head_dim):
        if not isinstance(width, int) or width < 2 or width % 2:
            raise ValueError('Head dimensions must be positive and even')
    for kv_heads in (text['num_key_value_heads'], global_kv_heads):
        if not isinstance(kv_heads, int) or kv_heads <= 0 or text['num_attention_heads'] % kv_heads:
            raise ValueError('KV heads must divide the number of query heads')
    return global_head_dim, global_kv_heads


def _rope_config(text, global_head_dim):
    rope = text.get('rope_parameters', {})
    local = rope.get('sliding_attention', {'rope_type': 'default', 'rope_theta': 10000.0})
    global_rope = rope.get('full_attention', {
        'rope_type': 'proportional', 'rope_theta': 1e6, 'partial_rotary_factor': 0.25,
    })
    if local.get('rope_type') != 'default' or global_rope.get('rope_type') != 'proportional':
        raise ValueError('Expected default local RoPE and proportional global RoPE')
    if local.get('partial_rotary_factor', 1.0) != 1.0 or global_rope.get('factor', 1.0) != 1.0:
        raise ValueError('Unsupported RoPE scaling')
    fraction = global_rope.get('partial_rotary_factor', 1.0)
    if not 0 < fraction <= 1 or int(global_head_dim * fraction) < 2:
        raise ValueError('Invalid global partial_rotary_factor')
    for theta in (local['rope_theta'], global_rope['rope_theta']):
        if not math.isfinite(theta) or theta <= 0:
            raise ValueError('RoPE theta must be finite and positive')
    return local, global_rope


def model_config(config, context_length=None):
    """Translate a supported dense Gemma 4 checkpoint into assembler arguments."""
    text = _validate_text_config(config)
    kinds = _layer_types(text)
    global_head_dim, global_kv_heads = _global_heads(text, kinds)
    local, global_rope = _rope_config(text, global_head_dim)

    limit = text['max_position_embeddings']
    if context_length is None:
        context_length = limit
    if not isinstance(context_length, int) or not 1 <= context_length <= limit:
        raise ValueError(f'context_length must be in [1, {limit}]')

    epsilon = text.get('rms_norm_eps', 1e-6)
    cap = text.get('final_logit_softcapping')
    if not math.isfinite(epsilon) or epsilon <= 0 or (cap is not None and (not math.isfinite(cap) or cap <= 0)):
        raise ValueError('Invalid RMS epsilon or logit softcap')
    return dict(
        vocab_size=text['vocab_size'],
        context_length=context_length,
        num_layers=len(kinds),
        embed_dim=text['hidden_size'],
        ffn_multiplier=text['intermediate_size'] // text['hidden_size'],
        num_heads=text['num_attention_heads'],
        head_dim=text['head_dim'],
        num_kv_heads=text['num_key_value_heads'],
        global_head_dim=global_head_dim,
        num_global_kv_heads=global_kv_heads,
        sliding_window=text['sliding_window'],
        layer_types=['global' if kind == 'full_attention' else 'sliding' for kind in kinds],
        local_theta=local['rope_theta'],
        global_theta=global_rope['rope_theta'],
        global_partial_rotary=global_rope.get('partial_rotary_factor', 1.0),
        kv_shared_global=text.get('attention_k_eq_v', False),
        rms_norm_eps=epsilon,
        logit_softcap=cap,
    )


def tensor_specs(config):
    """Expected source shapes and destinations, without allocating weight arrays."""
    embed_dim = config['embed_dim']
    specs = {'embed_tokens.weight': TensorSpec((config['vocab_size'], embed_dim), ('layers', 0, 'table')),
             'norm.weight': TensorSpec((embed_dim,), ('layers', config['num_layers'] + 1, 'gamma'))}
    for i, kind in enumerate(config['layer_types']):
        full = kind == 'global'
        head_dim = config['global_head_dim'] if full else config['head_dim']
        kv_heads = config['num_global_kv_heads'] if full else config['num_kv_heads']
        query_dim = config['num_heads'] * head_dim
        root = ('layers', i + 1)
        prefix = f'layers.{i}.'

        def add(name, shape, target, transpose=False):
            specs[prefix + name] = TensorSpec(shape, root + target, transpose)

        for name, shape, attr in [('q_proj', (query_dim, embed_dim), 'q_weights'),
                                  ('k_proj', (kv_heads * head_dim, embed_dim), 'k_weights'),
                                  ('o_proj', (embed_dim, query_dim), 'out_weights')]:
            add(f'self_attn.{name}.weight', shape, ('attn_block', attr), True)
        if not (full and config['kv_shared_global']):
            add('self_attn.v_proj.weight', (kv_heads * head_dim, embed_dim), ('attn_block', 'v_weights'), True)
        for name in ('q_norm', 'k_norm'):
            add(f'self_attn.{name}.weight', (head_dim,), ('attn_block', name, 'gamma'))
        hidden = embed_dim * config['ffn_multiplier']
        for name, shape, attr in [('gate_proj', (hidden, embed_dim), 'weights1'),
                                  ('up_proj', (hidden, embed_dim), 'weights2'),
                                  ('down_proj', (embed_dim, hidden), 'weights3')]:
            add(f'mlp.{name}.weight', shape, ('ffn', attr), True)
        for name, attr in [('input_layernorm', 'pre_attn_norm'),
                           ('post_attention_layernorm', 'post_attn_norm'),
                           ('pre_feedforward_layernorm', 'pre_ffn_norm'),
                           ('post_feedforward_layernorm', 'post_ffn_norm')]:
            add(name + '.weight', (embed_dim,), (attr, 'gamma'))
        add('layer_scalar', (1,), ('layer_scalar',))
    return specs


def target_array(model, path):
    for part in path:
        model = model[part] if isinstance(part, int) else getattr(model, part)
    return model


class Checkpoint:
    """Validate all shard headers before model allocation; mmap one tensor at a time."""

    def __init__(self, directory, context_length=None):
        self.directory = Path(directory).expanduser().resolve()
        self.raw_config = json.loads((self.directory / 'config.json').read_text())
        self.config = model_config(self.raw_config, context_length)
        self.specs = tensor_specs(self.config)
        self.tensors = self._read_shards()
        self._validate_manifest()
        self.report = self._build_report()

    def _read_shards(self):
        index_path = self.directory / 'model.safetensors.index.json'
        index = json.loads(index_path.read_text())['weight_map'] if index_path.exists() else None
        files = sorted(set(index.values())) if index is not None else ['model.safetensors']
        tensors = {}
        for filename in files:
            path = self.directory / filename
            if Path(filename).name != filename or not filename.endswith('.safetensors'):
                raise ValueError(f'Invalid shard filename: {filename}')
            # Rust safetensors validates offsets, lengths, shapes, and file coverage.
            with safe_open(path, framework='numpy') as handle:
                names = list(handle.keys())
                with path.open('rb') as stream:
                    length = struct.unpack('<Q', stream.read(8))[0]
                    header = json.loads(stream.read(length))
                for name in names:
                    if name in tensors:
                        raise ValueError(f'Duplicate checkpoint tensor: {name}')
                    meta = header[name]
                    tensors[name] = StoredTensor(
                        path=path, offset=8 + length + meta['data_offsets'][0],
                        shape=tuple(meta['shape']), dtype=meta['dtype'])
        if index is not None and (set(index) != set(tensors) or any(
                tensors[name].path.name != filename for name, filename in index.items())):
            raise ValueError('Shard contents do not match the safetensors index')
        return tensors

    def _validate_manifest(self):
        candidates = [p for p in ('model.language_model.', 'model.', '')
                      if p + 'embed_tokens.weight' in self.tensors]
        if len(candidates) != 1:
            raise ValueError('Expected exactly one text embedding table')
        self.prefix = candidates[0]
        expected = {self.prefix + name for name in self.specs}
        missing = expected - self.tensors.keys()
        if missing:
            raise ValueError(f'Missing checkpoint tensors: {sorted(missing)}')
        self.skipped = sorted(
            name for name in self.tensors if name.startswith(TEXT_ONLY_EXCLUSIONS))
        self.aliases = ['lm_head.weight'] if 'lm_head.weight' in self.tensors else []
        extra = self.tensors.keys() - expected - set(self.skipped) - set(self.aliases)
        if extra:
            raise ValueError(f'Unexpected checkpoint tensors: {sorted(extra)}')
        for name, spec in self.specs.items():
            self._validate_tensor(self.prefix + name, spec.shape)
        for name in self.aliases:
            self._validate_tensor(name, self.specs['embed_tokens.weight'].shape)

    def _build_report(self):
        return dict(
            loaded_tensors=len(self.specs),
            skipped_tensors=self.skipped,
            verified_tied_aliases=self.aliases,
            weight_and_scalar_bytes=sum(math.prod(spec.shape) * 4 for spec in self.specs.values()),
            shards=len({tensor.path for tensor in self.tensors.values()}),
            context_length=self.config['context_length'],
        )

    def _validate_tensor(self, name, shape):
        tensor = self.tensors[name]
        if tensor.shape != shape:
            raise ValueError(f'{name}: expected shape {shape}, got {tensor.shape}')
        if tensor.dtype not in ('BF16', 'F16', 'F32'):
            raise ValueError(f'{name}: unsupported dtype {tensor.dtype}')

    def _array(self, name):
        tensor = self.tensors[name]
        # BF16 is read as bits, avoiding a torch dependency or NumPy BF16 support.
        return np.memmap(tensor.path, mode='r', offset=tensor.offset, shape=tensor.shape,
                         dtype={'BF16': '<u2', 'F16': '<f2', 'F32': '<f4'}[tensor.dtype])

    def chunks(self, name, transpose=False, chunk_bytes=16 * 1024**2):
        if chunk_bytes < 4:
            raise ValueError('chunk_bytes must be at least 4')
        source = self._array(name)
        view = source.T if transpose else source
        row_bytes = math.prod(view.shape[1:]) * 4
        rows = max(1, chunk_bytes // max(4, row_bytes))
        bf16 = self.tensors[name].dtype == 'BF16'
        for start in range(0, view.shape[0], rows):
            stop = min(start + rows, view.shape[0])
            block = view[start:stop]
            if bf16:
                block = (block.astype(np.uint32) << 16).view(np.float32)
            block = np.ascontiguousarray(block, dtype=np.float32)
            if not np.isfinite(block).all():
                raise ValueError(f'{name}: non-finite checkpoint values')
            yield slice(start, stop), block

    def load_into(self, model, chunk_bytes=16 * 1024**2):
        for name, spec in self.specs.items():
            destination = target_array(model, spec.target)
            for rows, block in self.chunks(self.prefix + name, spec.transpose, chunk_bytes):
                if hasattr(destination, 'set'):  # synchronous bounded host -> CUDA transfer
                    destination[rows].set(block)
                else:
                    destination[rows] = block
        self._verify_tied_aliases(chunk_bytes)
        return dict(self.report)

    def _verify_tied_aliases(self, chunk_bytes):
        for alias in self.aliases:
            left = self.chunks(alias, chunk_bytes=chunk_bytes)
            right = self.chunks(self.prefix + 'embed_tokens.weight', chunk_bytes=chunk_bytes)
            for (_, a), (_, b) in zip(left, right):
                if not np.array_equal(a, b):
                    raise ValueError('lm_head.weight differs from the tied embedding table')

    def load_model(self, chunk_size=256):
        """Construct and load from this already-validated checkpoint manifest."""
        from gemma import gemma_gpt
        from utils import inference_mode, empty_weights

        with inference_mode(), empty_weights():
            model = gemma_gpt(**self.config, chunk_size=chunk_size)
        report = self.load_into(model)
        model.set_eval(True)
        return model, report


def load_checkpoint(directory, context_length=None, chunk_size=256):
    """Return an inference-only FP32 CuPy model and its tensor accounting report."""
    return Checkpoint(directory, context_length).load_model(chunk_size=chunk_size)
