import json
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import Gemma4UnifiedTextConfig, Gemma4UnifiedForCausalLM

from backend import xp
from checkpoint import Checkpoint, load_checkpoint, model_config
from gemma import gemma_gpt
from backend import inference_mode, empty_weights


@pytest.mark.parametrize('declared', [False, True])
def test_checkpoint_epsilon_fallback(reference, tmp_path, declared):
    from layers import RMSNorm, TransformerBlock
    save_reference(reference, tmp_path)
    path = tmp_path / 'config.json'
    config = json.loads(path.read_text())
    text = config.get('text_config', config)
    expected = text['rms_norm_eps'] if declared else .02
    if not declared:
        del text['rms_norm_eps']
    path.write_text(json.dumps(config))
    model, _ = load_checkpoint(tmp_path, default_rms_norm_eps=.02)
    norms = []
    for layer in model.layers:
        if isinstance(layer, RMSNorm):
            norms.append(layer)
        elif isinstance(layer, TransformerBlock):
            norms.extend(child for child in layer.blocks if isinstance(child, RMSNorm))
            norms.extend(child for child in (layer.attn_block.q_norm, layer.attn_block.k_norm,
                                             layer.attn_block.v_norm) if child is not None)
    assert norms and all(norm.eps == expected for norm in norms)


def host(array):
    return array.get() if hasattr(array, 'get') else np.asarray(array)


@pytest.fixture
def reference():
    torch.manual_seed(721)
    config = Gemma4UnifiedTextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        global_head_dim=16, num_global_key_value_heads=1, attention_k_eq_v=True,
        sliding_window=4, max_position_embeddings=32, rms_norm_eps=3e-5,
        layer_types=['sliding_attention', 'sliding_attention', 'full_attention'],
        final_logit_softcapping=30.0)
    config._attn_implementation = 'eager'
    model = Gemma4UnifiedForCausalLM(config).eval()
    # Nontrivial norm/scalar values prevent missing mappings from passing by default.
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            if 'norm' in name:
                tensor.uniform_(0.7, 1.3)
        for index, block in enumerate(model.model.layers):
            block.layer_scalar.fill_(0.8 + index * 0.15)
    return model


def save_reference(model, directory, bf16=False, multimodal=False):
    config = model.config.to_dict()
    if multimodal:
        config = {'model_type': 'gemma4_unified', 'text_config': config}
    (directory / 'config.json').write_text(json.dumps(config))
    state = {}
    for name, tensor in model.state_dict().items():
        if name == 'lm_head.weight':
            continue
        if multimodal:
            name = name.replace('model.', 'model.language_model.', 1)
        state[name] = tensor.detach().clone().to(torch.bfloat16 if bf16 else torch.float32)
    if multimodal:
        state['model.embed_vision.fake.weight'] = torch.ones(2, 2)
        state['model.embed_audio.fake.weight'] = torch.ones(2, 2)
    names = list(state)
    split = len(names) // 2
    weight_map = {}
    for i, group in enumerate((names[:split], names[split:])):
        filename = f'model-{i + 1:05d}-of-00002.safetensors'
        save_file({name: state[name] for name in group}, directory / filename)
        weight_map.update({name: filename for name in group})
    (directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
    if bf16:
        # Reference computes FP32 using exactly the stored BF16-rounded weights.
        model.load_state_dict({k: v.to(torch.bfloat16).float() for k, v in model.state_dict().items()})


def capture_ours(model, monkeypatch):
    outputs = {}
    modules = {'embed_tokens': model.layers[0], 'norm': model.layers[4]}
    for i, block in enumerate(model.layers[1:4]):
        modules[f'layers.{i}'] = block
        modules[f'layers.{i}.self_attn'] = block.attn_block
        modules[f'layers.{i}.mlp'] = block.ffn
        for theirs, ours in [('input_layernorm', 'pre_attn_norm'),
                             ('post_attention_layernorm', 'post_attn_norm'),
                             ('pre_feedforward_layernorm', 'pre_ffn_norm'),
                             ('post_feedforward_layernorm', 'post_ffn_norm')]:
            modules[f'layers.{i}.{theirs}'] = getattr(block, ours)
    for name, module in modules.items():
        original = module.forward
        def forward(x, original=original, name=name):
            result = original(x)
            outputs[name] = host(result).copy()
            return result
        monkeypatch.setattr(module, 'forward', forward)
    return outputs


@pytest.mark.parametrize('bf16,multimodal', [(False, False), (True, True)])
def test_reference_intermediates_and_cached_logits(reference, tmp_path, monkeypatch, bf16, multimodal):
    save_reference(reference, tmp_path, bf16, multimodal)
    device = os.environ.get('TORCHLESS_TEST_DEVICE', 'cpu')
    reference.to(device)
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
    with patch('backend.init_random_tensor', side_effect=AssertionError('checkpoint construction must not use RNG')):
        ours, report = load_checkpoint(tmp_path, context_length=32, chunk_size=2)
    assert ours.layers[0].table is ours.layers[-3].table
    assert len(report['skipped_tensors']) == (2 if multimodal else 0)
    assert not any(layer.gradients or layer.moments or layer.variances for layer in ours.layers)
    actual = capture_ours(ours, monkeypatch)
    expected, handles = {}, []
    for name, module in reference.model.named_modules():
        if name in ('embed_tokens', 'norm') or name.startswith('layers.'):
            def hook(module, args, output, name=name):
                output = output[0] if isinstance(output, tuple) else output
                if isinstance(output, torch.Tensor):
                    expected[name] = output.detach().cpu().numpy().copy()
            handles.append(module.register_forward_hook(hook))
    ids = np.random.default_rng(12).integers(3, 64, (2, 13), dtype=np.int64)
    with torch.no_grad():
        ref = reference(torch.tensor(ids, device=device), use_cache=False).logits.cpu().numpy()
    logits = host(ours.predict_logits(xp.asarray(ids)))
    for name, value in actual.items():
        np.testing.assert_allclose(value, expected[name], atol=3e-5, rtol=3e-4, err_msg=name)
    np.testing.assert_allclose(logits, ref, atol=3e-5, rtol=3e-4)
    for handle in handles:
        handle.remove()
    # Force compaction across several windows, with prefill blocks longer than one.
    for prefill in (1, 3, 7):
        ours._start_cache(2, 32, prefill)
        try:
            position = 0
            hf_cache = None
            for size in [prefill] + [1] * (ids.shape[1] - prefill):
                block = ids[:, position:position + size]
                with torch.no_grad():
                    result = reference(torch.tensor(block, device=device), past_key_values=hf_cache,
                                       use_cache=True)
                hf_cache = result.past_key_values
                cached = host(ours.predict_logits(xp.asarray(block)))
                position += size
                np.testing.assert_allclose(cached[:, -1], ref[:, position - 1], atol=3e-5, rtol=3e-4)
                np.testing.assert_allclose(cached[:, -1], result.logits[:, -1].cpu().numpy(), atol=3e-5, rtol=3e-4)
        finally:
            ours._stop_cache()


def test_loader_matches_all_transposes_and_scalars(reference, tmp_path):
    save_reference(reference, tmp_path, bf16=True)
    checkpoint = Checkpoint(tmp_path, 16)
    with inference_mode(), empty_weights():
        ours = gemma_gpt(**checkpoint.config)
    checkpoint.load_into(ours, chunk_bytes=64)
    # Check independently named destinations, with deliberately tiny transfer chunks.
    np.testing.assert_array_equal(host(ours.layers[1].ffn.act_weights),
                                  reference.model.layers[0].mlp.gate_proj.weight.detach().numpy().T)
    np.testing.assert_array_equal(host(ours.layers[3].attn_block.q_weights),
                                  reference.model.layers[2].self_attn.q_proj.weight.detach().numpy().T)
    np.testing.assert_array_equal(host(ours.layers[2].layer_scalar),
                                  reference.model.layers[1].layer_scalar.numpy())
    assert ours.layers[1].pre_attn_norm.eps == 3e-5


def test_hf_save_pretrained_roundtrip(reference, tmp_path):
    reference.save_pretrained(tmp_path, max_shard_size='10KB')
    ours, _ = load_checkpoint(tmp_path, 16)
    ids = np.array([[2, 4, 9, 7]], dtype=np.int64)
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids), use_cache=False).logits.numpy()
    np.testing.assert_allclose(host(ours.predict_logits(xp.asarray(ids))), expected, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize('fault', ['missing', 'extra', 'shape', 'dtype', 'alias'])
def test_bad_tensors_fail(reference, tmp_path, fault):
    config = reference.config.to_dict()
    (tmp_path / 'config.json').write_text(json.dumps(config))
    state = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    if fault == 'missing':
        del state['model.layers.0.layer_scalar']
    elif fault == 'extra':
        state['model.surprise.weight'] = torch.ones(1)
    elif fault == 'shape':
        state['model.layers.0.self_attn.q_proj.weight'] = torch.ones(1, 1)
    elif fault == 'dtype':
        state['model.norm.weight'] = torch.ones(32, dtype=torch.int64)
    else:
        state['lm_head.weight'].add_(1)
    save_file(state, tmp_path / 'model.safetensors')
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path, 16)


@pytest.mark.parametrize('key,value', [('attention_bias', True), ('num_kv_shared_layers', 1),
                                      ('use_bidirectional_attention', 'all'), ('enable_moe_block', True),
                                      ('tie_word_embeddings', False)])
def test_unsupported_config_rejected(reference, key, value):
    config = reference.config.to_dict()
    config[key] = value
    with pytest.raises(ValueError):
        model_config(config)


def test_official_12b_tensor_manifest():
    """Actual Google checkpoint headers, without allocating/downloading its weights."""
    from pathlib import Path
    from checkpoint import tensor_specs, TEXT_ONLY_EXCLUSIONS
    fixture = json.loads((Path(__file__).parent / 'fixtures/gemma4_12b_manifest.json').read_text())
    specs = tensor_specs(model_config(fixture['config']))
    remaining = dict(fixture['tensors'])
    for name, spec in specs.items():
        metadata = remaining.pop('model.language_model.' + name)
        assert tuple(metadata['shape']) == spec.shape, name
        assert metadata['dtype'] in ('BF16', 'F16', 'F32')
    assert remaining
    assert all(name.startswith(TEXT_ONLY_EXCLUSIONS) for name in remaining)


def test_fp16_and_tied_alias(reference, tmp_path):
    state = {name: value.detach().clone().half() for name, value in reference.state_dict().items()}
    save_file(state, tmp_path / 'model.safetensors')
    (tmp_path / 'config.json').write_text(json.dumps(reference.config.to_dict()))
    ours, report = load_checkpoint(tmp_path, 16)
    assert report['verified_tied_aliases'] == ['lm_head.weight']
    np.testing.assert_array_equal(host(ours.layers[0].table), state['model.embed_tokens.weight'].float().numpy())


def test_index_mismatch(reference, tmp_path):
    save_reference(reference, tmp_path)
    path = tmp_path / 'model.safetensors.index.json'
    index = json.loads(path.read_text())
    index['weight_map']['model.norm.weight'] = 'model-00001-of-00002.safetensors'
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match='index'):
        Checkpoint(tmp_path)
