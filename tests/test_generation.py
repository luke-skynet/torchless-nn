import json
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, AutoTokenizer

import cupy as xp
from checkpoint import load_checkpoint
from generate import encode_prompt, stop_tokens, main
from test_checkpoint import reference, save_reference, host


@pytest.fixture
def tokenizer(tmp_path):
    vocab = {'<pad>': 0, '<eos>': 1, '<bos>': 2, '<end_of_turn>': 3,
             '<unk>': 4, 'user': 5, 'assistant': 6, 'hello': 7, 'world': 8}
    backend = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='<bos>', eos_token='<eos>',
                                  pad_token='<pad>', unk_token='<unk>',
                                  additional_special_tokens=['<end_of_turn>'])
    tok.chat_template = "{{ bos_token }}{% for m in messages %}{{ m['role'] }} {{ m['content'] }}<end_of_turn>{% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
    tok.save_pretrained(tmp_path)
    return AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)


def test_text_to_checkpoint_to_generated_text(reference, tokenizer, tmp_path):
    save_reference(reference, tmp_path)
    (tmp_path / 'generation_config.json').write_text(json.dumps({'eos_token_id': [1, 3]}))
    ids = encode_prompt(tokenizer, 'hello world')
    assert ids == [2, 5, 7, 8, 3, 6]
    stops = stop_tokens(tmp_path, tokenizer, reference.config.to_dict())
    assert stops == [1, 3]
    ours, _ = load_checkpoint(tmp_path, 32, chunk_size=2)
    events = []
    generated = ours.generate(ids, 7, temperature=0, stop=stops, step=3, on_event=events.append)
    with torch.no_grad():
        expected = reference.generate(torch.tensor([ids]), max_new_tokens=7, do_sample=False,
                                      eos_token_id=stops, pad_token_id=0)
    np.testing.assert_array_equal(host(generated), expected[:, len(ids):].numpy())
    assert tokenizer.decode(host(generated[0]).tolist(), skip_special_tokens=True) == tokenizer.decode(
        expected[0, len(ids):].tolist(), skip_special_tokens=True)
    assert events == ['prefill_start', 'prefill_end', 'generation_end']


def test_prompt_modes(tokenizer):
    assert encode_prompt(tokenizer, 'hello world', raw=True) == tokenizer('hello world')['input_ids']
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match='chat template'):
        encode_prompt(tokenizer, 'hello')


@pytest.mark.parametrize('options', [{'step': 0}, {'step': -1}, {'temperature': -1},
                                     {'temperature': float('nan')}, {'temperature': float('inf')},
                                     {'top_k': 0}, {'max_new_tokens': -1}, {'max_new_tokens': 40}])
def test_invalid_generation_options(reference, tmp_path, options):
    save_reference(reference, tmp_path)
    model, _ = load_checkpoint(tmp_path, 32)
    kwargs = {'max_new_tokens': 2, **options}
    with pytest.raises(ValueError):
        model.generate([2, 4], **kwargs)
    assert all(layer.cache is None for layer in model.layers)


@pytest.mark.parametrize('ids', [[], [1.5], [-1], [64], [[[2]]]])
def test_invalid_prompt_tokens(reference, tmp_path, ids):
    save_reference(reference, tmp_path)
    model, _ = load_checkpoint(tmp_path, 32)
    with pytest.raises(ValueError):
        model.generate(ids, 2)


def test_stopping_rows_and_cleanup(reference, tmp_path, monkeypatch):
    save_reference(reference, tmp_path)
    model, _ = load_checkpoint(tmp_path, 32)
    model.set_eval(False)
    samples = iter([[[1], [5]], [[7], [3]]])
    monkeypatch.setattr(model, '_sample', lambda *args: xp.asarray(next(samples)))
    events = []
    result = model.generate([[2], [2]], 8, stop=[1, 3], temperature=0.7, on_event=events.append)
    np.testing.assert_array_equal(host(result), [[1, 1], [5, 3]])
    assert model.layers[-1].temperature == 1.0
    assert not model.layers[1].ffn.dropout.eval_mode
    assert all(layer.cache is None for layer in model.layers)
    assert model.layers[1].attn_block.cache is None


def test_failure_during_cache_allocation_cleans_up(reference, tmp_path, monkeypatch):
    save_reference(reference, tmp_path)
    model, _ = load_checkpoint(tmp_path, 32)
    model.set_eval(False)
    def fail(*args):
        raise RuntimeError('allocation failed')
    monkeypatch.setattr(model.layers[2], 'start_cache', fail)
    with pytest.raises(RuntimeError, match='allocation failed'):
        model.generate([2], 4, temperature=0.5)
    assert all(layer.cache is None for layer in model.layers)
    assert model.layers[1].attn_block.cache is None
    assert model.layers[-1].temperature == 1.0
    assert not model.layers[1].ffn.dropout.eval_mode


def test_no_unused_final_forward(reference, tmp_path, monkeypatch):
    save_reference(reference, tmp_path)
    model, _ = load_checkpoint(tmp_path, 32)
    calls = []
    original = model._forward
    def forward(x):
        calls.append(x.shape[1])
        return original(x)
    monkeypatch.setattr(model, '_forward', forward)
    model.generate([2, 3, 4, 5], 3, step=2, temperature=0)
    assert calls == [2, 2, 1, 1]
    calls.clear()
    assert model.generate([2], 0).shape == (1, 0)
    assert not calls


def test_inspect_cli_needs_no_cuda(reference, tmp_path, capsys):
    save_reference(reference, tmp_path)
    assert main(['--checkpoint', str(tmp_path), '--context-length', '32', '--inspect']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['loaded_tensors'] > 0
    assert report['shards'] == 2
