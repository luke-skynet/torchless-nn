"""Text generation from a local Gemma 4 Unified checkpoint on one CUDA GPU."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


def encode_prompt(tokenizer, prompt, raw=False):
    if raw:
        return tokenizer(prompt, add_special_tokens=True)['input_ids']
    if not tokenizer.chat_template:
        raise ValueError('No checkpoint chat template; use --raw-prompt for base-model completion')
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                         tokenize=True, add_generation_prompt=True, return_dict=False)


def stop_tokens(directory, tokenizer, config):
    path = Path(directory) / 'generation_config.json'
    generation = json.loads(path.read_text()) if path.exists() else {}
    text_config = config.get('text_config', config)
    default_eos = config.get('eos_token_id', text_config.get('eos_token_id'))
    eos = generation.get('eos_token_id', default_eos)
    if eos is None:
        result = []
    elif isinstance(eos, int):
        result = [eos]
    else:
        result = list(eos)
    if tokenizer.eos_token_id is not None:
        result.append(tokenizer.eos_token_id)
    # Gemma instruction turns can end with a token distinct from EOS.
    vocab = tokenizer.get_vocab()
    if '<end_of_turn>' in vocab:
        result.append(vocab['<end_of_turn>'])
    return sorted(set(result))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path, help='Local Hugging Face checkpoint directory')
    p.add_argument('--prompt', help='User message, or completion prefix with --raw-prompt')
    p.add_argument('--raw-prompt', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--context-length', type=int, default=8192, help='Allocated context, at most the checkpoint limit')
    p.add_argument('--temperature', type=float, default=0.0, help='0 is deterministic greedy decoding')
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--prefill-step', type=int, default=256)
    p.add_argument('--chunk-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--tf32', action='store_true', help='Enable TF32; default FP32 is preferable for parity checks')
    p.add_argument('--inspect', action='store_true', help='Validate config and shard headers without CUDA or loading weights')
    p.add_argument('--report', type=Path, help='Also save JSON tensor accounting and timing metrics here')
    return p


def prepare_prompt(args, checkpoint):
    """Validate generation options and tokenize before allocating GPU memory."""
    if min(args.max_new_tokens, args.prefill_step, args.chunk_size, args.top_k) < 1:
        raise ValueError('Token counts, step, chunk size, and top-k must be positive')
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False)
    tokens = encode_prompt(tokenizer, args.prompt, args.raw_prompt)
    if not tokens or len(tokens) + args.max_new_tokens > args.context_length:
        raise ValueError('Prompt plus requested generation exceeds --context-length (or prompt is empty)')
    stops = stop_tokens(args.checkpoint, tokenizer, checkpoint.raw_config)
    vocab_size = checkpoint.config['vocab_size']
    if any(token < 0 or token >= vocab_size for token in tokens + stops):
        raise ValueError('Tokenizer IDs do not fit the checkpoint vocabulary')
    return tokenizer, tokens, stops


def configure_cuda(args):
    # Set TF32 before importing CuPy. --help and --inspect never need CUDA.
    os.environ['CUPY_TF32'] = '1' if args.tf32 else '0'
    import cupy as cp
    from cupy.cuda import cublas

    device = cp.cuda.Device(args.device)
    device.use()
    math_mode = cublas.CUBLAS_TF32_TENSOR_OP_MATH if args.tf32 else cublas.CUBLAS_DEFAULT_MATH
    cublas.setMathMode(device.cublas_handle, math_mode)
    return cp


class RunMetrics:
    """Allocator high-water marks and synchronized generation event timestamps."""

    def __init__(self, cp):
        self.cp = cp
        self.pool = cp.get_default_memory_pool()
        self.peak_used = 0
        self.peak_reserved = 0
        self.marks = {}

    def allocate(self, size):
        pointer = self.pool.malloc(size)
        self.peak_used = max(self.peak_used, self.pool.used_bytes())
        self.peak_reserved = max(self.peak_reserved, self.pool.total_bytes())
        return pointer

    def event(self, name):
        self.cp.cuda.get_current_stream().synchronize()
        self.marks[name] = time.perf_counter()

    def report(self, prompt_tokens, generated_tokens):
        prefill = self.marks['prefill_end'] - self.marks['prefill_start']
        decode = self.marks['generation_end'] - self.marks['prefill_end']
        return dict(
            peak_cupy_used_bytes=self.peak_used,
            peak_cupy_reserved_bytes=self.peak_reserved,
            load_seconds=self.marks['load_end'] - self.marks['load_start'],
            prefill_seconds=prefill,
            generation_seconds=decode,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            prefill_tokens_per_second=prompt_tokens / prefill,
            generated_tokens_per_second=generated_tokens / decode,
            memory_note='CuPy allocator high-water marks; excludes CUDA context and external workspaces',
        )


def run_generation(args, checkpoint, tokens, stops):
    cp = configure_cuda(args)
    metrics = RunMetrics(cp)
    with cp.cuda.using_allocator(metrics.allocate):
        print(f'Loading {checkpoint.report["loaded_tensors"]} text tensors in FP32...', file=sys.stderr)
        metrics.event('load_start')
        model, report = checkpoint.load_model(chunk_size=args.chunk_size)
        metrics.event('load_end')
        model.rng = cp.random.default_rng(args.seed)
        generated = model.generate(
            tokens, args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, step=args.prefill_step, stop=stops, on_event=metrics.event)
        ids = cp.asnumpy(generated[0]).tolist()

    report.update(metrics.report(len(tokens), len(ids)), tf32=args.tf32, device=args.device)
    return ids, report


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    from checkpoint import Checkpoint

    try:
        checkpoint = Checkpoint(args.checkpoint, args.context_length)
        if args.inspect:
            print(json.dumps(checkpoint.report, indent=2))
            return 0
        if args.prompt is None:
            cli.error('--prompt is required unless --inspect is used')

        tokenizer, tokens, stops = prepare_prompt(args, checkpoint)
        ids, report = run_generation(args, checkpoint, tokens, stops)
        print(tokenizer.decode(ids, skip_special_tokens=True))
        report_json = json.dumps(report, indent=2)
        print(report_json, file=sys.stderr)
        if args.report:
            args.report.write_text(report_json + '\n')
        return 0
    except (ValueError, OSError, ImportError) as error:
        cli.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
