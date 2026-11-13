"""Offline Video-MME v1 evaluation of Base or an exact saved SEED adapter.

Frame-only, fixed uniform sampling; no teacher, foreground mask, or yes/no margin.
"""
import argparse
import contextlib
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from seed_video.evaluation.videomme import summarize

MODELS = {'qwen25': 'Qwen/Qwen2.5-VL-7B-Instruct',
          'qwen3': 'Qwen/Qwen3-VL-8B-Instruct',
          'internvl35': 'OpenGVLab/InternVL3_5-8B-Instruct'}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def adapter_config(path, model_id, frames, size):
    path = Path(path)
    if not path.name.endswith('_adapter.pt') or not path.is_file():
        raise ValueError('Expected an existing exact *_adapter.pt, not a training checkpoint.')
    metadata = path.with_name(path.name.removesuffix('_adapter.pt') + '.json')
    config = json.loads(metadata.read_text())
    hp = config['hp']
    for key, expected in [('model_id', model_id), ('n_frames', frames), ('frame_size', size)]:
        if hp[key] != expected:
            raise ValueError(f'Adapter {key}={hp[key]} does not match evaluation {expected}.')
    if not config.get('lora_targets') or not config.get('train'):
        raise ValueError('Adapter metadata lacks targets or training-item provenance.')
    for key in ('lora_r', 'lora_alpha', 'lora_dropout'):
        if key not in hp:
            raise ValueError(f'Adapter metadata lacks {key}')
    return config, sha(metadata)


def cached_model_fingerprint(model_id):
    from huggingface_hub import try_to_load_from_cache
    config = try_to_load_from_cache(model_id, 'config.json')
    if not isinstance(config, str) or not Path(config).is_file():
        raise ValueError(f'Model config is not cached locally: {model_id}')
    snapshot = Path(config).parent
    index = snapshot / 'model.safetensors.index.json'
    names = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    weights = {}
    for name in sorted(names):
        weight = snapshot / name
        stat = weight.stat()
        weights[name] = [weight.resolve().name, stat.st_size, stat.st_mtime_ns]
    return dict(revision=snapshot.name, config_sha=sha(config), weights=weights)


def load_runtime(model_id, frames, size, adapter=None):
    import torch
    import numpy as np
    from PIL import Image
    from functools import lru_cache

    runtime = SimpleNamespace(
        torch=torch, np=np, Image=Image, lru_cache=lru_cache,
        contextlib=contextlib, DEV='cuda', args=SimpleNamespace(base_only=True),
        CFG=dict(model_id=model_id, n_frames=frames, frame_size=size,
                 alpha=0.8, alpha_s=0.8, beta=0.4, evaluation_only=True))
    if not torch.cuda.is_available():
        raise RuntimeError('Run Video-MME inference inside a GPU allocation.')
    torch.manual_seed(0)
    backend = 'internvl' if model_id.startswith('OpenGVLab/') else 'qwen'
    importlib.import_module(f'seed_video.models.{backend}').run(runtime)
    if adapter:
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        config, _ = adapter_config(adapter, model_id, frames, size)
        hp = config['hp']
        runtime.model = get_peft_model(runtime.model, LoraConfig(
            r=hp['lora_r'], lora_alpha=hp['lora_alpha'], lora_dropout=hp['lora_dropout'],
            target_modules=config['lora_targets'], bias='none'))
        state = torch.load(adapter, map_location='cpu', weights_only=True)
        expected = get_peft_model_state_dict(runtime.model)
        if expected.keys() != state.keys() or any(expected[k].shape != state[k].shape for k in expected):
            raise ValueError('Adapter tensor names/shapes do not match the model; refusing partial load.')
        set_peft_model_state_dict(runtime.model, state)
    runtime.model.eval()
    runtime.model.requires_grad_(False)
    runtime.model.config.use_cache = True
    importlib.import_module('seed_video.data.video').run(runtime)
    return runtime


def generate_answer(runtime, item):
    from seed_video.data.videomme import build_prompt
    from seed_video.training.losses import response_tokens

    frames, duration = runtime._decode(item['video'], runtime.CFG['n_frames'], runtime.CFG['frame_size'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'Invalid duration: {item["video"]}')
    inputs = runtime.build_inputs(frames, build_prompt(item), len(frames) / duration, yn=False).to(runtime.DEV)
    for module in runtime.model.modules():
        if hasattr(module, 'rope_deltas'):
            module.rope_deltas = None
    generate = getattr(runtime, 'generate_response', None)
    with runtime.torch.inference_mode(), runtime.keep_last(1):
        kwargs = dict(max_new_tokens=8, do_sample=False)
        outputs = generate(inputs, **kwargs) if generate else runtime.model.generate(**inputs, **kwargs)
    tokens = response_tokens(outputs, inputs['input_ids'], runtime.torch)
    return runtime.tok.decode(tokens, skip_special_tokens=True).strip()


def run(args):
    from seed_video.data.videomme import load_videomme, parse_answer
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    items = load_videomme(args.data_root)
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise ValueError('No evaluation questions.')
    adapter = args.adapter.resolve() if args.adapter else None
    adapter_meta = adapter_config(adapter, MODELS[args.model], args.n_frames, args.frame_size) if adapter else None
    video_stats = {}
    for item in items:
        path = Path(item['video'])
        stat = path.stat()
        video_stats[str(path)] = [stat.st_size, stat.st_mtime_ns]
    repo = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__), repo / 'scripts/decode_eof_fallback.py',
                    *[repo / f'src/seed_video/{name}' for name in (
                        'data/videomme.py', 'data/video.py', 'models/qwen.py', 'models/internvl.py',
                        'evaluation/videomme.py', 'training/losses.py')]]
    protocol = dict(version='videomme_v1_frame_only_v1', model=MODELS[args.model],
                    frames=args.n_frames, frame_size=args.frame_size, subtitles=False, audio=False,
                    decoding='decord_single_thread_eof_sequential', generation=dict(do_sample=False, max_new_tokens=8),
                    limit=args.limit, annotation_sha=sha(args.data_root / 'questions.json'),
                    model_checkpoint=cached_model_fingerprint(MODELS[args.model]),
                    inputs=items, video_stats=video_stats,
                    source={str(p.relative_to(repo)): sha(p) for p in source_files},
                    packages={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'decord')})
    meta = dict(protocol=protocol, adapter=str(adapter) if adapter else None,
                adapter_sha=sha(adapter) if adapter else None,
                adapter_metadata_sha=adapter_meta[1] if adapter_meta else None)
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output / '.worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another evaluator owns this output directory.')
        checkpoint = args.output / 'predictions.json'
        records = []
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved['meta'] != meta:
                raise ValueError('Evaluation data/protocol/adapter changed. Use a new output directory.')
            records = saved['predictions']
            ids = [r['question_id'] for r in records]
            if len(set(ids)) != len(ids) or ids != [r['question_id'] for r in items[:len(ids)]]:
                raise ValueError('Resume predictions do not match the ordered question manifest.')
        atomic_json(checkpoint, dict(meta=meta, predictions=records))
        if len(records) < len(items):
            runtime = load_runtime(MODELS[args.model], args.n_frames, args.frame_size, adapter)
            fallback_spec = importlib.util.spec_from_file_location('videomme_eof', repo / 'scripts/decode_eof_fallback.py')
            fallback = importlib.util.module_from_spec(fallback_spec)
            fallback_spec.loader.exec_module(fallback)
            audit = {}
            audit_path = args.output / 'decoder_audit.json'
            if audit_path.exists():
                audit.update(json.loads(audit_path.read_text()))
            runtime.VideoReader = fallback.wrap_reader(runtime.VideoReader, audit, lambda: atomic_json(audit_path, audit))
            try:
                for item in items[len(records):]:
                    answer = generate_answer(runtime, item)
                    records.append(dict(item, response=answer, prediction=parse_answer(answer)))
                    print(f'Video-MME {len(records)}/{len(items)}: {item["question_id"]}', flush=True)
                    if len(records) % args.checkpoint_every == 0:
                        atomic_json(checkpoint, dict(meta=meta, predictions=records))
            finally:
                atomic_json(checkpoint, dict(meta=meta, predictions=records))
        rows = summarize(records)
        with (args.output / 'results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['group', 'value', 'n', 'accuracy', 'invalid'])
            writer.writeheader()
            writer.writerows(rows)
        atomic_json(args.output / 'COMPLETE.json', dict(n=len(records), smoke=args.limit is not None,
                    protocol_sha=hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
                    predictions_sha=sha(checkpoint)))
        print('VIDEOMME COMPLETE', args.output, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--n-frames', type=int, default=8)
    parser.add_argument('--frame-size', type=int, default=448)
    parser.add_argument('--limit', type=int, help='Smoke only, never report as full Video-MME.')
    parser.add_argument('--checkpoint-every', type=int, default=5)
    args = parser.parse_args(argv)
    if args.n_frames < 2 or args.n_frames % 2 or args.frame_size < 28 or args.checkpoint_every < 1:
        parser.error('Require positive even n-frames>=2, frame-size>=28 and checkpoint-every>=1.')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    run(args)


if __name__ == '__main__':
    main()
