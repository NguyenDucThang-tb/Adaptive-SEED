#!/usr/bin/env python3
"""Compare Qwen base and a CounterVid PEFT adapter on full VideoHallucer tasks."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from seed_video.data.transfer import VH_TASKS


def read_pairs(root, tasks):
    root = Path(root)
    result = []
    for task in tasks:
        sub = VH_TASKS[task]
        folder = root / sub
        videos = {}
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix.lower() in ('.mp4', '.mkv', '.webm', '.avi', '.mov'):
                if path.name in videos and path.resolve() != videos[path.name].resolve():
                    raise ValueError(f'Ambiguous video name: {path.name}')
                videos[path.name] = path
        seen = set()
        for ex in json.loads((folder / f'{sub}.json').read_text()):
            basic, hallucination = ex['basic'], ex['hallucination']
            signature = tuple(x for q in (basic, hallucination) for x in (q['video'], q['question']))
            if signature in seen:
                continue
            seen.add(signature)
            pair_id = f'{task}/{len(seen) - 1}'
            for role, q in (('basic', basic), ('hallucination', hallucination)):
                filename = Path(q['video']).name
                if filename not in videos:
                    raise FileNotFoundError(f'{task}: {filename}')
                answer = q['answer'].strip().lower()
                if answer not in ('yes', 'no'):
                    raise ValueError(f'Invalid answer in {pair_id}: {answer}')
                result.append(dict(task=task, pair_id=pair_id, role=role,
                                   video=str(videos[filename].resolve()), question=q['question'], answer=answer))
        if not seen:
            raise ValueError(f'No pairs in {folder}')
    return result


def summarize(rows, models=('base', 'adaptive')):
    summary = {}
    for task in sorted({row['task'] for row in rows}):
        selected = [row for row in rows if row['task'] == task]
        pairs = {}
        for row in selected:
            pairs.setdefault(row['pair_id'], []).append(row)
        if any(len(pair) != 2 or {row['role'] for row in pair} != {'basic', 'hallucination'} for pair in pairs.values()):
            raise ValueError(f'Incomplete or duplicate pair for {task}')
        summary[task] = {'questions': len(selected), 'pairs': len(pairs)}
        for model in models:
            correct = sum(row[f'{model}_prediction'] == row['answer'] for row in selected)
            pair_correct = sum(all(row[f'{model}_prediction'] == row['answer'] for row in pair)
                               for pair in pairs.values())
            summary[task][model] = dict(correct=correct, accuracy=correct / len(selected),
                                        pair_correct=pair_correct, pair_accuracy=pair_correct / len(pairs))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tasks', nargs='+', choices=sorted(VH_TASKS), default=['TPH', 'ORH'])
    parser.add_argument('--limit-pairs', type=int, help='Smoke only: first N pairs per task')
    args = parser.parse_args()
    if args.limit_pairs is not None and args.limit_pairs < 1:
        parser.error('--limit-pairs must be positive')
    if len(set(args.tasks)) != len(args.tasks):
        parser.error('Duplicate tasks')
    if args.frames < 2 or args.frames % 2:
        parser.error('--frames must be a positive even number')
    import cv2
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    items = read_pairs(args.data_root, args.tasks)
    if args.limit_pairs:
        items = [row for row in items if int(row['pair_id'].split('/')[-1]) < args.limit_pairs]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to('cuda')
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    yes = processor.tokenizer.encode('Yes', add_special_tokens=False)[0]
    no = processor.tokenizer.encode('No', add_special_tokens=False)[0]
    if yes == no:
        raise ValueError('Yes and No token IDs are identical')
    rows = []
    for item in items:
        cap = cv2.VideoCapture(item['video'])
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if not cap.isOpened() or n < 1:
                raise RuntimeError(f'Cannot read {item["video"]}')
            frames = []
            for i in np.rint(np.linspace(0, n - 1, args.frames)).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Cannot decode {item["video"]}: {i}')
                frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (448, 448)))
        finally:
            cap.release()
        prompt = processor.apply_chat_template([{'role': 'user', 'content': [
            {'type': 'video'}, {'type': 'text', 'text': item['question'].strip() + '\nAnswer with yes or no.'}]}],
            tokenize=False, add_generation_prompt=True)
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        inp = processor(text=[prompt], videos=[video], fps=args.frames / max(n / fps, 0.001), return_tensors='pt').to('cuda')
        with torch.no_grad():
            if args.adapter:
                with model.disable_adapter():
                    base = model(**inp).logits[0, -1].float()
                adaptive = model(**inp).logits[0, -1].float()
                predictions = (('base', base), ('adaptive', adaptive))
            else:
                base = model(**inp).logits[0, -1].float()
                predictions = (('base', base),)
        row = dict(item)
        for name, logits in predictions:
            margin = (logits[yes] - logits[no]).item()
            if not np.isfinite(margin):
                raise ValueError(f'Nonfinite {name} margin: {item["pair_id"]}')
            row[f'{name}_margin'] = margin
            row[f'{name}_prediction'] = 'yes' if margin > 0 else 'no'
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = summarize(rows, ('base', 'adaptive') if args.adapter else ('base',))
    metadata = dict(model=args.model, adapter=args.adapter, tasks=args.tasks, frames=args.frames, size=448,
                    yes_token=yes, no_token=no, limit_pairs=args.limit_pairs,
                    protocol='first-token Yes/No margin', results=summary)
    (out / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (out / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('EVALUATION COMPLETE:', json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
