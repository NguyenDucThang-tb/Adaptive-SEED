"""Extend the legacy VH-trained Qwen2.5 Both sweep without overwriting it."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


VALUES = (0, 0.25, 0.5, 0.75, 0.9, 1)
REL = Path('cdt_distill_cache_f8_448/seed_component_ablation/runs')
RUNS = ('seed_base_s0', 'adaptive_pos_neg_s0')


def tag(ap, an):
    return f'ap{ap:g}_an{an:g}_tf32'


def result_path(work, ap, an):
    return work / REL / tag(ap, an) / 'results.csv'


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def complete(rows, run):
    # The legacy full evaluation has 310 VH questions and 409 EH questions.
    for split, count, metrics in (
        ('VH_TEST', 310, ('q_acc', 'pair_acc')),
        ('EH_ALL', 409, ('q_acc',)),
    ):
        found = [r for r in rows if r.get('run') == run and r.get('split') == split]
        if len(found) != 1:
            return False
        try:
            if float(found[0]['n']) != count:
                return False
            if any(not math.isfinite(float(found[0][m])) or
                   not 0 <= float(found[0][m]) <= 1 for m in metrics):
                return False
        except (KeyError, ValueError, TypeError):
            return False
    return True


def work_for(root, ap, an):
    return root / f'ap_{ap:g}'.replace('.', 'p') / f'an_{an:g}'.replace('.', 'p')


def plan(old, root):
    points = []
    for ap in VALUES:
        for an in VALUES:
            previous = read_rows(result_path(old, ap, an))
            current = read_rows(result_path(work_for(root, ap, an), ap, an))
            done = {run: complete(previous, run) or complete(current, run) for run in RUNS}
            points.append((ap, an, done))
    return points


def protocol(repo, args):
    digest = hashlib.sha256()
    for path in sorted((repo / 'src').rglob('*.py')):
        digest.update(str(path.relative_to(repo)).encode())
        digest.update(path.read_bytes())
    return dict(source_sha256=digest.hexdigest(), model='Qwen/Qwen2.5-VL-7B-Instruct',
                old_work=str(args.old_work.resolve()), env=str(args.env_archive.resolve()),
                hf_cache=str(args.hf_cache.resolve()), vh=str(args.vh_data.resolve()),
                eh=str(args.eh_data.resolve()), teacher_frames=32, train_dataset='videohallucer')


def initialize(repo, args):
    root = args.root
    config = protocol(repo, args)
    saved = root / 'protocol.json'
    if saved.exists():
        if json.loads(saved.read_text()) != config:
            raise RuntimeError('Protocol/code changed: use a new root, not incompatible checkpoints.')
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('New root must be empty; refusing to mix experiments.')
    root.mkdir(parents=True, exist_ok=True)
    code = root / 'code'
    shutil.copytree(repo / 'src', code / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    (code / 'scripts').mkdir()
    for name in ('submit_legacy_q25_grid.py', 'run_legacy_q25_grid.pbs'):
        shutil.copy2(repo / 'scripts' / name, code / 'scripts' / name)
    (root / 'logs').mkdir()
    saved.write_text(json.dumps(config, indent=2) + '\n')


def require_inputs(args):
    for path in (args.env_archive, args.vh_data, args.eh_data / 'questions/entire_questions.json',
                 args.eh_data / 'questions/mix_questions.json',
                 args.eh_data / 'questions/misleading_questions.json'):
        if not path.exists():
            raise RuntimeError(f'Missing input: {path}')
    for model in ('models--Qwen--Qwen2.5-VL-7B-Instruct', 'models--ZhengPeng7--BiRefNet'):
        if not any((args.hf_cache / model / 'snapshots').glob('*/config.json')):
            raise RuntimeError(f'Missing cached model: {model}')


def submit(repo, args, points):
    require_inputs(args)
    journal = args.root / 'jobs.tsv'
    if journal.exists():
        active = subprocess.check_output(['qstat', '-u', os.environ['USER']], text=True)
        old_ids = {line.split('\t')[0].split('.')[0] for line in journal.read_text().splitlines()}
        active_ids = set(re.findall(r'^\s*(\d+)\.', active, re.MULTILINE))
        if old_ids & active_ids:
            raise RuntimeError('This grid still has active jobs; wait before resubmitting.')
    initialize(repo, args)
    parents = {}
    for ap, an, done in points:
        if all(done.values()):
            continue
        work = work_for(args.root, ap, an)
        work.mkdir(parents=True, exist_ok=True)
        for name, source in (('VideoHallucer', args.vh_data), ('EventHallusion', args.eh_data)):
            link = work / name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(source.resolve(), target_is_directory=True)
            if link.resolve() != source.resolve():
                raise RuntimeError(f'Unexpected data link: {link}')
        dest = result_path(work, ap, an)
        previous = result_path(args.old_work, ap, an)
        if not dest.exists() and previous.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous, dest)
        name = f'q25old_{ap:g}_{an:g}'.replace('.', 'p')
        env = f'GRID_ROOT={args.root},POINT_WORK={work},AP={ap:g},AN={an:g}'
        if ',' in str(args.root):
            raise RuntimeError('PBS paths cannot contain commas.')
        cmd = ['qsub', '-P', 'hn98', '-q', 'gpuhopper',
               '-l', 'ncpus=12,ngpus=1,mem=160GB,jobfs=100GB,walltime=01:00:00',
               '-l', 'storage=scratch/hn98+scratch/jp09+gdata/hn98',
               '-v', env, '-N', name,
               '-o', str(args.root / 'logs' / f'{name}.out'), '-j', 'oe']
        if ap in parents:
            # Each pair has separate checkpoints: timeout must not cancel other pairs.
            cmd += ['-W', f'depend=afterany:{parents[ap]}']
        cmd += [str(args.root / 'code/scripts/run_legacy_q25_grid.pbs')]
        job = subprocess.check_output(cmd, text=True).strip()
        parents[ap] = job
        with journal.open('a') as stream:
            stream.write(f'{job}\t{ap:g}\t{an:g}\t{work}\n')
        print(f'SUBMITTED {ap:g}:{an:g} -> {job}', flush=True)


def worker(args):
    config = json.loads((args.root / 'protocol.json').read_text())
    ap, an = args.point
    work = work_for(args.root, ap, an)
    rows = read_rows(result_path(work, ap, an))
    needed = [run.removesuffix('_s0') for run in RUNS if not complete(rows, run)]
    if not needed:
        print('SKIP: full Base_fix and Adaptive results already exist', flush=True)
        return
    print(f'POINT {ap:g}:{an:g}; arms={needed}; WORK={work}', flush=True)
    cmd = [sys.executable, '-u', '-m', 'seed_video', '--stage', 'core', '--work-dir', str(work),
           '--train-dataset', 'videohallucer', '--teacher-frames', '32',
           '--a-pos', str(ap), '--a-neg', str(an), '--arms', *needed]
    subprocess.run(cmd, check=True)
    rows = read_rows(result_path(work, ap, an))
    if not all(complete(rows, run) for run in RUNS):
        raise RuntimeError('Process ended but full VH_TEST/EH_ALL results are missing.')
    (work / 'COMPLETE.json').write_text(json.dumps(dict(ap=ap, an=an, model=config['model'])) + '\n')
    print(f'POINT COMPLETE {ap:g}:{an:g}; adapters/checkpoints: {work}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--old-work', type=Path)
    parser.add_argument('--env-archive', type=Path)
    parser.add_argument('--hf-cache', type=Path)
    parser.add_argument('--vh-data', type=Path)
    parser.add_argument('--eh-data', type=Path)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--point', nargs=2, type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.point:
        worker(args)
        return
    if not args.old_work or not (args.old_work / REL).is_dir():
        parser.error('--old-work must contain the legacy seed_component_ablation/runs directory')
    points = plan(args.old_work, args.root)
    for ap, an, done in points:
        state = 'REUSE' if all(done.values()) else 'SUBMIT'
        print(f'{state:6} ap={ap:g} an={an:g} Base_fix={done[RUNS[0]]} Adaptive={done[RUNS[1]]}')
    print(f'Total=36; reuse={sum(all(d.values()) for _, _, d in points)}; '
          f'pending={sum(not all(d.values()) for _, _, d in points)}', flush=True)
    if args.submit:
        if not all((args.env_archive, args.hf_cache, args.vh_data, args.eh_data)):
            parser.error('--submit requires --env-archive, --hf-cache, --vh-data and --eh-data')
        submit(Path(__file__).resolve().parents[1], args, points)


if __name__ == '__main__':
    main()
