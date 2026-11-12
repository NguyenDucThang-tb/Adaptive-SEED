"""Plan, submit, resume and collect the explicit EH -> five-task VH experiment.

Uses only the Python standard library outside GPU workers. Default is a dry plan.
"""
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
import time

MODELS = {'qwen25': 'Qwen/Qwen2.5-VL-7B-Instruct',
          'qwen3': 'Qwen/Qwen3-VL-8B-Instruct',
          'internvl35': 'OpenGVLab/InternVL3_5-8B-Instruct'}
TASKS = {'ORH': 'object_relation', 'TPH': 'temporal', 'SDH': 'semantic_detail',
         'EFH': 'external_factual', 'ENFH': 'external_nonfactual'}
REVISION = 'c93475bf1be466c1af90d005749ccb4e6b3ad066'
VALUES = (0, 0.25, 0.5, 0.75, 0.9, 1)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def entry(kind, ap, an, group=None):
    ident = f'{kind}_ap{ap:g}_an{an:g}'.replace('.', 'p')
    run = ('base' if kind == 'base' else f'{kind}@T_pos_tcd_s0'
           if kind in ('kd_basic', 'onpolicy_gkd', 'video_opd') else
           {'pos': 'adaptive_pos_only_s0', 'neg': 'adaptive_neg_only_s0',
            'both': 'adaptive_pos_neg_s0'}[kind])
    stage = 'distillation_methods' if kind in ('base', 'kd_basic', 'onpolicy_gkd', 'video_opd') else 'seed_component_ablation'
    return dict(id=ident, kind=kind, ap=ap, an=an, run=run, group=group or kind,
                result=f'cdt_distill_cache_f8_448/{stage}/runs/ap{ap:g}_an{an:g}_tf32/results.csv')


def entries(sweep):
    points = [entry(kind, 0.9, 0.25) for kind in ('base', 'kd_basic', 'onpolicy_gkd', 'video_opd')]
    if not sweep:
        return points + [entry(kind, 0.9, 0.25) for kind in ('pos', 'neg', 'both')]
    points += [entry('pos', v, 0.25) for v in VALUES[1:]]
    points += [entry('neg', 0.9, v) for v in VALUES[1:]]
    points += [entry('both', ap, an, f'both_ap{ap:g}') for ap in VALUES for an in VALUES]
    return points


def read_results(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def complete(root, point):
    work = Path(root) / 'work' / point['id']
    marker = work / 'COMPLETE.json'
    return marker.exists() and results_ready(work, point)


def results_ready(work, point):
    rows = [r for r in read_results(work / point['result']) if r['run'] == point['run']]
    required = ['EH_VAL', 'EH_TEST', 'eh_entire_TEST', 'eh_misleading_TEST', 'eh_mix_TEST',
                'VH_TEST', *[f'VH_{t}_TEST' for t in TASKS]]
    for split in required:
        matches = [r for r in rows if r['split'] == split]
        if len(matches) != 1:
            return False
        try:
            row = matches[0]
            if int(row['n']) <= 0:
                return False
            metrics = ['q_acc', 'pair_acc'] if split.startswith('VH') else ['q_acc']
            if any(not math.isfinite(float(row[m])) or not 0 <= float(row[m]) <= 1 for m in metrics):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    by_split = {r['split']: r for r in rows}
    if sum(int(by_split[f'VH_{t}_TEST']['n']) for t in TASKS) != int(by_split['VH_TEST']['n']):
        return False
    if sum(int(by_split[f'eh_{t}_TEST']['n']) for t in ('entire', 'misleading', 'mix')) != int(by_split['EH_TEST']['n']):
        return False
    return True


def preflight(args):
    errors, hashes = [], {}
    for subset in TASKS.values():
        annotation = args.vh_data / subset / f'{subset}.json'
        if not annotation.is_file():
            errors.append(f'Missing VH task: {annotation}')
            continue
        hashes[f'VH/{subset}'] = sha(annotation)
        examples = json.loads(annotation.read_text())
        if not examples:
            errors.append(f'Empty task: {annotation}')
        missing = set()
        for ex in examples:
            for role in ('basic', 'hallucination'):
                q = ex[role]
                video = args.vh_data / subset / 'videos' / Path(q['video']).name
                if not video.is_file():
                    missing.add(str(video))
        if missing:
            errors.append(f'{subset}: {len(missing)} missing videos; example: {sorted(missing)[0]}')
    for split in ('entire', 'mix', 'misleading'):
        annotation = args.eh_data / 'questions' / f'{split}_questions.json'
        if annotation.is_file():
            hashes[f'EH/{split}'] = sha(annotation)
        else:
            errors.append(f'Missing EH annotations: {annotation}')
        folder = args.eh_data / 'videos' / ('interleave' if split == 'mix' else split)
        if not folder.is_dir() or not any(folder.iterdir()):
            errors.append(f'Missing EH videos: {folder}')
    if not args.env_archive.is_file():
        errors.append(f'Missing environment archive: {args.env_archive}')
    for model in (MODELS[args.model], 'ZhengPeng7/BiRefNet'):
        directory = args.hf_cache / ('models--' + model.replace('/', '--')) / 'snapshots'
        snapshots = [p.parent for p in directory.glob('*/config.json') if p.is_file()]
        if not snapshots:
            errors.append(f'Missing model snapshot: {directory}')
        for snapshot in snapshots:
            hashes[f'model/{model}/{snapshot.name}/config'] = sha(snapshot / 'config.json')
            index = snapshot / 'model.safetensors.index.json'
            weights = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
            for weight in weights:
                if not (snapshot / weight).is_file():
                    errors.append(f'Missing weights: {snapshot / weight}')
                else:
                    file = snapshot / weight
                    stat = file.stat()
                    hashes[f'model/{model}/{snapshot.name}/{weight}'] = (
                        f'{file.resolve().name}:{stat.st_size}:{stat.st_mtime_ns}')
        ref = directory.parent / 'refs/main'
        if ref.is_file():
            hashes[f'model/{model}/refs/main'] = ref.read_text().strip()
    if errors:
        raise ValueError('\n'.join(errors))
    return hashes


def config_for(args, points, hashes, repo):
    digest = hashlib.sha256()
    for p in sorted((repo / 'src').rglob('*.py')):
        digest.update(str(p.relative_to(repo)).encode())
        digest.update(p.read_bytes())
    return dict(version='eh_to_vh_full_v1', model_key=args.model, model=MODELS[args.model],
                source=digest.hexdigest(), epochs=args.epochs, teacher_frames=32, student_frames=8,
                vh_revision=REVISION, tasks=list(TASKS), eh_seed=42, eh_fractions=[0.7, 0.1],
                env_archive=str(args.env_archive.resolve()), env_sha256=sha(args.env_archive),
                hf_cache=str(args.hf_cache.resolve()), eh_data=str(args.eh_data.resolve()),
                vh_data=str(args.vh_data.resolve()), input_fingerprints=hashes, points=points)


def submit(args, config, repo):
    root = args.root
    saved = root / 'experiment.json'
    if saved.exists():
        if json.loads(saved.read_text()) != config:
            raise ValueError('Experiment/code/data/env changed. Use a new root; do not mix checkpoints.')
    elif root.exists() and any(root.iterdir()):
        raise ValueError('New experiment root must be empty.')
    journal = root / 'jobs.tsv'
    if journal.exists():
        active = subprocess.check_output(['qstat', '-u', os.environ['USER']], text=True)
        active_ids = set(re.findall(r'^\s*(\d+)\.', active, re.MULTILINE))
        previous = {line.split('\t')[0].split('.')[0] for line in journal.read_text().splitlines()}
        if previous & active_ids:
            raise ValueError('This experiment still has active jobs; do not submit duplicate workers.')
    if not saved.exists():
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(repo / 'src', root / 'code/src', ignore=shutil.ignore_patterns('__pycache__'))
        (root / 'code/scripts').mkdir()
        for name in ('eh_vh_experiment.py', 'run_eh_vh_experiment.pbs'):
            shutil.copy2(repo / 'scripts' / name, root / 'code/scripts' / name)
        atomic_json(saved, config)
    (root / 'logs').mkdir(exist_ok=True)
    parents = {}
    stamp = time.strftime('%Y%m%dT%H%M%S')
    for point in config['points']:
        if complete(root, point):
            continue
        work = root / 'work' / point['id']
        work.mkdir(parents=True, exist_ok=True)
        for name, source in (('EventHallusion', args.eh_data), ('VideoHallucer', args.vh_data)):
            link = work / name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(source.resolve(), target_is_directory=True)
            if link.resolve() != source.resolve():
                raise ValueError(f'Unexpected dataset link: {link}')
        previous = parents.get(point['group'])
        for segment in range(args.segments):
            label = f'{point["id"]}_{stamp}_{segment + 1}'
            cmd = ['qsub', '-P', 'hn98', '-q', 'gpuhopper',
                   '-l', 'ncpus=12,ngpus=1,mem=160GB,jobfs=100GB,walltime=01:00:00',
                   '-l', 'storage=scratch/hn98+scratch/jp09+gdata/hn98',
                   '-v', f'EXPERIMENT_ROOT={root},EXPERIMENT_POINT={point["id"]}',
                   '-N', f'ehvh_{args.model}_{point["kind"]}', '-j', 'oe',
                   '-o', str(root / 'logs' / f'{label}.out')]
            if previous:
                cmd += ['-W', f'depend=afterany:{previous}']
            cmd += [str(root / 'code/scripts/run_eh_vh_experiment.pbs')]
            previous = subprocess.check_output(cmd, text=True).strip()
            with journal.open('a') as stream:
                stream.write(f'{previous}\t{point["id"]}\t{segment + 1}\t{label}\n')
            print(f'SUBMITTED {point["id"]} segment={segment + 1} -> {previous}', flush=True)
        parents[point['group']] = previous


def command(config, point, work):
    cmd = [sys.executable, '-u', '-m', 'seed_video', '--stage', 'core', '--work-dir', str(work),
           '--train-dataset', 'eventhallusion', '--eval-videohallucer', '--vh-eval-tasks', *TASKS,
           '--eh-train-fraction', '0.7', '--eh-val-fraction', '0.1', '--eh-split-seed', '42',
           '--teacher-frames', '32', '--epochs', str(config['epochs']),
           '--a-pos', str(point['ap']), '--a-neg', str(point['an'])]
    if point['kind'] == 'base':
        cmd += ['--base-only']
    elif point['kind'] in ('kd_basic', 'onpolicy_gkd', 'video_opd'):
        cmd += ['--distill-methods', point['kind']]
    else:
        cmd += ['--arms', point['run'].removesuffix('_s0')]
    return cmd


def worker(root, ident):
    config = json.loads((root / 'experiment.json').read_text())
    point = next(p for p in config['points'] if p['id'] == ident)
    if complete(root, point):
        print(f'SKIP COMPLETE {ident}', flush=True)
        return
    work = root / 'work' / ident
    os.environ.update(SEED_MODEL_ID=config['model'], SEED_PROTOCOL_DIR=config.get('manifest_dir', str(root / 'manifests')),
                      SEED_FOREGROUND_CACHE=str(root / 'shared/foreground'),
                      SEED_BASE_CACHE_DIR=str(root / 'shared/base'),
                      SEED_TEACHER_CACHE_DIR=str(root / 'shared/teachers'))
    print(f'START {ident}; model={config["model"]}; checkpoint root={work}', flush=True)
    subprocess.run(command(config, point, work), check=True)
    if not results_ready(work, point):
        raise RuntimeError('Process exited without all five VH tasks and held-out EH results.')
    adapters = sorted(str(p.relative_to(work)) for p in work.rglob('*_adapter.pt'))
    if point['kind'] != 'base' and not adapters:
        raise RuntimeError('Adapter missing after training; not marking complete.')
    atomic_json(work / 'COMPLETE.json', dict(point=point, adapters=adapters, source=config['source']))
    print(f'EXPERIMENT POINT COMPLETE {ident}; adapters={adapters}', flush=True)


def collect(root):
    config = json.loads((root / 'experiment.json').read_text())
    base_point = next(p for p in config['points'] if p['kind'] == 'base')
    base_rows = {r['split']: r for r in read_results(root / 'work' / base_point['id'] / base_point['result']) if r['run'] == 'base'}
    all_rows, candidates = [], {}
    for point in config['points']:
        work = root / 'work' / point['id']
        ready = complete(root, point)
        rows = [r for r in read_results(work / point['result']) if r['run'] == point['run']]
        if not rows:
            all_rows.append(dict(model=config['model'], method=point['kind'], ap=point['ap'],
                                 an=point['an'], status='MISSING', split=''))
        for row in rows:
            record = dict(model=config['model'], method=point['kind'], ap=point['ap'], an=point['an'],
                          status='RESULTS_READY' if ready else 'PARTIAL', **row)
            reference = base_rows.get(row['split'])
            if reference and reference['n'] == row['n']:
                record['main_base_q_acc'] = reference['q_acc']
                record['delta_main_q_pp'] = 100 * (float(row['q_acc']) - float(reference['q_acc']))
                if row['split'].startswith('VH'):
                    record['main_base_pair_acc'] = reference['pair_acc']
                    record['delta_main_pair_pp'] = 100 * (float(row['pair_acc']) - float(reference['pair_acc']))
            all_rows.append(record)
        if ready:
            val = next(r for r in rows if r['split'] == 'EH_VAL')
            candidates.setdefault(point['kind'], []).append((float(val['q_acc']), -point['ap'], -point['an'], point))
    for filename, records in (
        ('all_results.csv', all_rows),
        ('selected_by_eh_val.csv', [dict(method=k, eh_val_acc=max(v, key=lambda x: x[:3])[0],
                                        selection_complete=len(v) == sum(p['kind'] == k for p in config['points']),
                                        **max(v, key=lambda x: x[:3])[3]) for k, v in candidates.items()]),
    ):
        keys = list(dict.fromkeys(key for row in records for key in row))
        with (root / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
        print('Saved:', root / filename)
    for p in config['points']:
        print('RESULTS_READY' if complete(root, p) else 'INCOMPLETE', p['id'])
    print('Selection uses EH_VAL only. Verify PBS Exit Status in logs; RESULTS_READY is not scheduler status.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', choices=MODELS, default='qwen25')
    parser.add_argument('--env-archive', type=Path)
    parser.add_argument('--hf-cache', type=Path)
    parser.add_argument('--eh-data', type=Path)
    parser.add_argument('--vh-data', type=Path)
    parser.add_argument('--manifest-dir', type=Path,
                        help='Optional shared immutable split manifests across model roots.')
    parser.add_argument('--sweep', action='store_true', help='50 configurations: Base, 3 KD, 5 Pos, 5 Neg, 36 Both.')
    parser.add_argument('--segments', type=int, default=1, help='Finite one-hour resume segments per configuration.')
    parser.add_argument('--epochs', type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--submit', action='store_true')
    mode.add_argument('--collect', action='store_true')
    mode.add_argument('--worker', metavar='POINT', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.worker:
        worker(args.root, args.worker)
        return
    if args.collect:
        collect(args.root)
        return
    if args.segments < 1 or args.epochs < 1:
        parser.error('--segments and --epochs must be positive')
    if not all((args.env_archive, args.hf_cache, args.eh_data, args.vh_data)):
        parser.error('Required: --env-archive --hf-cache --eh-data --vh-data')
    if ',' in str(args.root) or any(c.isspace() for c in str(args.root)):
        parser.error('PBS experiment root must not contain commas or whitespace')
    hashes = preflight(args)
    points = entries(args.sweep)
    for point in points:
        print('REUSE' if complete(args.root, point) else 'PENDING', point['id'])
    pending = sum(not complete(args.root, p) for p in points)
    print(f'Model={MODELS[args.model]}; tasks={list(TASKS)}; epochs={args.epochs}; '
          f'configurations={len(points)}; pending={pending}; one-hour jobs={pending * args.segments}')
    if args.submit:
        config = config_for(args, points, hashes, Path(__file__).resolve().parents[1])
        config['manifest_dir'] = str((args.manifest_dir or args.root / 'manifests').resolve())
        submit(args, config, Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    main()
