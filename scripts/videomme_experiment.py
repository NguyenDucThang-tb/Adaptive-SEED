"""Plan (default) or submit a frozen, fixed-budget source -> VideoMME experiment."""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

MODELS = {'qwen25': 'Qwen/Qwen2.5-VL-7B-Instruct',
          'qwen3': 'Qwen/Qwen3-VL-8B-Instruct',
          'internvl35': 'OpenGVLab/InternVL3_5-8B-Instruct'}
DATASETS = ('videohallucer', 'eventhallusion', 'mixed')
TOOLS = ('videomme_experiment.py', 'videomme_train.py', 'run_videomme_experiment.pbs',
         'eval_videomme.py', 'decode_eof_fallback.py')
REPO = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_files(repo):
    return sorted((repo / 'src').rglob('*.py')) + [repo / 'scripts' / n for n in TOOLS]


def code_hashes(repo):
    return {str(p.relative_to(repo)): sha(p) for p in code_files(repo)}


def script_path(root, name):
    frozen = root / 'code/scripts' / name
    return frozen if frozen.is_file() else Path(__file__).with_name(name)


def jobs(datasets):
    return [dict(id='eval_base', kind='eval', dataset='base', parent=None)] + [
        point for dataset in datasets for point in (
            dict(id=f'train_{dataset}', kind='train', dataset=dataset, parent=None),
            dict(id=f'eval_{dataset}', kind='eval', dataset=dataset, parent=f'train_{dataset}'))]


def preflight(args, repo=REPO):
    """Resolve all benchmark videos using the evaluator's own stdlib loader."""
    fingerprints = {}
    if not args.env_archive.is_file() or args.env_archive.stat().st_size == 0:
        raise ValueError(f'Missing/nonempty environment archive required: {args.env_archive}')
    loader = load_module(repo / 'src/seed_video/data/videomme.py', '_mme_preflight')
    items = loader.load_videomme(args.data_root)
    if not items:
        raise ValueError('Empty VideoMME dataset')
    fingerprints['VideoMME/questions.json'] = sha(args.data_root / 'questions.json')
    videos = {Path(i['video']).resolve() for i in items}
    # The core VH pipeline also evaluates EH; keep both inputs available per work.
    subsets = ('temporal',) if args.smoke_pairs else ('temporal', 'object_relation')
    if any(d in args.datasets for d in ('videohallucer', 'mixed')):
        for subset in subsets:
            annotations = sorted((args.vh_data / subset).glob('*.json'))
            if not annotations:
                raise ValueError(f'Missing VH annotations: {args.vh_data / subset}')
            index = {p.name: p for p in (args.vh_data / subset).rglob('*') if p.is_file()}
            for annotation in annotations:
                fingerprints[str(annotation.resolve())] = sha(annotation)
                for row in json.loads(annotation.read_text()):
                    for role in ('basic', 'hallucination'):
                        name = Path(row[role]['video']).name
                        if name not in index:
                            raise ValueError(f'Missing VH video: {subset}/{name}')
                        videos.add(index[name].resolve())
    for subset, folders in (('entire', ('entire',)), ('mix', ('interleave', 'mix')),
                            ('misleading', ('misleading',))):
        annotation = args.eh_data / 'questions' / f'{subset}_questions.json'
        fingerprints[str(annotation.resolve())] = sha(annotation)
        records = json.loads(annotation.read_text())
        if not records:
            raise ValueError(f'Empty EH annotations: {annotation}')
        index = {folder: {p.stem: p for p in args.eh_data.rglob('*')
                          if p.is_file() and p.parent.name.lower() == folder
                          and p.suffix.lower() in ('.mp4', '.mkv', '.webm', '.avi', '.mov')}
                 for folder in folders}
        for row in records:
            ident = row['id']
            prefix, _, number = ident.partition('_')
            n = str(int(number)) if number.isdigit() else number
            candidates = [ident, f'{prefix}_{n}', f'{prefix}{n}', number, n]
            if subset == 'mix':
                candidates += [f'interleave_{number}', f'interleave_{n}', f'interleave{n}']
            match = next((index[f][c] for f in folders for c in candidates if c in index[f]), None)
            if match is None:
                raise ValueError(f'Missing EH video: {subset}/{ident}')
            videos.add(match.resolve())
    fingerprints['videos'] = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(videos)}
    for model in (MODELS[args.model], 'ZhengPeng7/BiRefNet'):
        directory = args.hf_cache / ('models--' + model.replace('/', '--'))
        ref = directory / 'refs/main'
        if not ref.is_file():
            raise ValueError(f'Missing offline model ref: {ref}')
        snapshot = directory / 'snapshots' / ref.read_text().strip()
        fingerprints[str(ref)] = sha(ref)
        fingerprints[str(snapshot / 'config.json')] = sha(snapshot / 'config.json')
        index = snapshot / 'model.safetensors.index.json'
        weights = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
        if index.exists():
            fingerprints[str(index)] = sha(index)
        for weight in weights:
            p = snapshot / weight
            if not p.is_file() or not p.stat().st_size:
                raise ValueError(f'Missing model weights: {p}')
            fingerprints[str(p)] = [str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns]
    return fingerprints


def config_for(args, repo=REPO):
    return dict(version='videomme_experiment2_v1', model_key=args.model, model=MODELS[args.model],
                datasets=args.datasets, ap=args.ap, an=args.an, max_train_steps=args.max_train_steps,
                epochs=args.epochs, teacher_frames=32, n_frames=8, frame_size=448,
                epsilon_pp=args.epsilon_pp,
                smoke=args.smoke_pairs is not None or args.limit is not None,
                smoke_pairs=args.smoke_pairs, limit=args.limit,
                coefficient_selection='predeclared_or_source_validation_only_not_VideoMME',
                env_archive=str(args.env_archive), env_sha256=sha(args.env_archive),
                hf_cache=str(args.hf_cache), data_root=str(args.data_root),
                vh_data=str(args.vh_data), eh_data=str(args.eh_data),
                input_fingerprints=preflight(args, repo), code=code_hashes(repo), jobs=jobs(args.datasets))


def submitted_jobs(root):
    journal = root / 'jobs.tsv'
    accepted = {}
    if journal.exists():
        with journal.open(newline='') as stream:
            for job, ident in csv.reader(stream, delimiter='\t'):
                if ident in accepted or not re.fullmatch(r'\d+(?:[.][\w.-]+)?', job):
                    raise ValueError('Invalid/duplicate jobs.tsv entry; reconcile before submission')
                accepted[ident] = job
    return accepted


def qsub_command(root, config, point, parent=None):
    if any(c in str(root) for c in ',\n\r'):
        raise ValueError('PBS experiment root cannot contain comma/newline')
    cmd = ['qsub', '-P', 'hn98', '-q', 'gpuhopper', '-l',
           'ncpus=12,ngpus=1,mem=160GB,jobfs=100GB,walltime=01:30:00',
           '-l', 'storage=gdata/hn98+scratch/hn98+scratch/jp09',
           '-v', f'EXPERIMENT_ROOT={root},EXPERIMENT_JOB={point["id"]}',
           '-N', f'mme_{config["model_key"]}_{point["id"]}', '-j', 'oe',
           '-o', str(root / 'logs' / f'{point["id"]}.out')]
    if parent:
        cmd += ['-W', f'depend=afterok:{parent}']
    return cmd + [str(script_path(root, 'run_videomme_experiment.pbs'))]


def validate_root(root, config):
    saved = root / 'experiment.json'
    if saved.exists():
        if json.loads(saved.read_text()) != config:
            raise ValueError('Immutable config/code/data/env changed; choose a new root')
        for name, digest in config['code'].items():
            if sha(root / 'code' / name) != digest:
                raise ValueError(f'Frozen code changed: {name}')
    elif root.exists() and any(root.iterdir()):
        raise ValueError('New root must be empty')


def submit(args, config, repo=REPO):
    root = args.root
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f'.{root.name}.submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_root(root, config)
        if (root / 'submission_pending.json').exists():
            raise ValueError('Ambiguous qsub attempt; reconcile submission_pending.json with PBS first')
        accepted = submitted_jobs(root)
        if accepted and not args.fill_unsubmitted:
            raise ValueError('Root already submitted; use --fill-unsubmitted (never retries accepted jobs)')
        if not (root / 'experiment.json').exists():
            shutil.copytree(repo / 'src', root / 'code/src', ignore=shutil.ignore_patterns('__pycache__'))
            (root / 'code/scripts').mkdir()
            for name in TOOLS:
                shutil.copy2(repo / 'scripts' / name, root / 'code/scripts' / name)
            for name, digest in config['code'].items():
                if sha(root / 'code' / name) != digest:
                    raise ValueError('Source changed during snapshot; use a clean new root')
            atomic_json(root / 'experiment.json', config)
        (root / 'logs').mkdir(exist_ok=True)
        for point in config['jobs']:
            if point['id'] in accepted:
                continue
            parent = accepted[point['parent']] if point['parent'] else None
            cmd = qsub_command(root, config, point, parent)
            pending = root / 'submission_pending.json'
            atomic_json(pending, dict(point=point, command=cmd))
            try:
                job = subprocess.check_output(cmd, text=True).strip()
            except subprocess.CalledProcessError:
                pending.unlink()
                print('qsub rejected job; accepted jobs retained. Resume with --fill-unsubmitted --submit.')
                return False
            if not re.fullmatch(r'\d+(?:[.][\w.-]+)?', job):
                raise ValueError('Unexpected qsub response; reconcile submission_pending.json before retry')
            with (root / 'jobs.tsv').open('a') as stream:
                stream.write(f'{job}\t{point["id"]}\n')
                stream.flush()
                os.fsync(stream.fileno())
            pending.unlink()
            accepted[point['id']] = job
            print(f'SUBMITTED {point["id"]}: {job}', flush=True)
    return True


def train_command(root, config, dataset):
    cmd = [sys.executable, '-u', '-m', 'seed_video', '--stage', 'core',
           '--work-dir', str(root / 'work' / dataset), '--train-dataset', dataset,
           '--arms', 'adaptive_pos_neg', '--max-train-steps', str(config['max_train_steps']),
           '--epochs', str(config['epochs']), '--teacher-frames', '32',
           '--a-pos', str(config['ap']), '--a-neg', str(config['an'])]
    if config['smoke_pairs'] is not None:
        cmd += ['--smoke-pairs', str(config['smoke_pairs'])]
    return cmd


def exact_adapter(root, config, dataset):
    work = root / 'work' / dataset
    matches = sorted(work.rglob('adaptive_pos_neg_s0_*_adapter.pt'))
    if len(matches) != 1:
        raise ValueError(f'Expected one exact adaptive_pos_neg s0 adapter in {work}; found {len(matches)}')
    adapter = matches[0]
    metadata = adapter.with_name(adapter.name.removesuffix('_adapter.pt') + '.json')
    saved = json.loads(metadata.read_text())
    manifest = json.loads((work / 'training_manifest.json').read_text())
    budget_path = metadata.with_name(metadata.stem + '_training_budget.json')
    budget = json.loads(budget_path.read_text())
    if (saved.get('hp', {}).get('train_dataset') != dataset or manifest['dataset'] != dataset
            or saved.get('train_dataset', dataset) != dataset):
        raise ValueError('Adapter training dataset metadata disagrees with source')
    if (budget.get('max_train_steps') != config['max_train_steps']
            or budget.get('optimizer_steps_completed') != config['max_train_steps']):
        raise ValueError('Adapter optimizer budget not completed exactly')
    cfg_hash = hashlib.md5(json.dumps(saved, sort_keys=True, default=str).encode()).hexdigest()[:10]
    if budget.get('cfg_hash') != cfg_hash:
        raise ValueError('Adapter optimizer budget cfg_hash mismatch')
    if saved['train'] != [item['id'] for item in manifest['splits']['train']]:
        raise ValueError('Adapter training-item metadata disagrees with audited manifest')
    hp = saved['hp']
    for name, expected in (('model_id', config['model']), ('max_train_steps', config['max_train_steps']),
                           ('epochs', config['epochs']), ('n_frames', 8), ('frame_size', 448)):
        if hp.get(name) != expected:
            raise ValueError(f'Adapter metadata mismatch: {name}')
    teacher = saved['teacher']
    if 'spec' in teacher:
        if teacher.get('name') != 'T_pos_tcd':
            raise ValueError('Adapter teacher is not T_pos_tcd')
        teacher = teacher['spec']
    if teacher.get('a_pos') != config['ap'] or teacher.get('negs', {}).get('tcd') != config['an']:
        raise ValueError('Adapter alpha metadata mismatch')
    return adapter.resolve()


def eval_command(root, config, dataset, adapter=None):
    cmd = [sys.executable, '-u', str(script_path(root, 'eval_videomme.py')),
           '--model', config['model_key'], '--data-root', config['data_root'],
           '--output', str(root / 'evaluation' / dataset), '--n-frames', '8', '--frame-size', '448']
    if adapter is not None:
        cmd += ['--adapter', str(adapter)]
    if config['limit'] is not None:
        cmd += ['--limit', str(config['limit'])]
    return cmd


def worker(root, ident):
    config = json.loads((root / 'experiment.json').read_text())
    validate_root(root, config)
    point = next(p for p in config['jobs'] if p['id'] == ident)
    os.environ.update(SEED_MODEL_ID=config['model'], HF_HUB_CACHE=config['hf_cache'],
                      HF_HOME=str(Path(config['hf_cache']).parent if Path(config['hf_cache']).name == 'hub'
                                  else Path(config['hf_cache'])), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      SEED_CHECKPOINT_EVERY_STEPS='1', SEED_EVAL_CHECKPOINT_EVERY='5')
    dataset = point['dataset']
    if point['kind'] == 'train':
        work = root / 'work' / dataset
        work.mkdir(parents=True, exist_ok=True)
        for name, source in (('VideoHallucer', config['vh_data']), ('EventHallusion', config['eh_data'])):
            link = work / name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(source, target_is_directory=True)
            if link.resolve() != Path(source).resolve():
                raise ValueError(f'Dataset link mismatch: {link}')
        wrapper = script_path(root, 'videomme_train.py')
        cmd = [sys.executable, '-u', str(wrapper), '--root', str(root), '--dataset', dataset]
    else:
        adapter = None
        if dataset != 'base':
            marker = json.loads((root / 'work' / dataset / 'TRAIN_COMPLETE.json').read_text())
            adapter = exact_adapter(root, config, dataset)
            if (marker['dataset'] != dataset or marker['adapter'] != str(adapter)
                    or marker['adapter_sha256'] != sha(adapter)
                    or marker['training_manifest_sha256'] != sha(root / 'work' / dataset / 'training_manifest.json')):
                raise ValueError('Adapter differs from successful source training marker')
        cmd = eval_command(root, config, dataset, adapter)
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if point['kind'] == 'eval':
        completion = root / 'evaluation' / dataset / 'COMPLETE.json'
        if not completion.is_file():
            raise RuntimeError('Evaluator exited without COMPLETE.json')
        if config['smoke']:
            # A full target evaluation of smoke-trained weights is still a smoke experiment.
            marker = json.loads(completion.read_text())
            atomic_json(completion, dict(marker, smoke=True))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', choices=MODELS, default='qwen25')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    for flag in ('env-archive', 'hf-cache', 'data-root', 'vh-data', 'eh-data'):
        p.add_argument('--' + flag, type=Path)
    p.add_argument('--ap', type=float, default=.9)
    p.add_argument('--an', type=float, default=.25)
    p.add_argument('--max-train-steps', type=int, default=50)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--epsilon-pp', type=float, default=1.0,
                   help='Predeclared retention noninferiority margin in percentage points')
    p.add_argument('--smoke-pairs', type=int)
    p.add_argument('--limit', type=int)
    p.add_argument('--submit', action='store_true')
    p.add_argument('--fill-unsubmitted', action='store_true')
    p.add_argument('--worker', choices=[j['id'] for j in jobs(DATASETS)], help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.root = args.root.expanduser().resolve()
    if args.worker:
        worker(args.root, args.worker)
        return
    for name in ('env_archive', 'hf_cache', 'data_root', 'vh_data', 'eh_data'):
        value = getattr(args, name)
        if value is None:
            p.error('--' + name.replace('_', '-') + ' is required for planning/submission')
        setattr(args, name, value.expanduser().resolve())
    for name in ('max_train_steps', 'epochs', 'smoke_pairs', 'limit'):
        value = getattr(args, name)
        if value is not None and value < 1:
            p.error('--' + name.replace('_', '-') + ' must be positive')
    if not all(0 <= x <= 1 for x in (args.ap, args.an)):
        p.error('--ap and --an must be finite and in [0, 1]')
    if not 0 <= args.epsilon_pp <= 100:
        p.error('--epsilon-pp must be finite and in [0, 100]')
    args.datasets = [d for d in DATASETS if d in args.datasets]
    smoke = args.smoke_pairs is not None or args.limit is not None
    if smoke and 'smoke' not in args.root.name.lower():
        p.error('Smoke flags require a distinct root whose name contains smoke')
    config = config_for(args)
    validate_root(args.root, config)
    accepted = submitted_jobs(args.root)
    print(f'{"SMOKE" if smoke else "FULL"} {args.model}: {len(config["jobs"])} jobs, '
          f'B={args.max_train_steps}, AP={args.ap}, AN={args.an}; dry-run={not args.submit}')
    print('Alphas predeclared or selected on source validation only; never on VideoMME.')
    for point in config['jobs']:
        print(point['id'], 'ACCEPTED ' + accepted[point['id']] if point['id'] in accepted
              else 'afterok:' + point['parent'] if point['parent'] else 'independent')
        if point['kind'] == 'train':
            print('  ' + shlex.join(train_command(args.root, config, point['dataset'])))
        else:
            print('  ' + shlex.join(eval_command(args.root, config, point['dataset'],
                  '<exact adapter from matching source work>' if point['parent'] else None)))
    if args.submit and not submit(args, config):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
