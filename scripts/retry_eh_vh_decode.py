"""Evaluation-only recovery for an immutable EH -> VH experiment snapshot.

The decoder override is recorded separately from the original training source.
No frozen source files or training configurations are edited.
"""
import argparse
import contextlib
import csv
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_hash(root):
    base = root / 'code'
    digest = hashlib.sha256()
    for path in sorted((base / 'src').rglob('*.py')):
        digest.update(str(path.relative_to(base)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(root, ident):
    config = json.loads((root / 'experiment.json').read_text())
    if snapshot_hash(root) != config['source']:
        raise ValueError('Frozen source changed; refusing checkpoint recovery.')
    point = next((p for p in config['points'] if p['id'] == ident), None)
    if point is None or point['kind'] == 'base':
        raise ValueError('Choose one existing trained point, not base.')
    work = root / 'work' / ident
    if not list(work.rglob('*_adapter.pt')):
        raise ValueError(f'No saved adapter in {work}; recovery never trains.')
    for name in ('EventHallusion', 'VideoHallucer'):
        if not (work / name).is_dir():
            raise ValueError(f'Missing dataset: {work / name}')
    return config, point, work


def check_inactive(root, ident):
    active = subprocess.check_output(['qstat', '-u', os.environ['USER']], text=True)
    ids = set(re.findall(r'^\s*(\d+)\.', active, re.MULTILINE))
    ids.discard(os.environ.get('PBS_JOBID', '').split('.')[0])
    journal = root / 'jobs.tsv'
    if journal.exists():
        with journal.open(newline='') as stream:
            for row in csv.reader(stream, delimiter='\t'):
                if len(row) >= 2 and row[1] == ident and row[0].split('.')[0] in ids:
                    raise ValueError(f'Point already has an active job: {row[0]}')


@contextlib.contextmanager
def recovery_hooks(video, trainer, point, audit, report=lambda: None):
    """Inject decoder settings and forbid falling through to training."""
    old_video_run, old_train = video.run, trainer.train_arm
    fallback_path = Path(__file__).with_name('decode_eof_fallback.py')
    spec = importlib.util.spec_from_file_location('retry_eof_fallback', fallback_path)
    fallback = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fallback)

    def video_run(runtime):
        old_video_run(runtime)
        runtime.VideoReader = fallback.wrap_reader(runtime.VideoReader, audit, report)
        print('DECODER RECOVERY: num_threads=1 + sequential EOF fallback; frame indices unchanged', flush=True)

    def evaluation_only(runtime, name, seed):
        if f'{name}_s{seed}' != point['run']:
            raise ValueError('Unexpected adapter requested during recovery.')
        _, adapter, metadata = runtime.arm_paths(name, seed)
        adapter, metadata = Path(adapter), Path(metadata)
        if not adapter.is_file() or not metadata.is_file():
            raise ValueError(f'Exact adapter/config missing: {adapter}; refusing training.')
        expected = json.loads(json.dumps(runtime.student_cfg(name, seed), default=str))
        if json.loads(metadata.read_text()) != expected:
            raise ValueError('Saved adapter configuration does not match this run.')
        before = sha(adapter)
        audit['adapter'] = str(adapter)
        audit['adapter_sha256'] = before
        print(f'EVALUATION ONLY: reuse {adapter}', flush=True)
        try:
            output = old_train(runtime, name, seed)
        finally:
            if sha(adapter) != before:
                raise RuntimeError('Saved adapter changed during evaluation recovery.')
        audit['adapter_verified_unchanged'] = True
        return output

    video.run, trainer.train_arm = video_run, evaluation_only
    try:
        yield
    finally:
        video.run, trainer.train_arm = old_video_run, old_train


def execute(root, ident):
    import fcntl
    config, point, work = prepare(root, ident)
    check_inactive(root, ident)
    with (work / '.decode_recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = root / 'code'
        for name, module in list(sys.modules.items()):
            if name == 'seed_video' or name.startswith('seed_video.'):
                location = getattr(module, '__file__', None)
                if location and not Path(location).resolve().is_relative_to(frozen / 'src'):
                    raise ValueError('Run recovery in a fresh interpreter using the frozen source.')
        sys.path.insert(0, str(frozen / 'src'))
        spec = importlib.util.spec_from_file_location('frozen_eh_vh_driver', frozen / 'scripts/eh_vh_experiment.py')
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        if driver.complete(root, point):
            print(f'SKIP COMPLETE {ident}', flush=True)
            return
        os.environ.update(
            SEED_MODEL_ID=config['model'],
            SEED_PROTOCOL_DIR=config.get('manifest_dir', str(root / 'manifests')),
            SEED_FOREGROUND_CACHE=str(root / 'shared/foreground'),
            SEED_BASE_CACHE_DIR=str(root / 'shared/base'),
            SEED_TEACHER_CACHE_DIR=str(root / 'shared/teachers'),
        )
        audit = dict(point=ident, training_source=config['source'],
                     recovery_script_sha256=sha(__file__), num_threads=1,
                     eof_fallback='opencv_sequential',
                     fallback_script_sha256=sha(Path(__file__).with_name('decode_eof_fallback.py')),
                     eof_retry_max=os.environ.get('DECORD_EOF_RETRY_MAX', '10240'),
                     reuse_partial_evaluation=True, training_allowed=False,
                     job=os.environ.get('PBS_JOBID'), state='STARTED')
        audit_path = work / 'decoder_recovery' / f'{time.time_ns()}.json'
        driver.atomic_json(audit_path, audit)
        try:
            video = importlib.import_module('seed_video.data.video')
            trainer = importlib.import_module('seed_video.training.trainer')
            cli = importlib.import_module('seed_video.cli')
            with recovery_hooks(video, trainer, point, audit, lambda: driver.atomic_json(audit_path, audit)):
                cli.main(driver.command(config, point, work)[4:])
            if not audit.get('adapter_verified_unchanged') or not driver.results_ready(work, point):
                raise RuntimeError('Recovery did not produce all expected method results.')
            audit['state'] = 'COMPLETE'
            driver.atomic_json(work / 'COMPLETE.json', dict(
                point=point, adapters=[str(Path(audit['adapter']).relative_to(work))],
                source=config['source'], decoder_recovery=str(audit_path.relative_to(work))))
            print(f'EXPERIMENT POINT COMPLETE {ident}; decoder recovery', flush=True)
        except BaseException as error:
            audit.update(state='FAILED', error=str(error))
            raise
        finally:
            driver.atomic_json(audit_path, audit)


def submit(root, ident):
    import fcntl
    with (root.parent / f'.{root.name}.submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_inactive(root, ident)
        script = Path(__file__).resolve()
        pbs = script.with_suffix('.pbs')
        fallback = script.with_name('decode_eof_fallback.py')
        version = hashlib.sha256(script.read_bytes() + pbs.read_bytes() + fallback.read_bytes()).hexdigest()[:16]
        tools = root / 'recovery_tools' / version
        tools.mkdir(parents=True, exist_ok=True)
        for path in (script, pbs, fallback):
            destination = tools / path.name
            if not destination.exists():
                shutil.copy2(path, destination)
            if destination.read_bytes() != path.read_bytes():
                raise ValueError('Recovery tool snapshot mismatch.')
        label = f'{ident}_decode1_{time.time_ns()}'
        log = root / 'logs' / f'{label}.out'
        job = subprocess.check_output([
            'qsub', '-P', 'hn98', '-q', 'gpuhopper',
            '-l', 'ncpus=12,ngpus=1,mem=160GB,jobfs=100GB,walltime=01:30:00',
            '-l', 'storage=scratch/hn98+scratch/jp09+gdata/hn98',
            '-v', f'EXPERIMENT_ROOT={root},EXPERIMENT_POINT={ident},RETRY_SCRIPT={tools / script.name}',
            '-N', 'ehvh_decode_retry', '-j', 'oe', '-o', str(log), str(tools / pbs.name),
        ], text=True).strip()
        with (root / 'jobs.tsv').open('a') as stream:
            stream.write(f'{job}\t{ident}\t1\t{label}\n')
        print(f'SUBMITTED {ident} -> {job}\nLOG={log}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--point', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--submit', action='store_true')
    mode.add_argument('--execute', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if any(c.isspace() for c in str(root)) or ',' in str(root):
        parser.error('Root must not contain whitespace or commas.')
    _, _, work = prepare(root, args.point)
    print(f'RECOVERY PLAN: {args.point}; decoder threads=1; training forbidden; work={work}')
    if args.execute:
        execute(root, args.point)
    elif args.submit:
        submit(root, args.point)


if __name__ == '__main__':
    main()
