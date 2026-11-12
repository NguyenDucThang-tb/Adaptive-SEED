"""Enable an audited single-thread decoder for already queued EH/VH workers.

PBS keeps its existing job IDs and dependencies. Only the disk Python
entrypoint is wrapped atomically; frozen training source is never edited.
"""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


MARKER = '# EH_VH_QUEUED_DECODE_V1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted((root / 'code/src').rglob('*.py')):
        digest.update(str(path.relative_to(root / 'code')).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + '\n')


def shim_text(manifest):
    return f'''{MARKER}
import importlib.util
from pathlib import Path

_manifest = Path({str(manifest)!r})
_spec = importlib.util.spec_from_file_location("queued_decode_runtime", _manifest.parent / "queued_eh_vh_decode.py")
_runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runtime)
_driver = _runtime.load_driver(_manifest)

def __getattr__(name):
    return getattr(_driver, name)

if __name__ == "__main__":
    _runtime.dispatch(_driver, _manifest)
'''


def verify(manifest):
    data = json.loads(manifest.read_text())
    root = Path(data['root'])
    if source_hash(root) != data['training_source']:
        raise ValueError('Frozen training source changed; refusing queued override.')
    for name in data['files']:
        if sha(manifest.parent / name) != data['files'][name]:
            raise ValueError(f'Override tool checksum mismatch: {name}')
    return data


def load_driver(manifest):
    verify(manifest)
    spec = importlib.util.spec_from_file_location('original_eh_vh_driver', manifest.parent / 'original_driver.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


def install(root, apply=False):
    config = json.loads((root / 'experiment.json').read_text())
    if source_hash(root) != config['source']:
        raise ValueError('Frozen source does not match experiment.json; not patching.')
    entrypoint = root / 'code/scripts/eh_vh_experiment.py'
    entry_content = entrypoint.read_text()
    content = entry_content
    fallback = Path(__file__).with_name('decode_eof_fallback.py')
    if content.startswith(MARKER):
        # Verify and unwrap the previous installation, never stack shims.
        tree = ast.parse(content)
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == '_manifest' for t in node.targets))
        manifest = Path(ast.literal_eval(assignment.value.args[0]))
        data = verify(manifest)
        if Path(data['root']) != root or content != shim_text(manifest):
            raise ValueError('Existing queued override is not the expected shim.')
        if (data['files']['queued_eh_vh_decode.py'] == sha(__file__)
                and data['files'].get('decode_eof_fallback.py') == sha(fallback)):
            print('ALREADY ENABLED: queued decoder override with EOF fallback; no changes made.')
            return
        content = (manifest.parent / 'original_driver.py').read_text()
        print('UPGRADE: preserve previous tools; switch only future worker starts.')
    if 'subprocess.run(command(config, point, work), check=True)' not in content:
        raise ValueError('Unrecognized frozen worker launcher; refusing automatic patch.')
    print(f'QUEUED PATCH PLAN: {entrypoint}; num_threads=1 + sequential EOF fallback; no qdel/qsub/qalter')
    if not apply:
        return
    import fcntl
    with (root / '.queued_decode_install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if entrypoint.read_text() != entry_content:
            raise ValueError('Entrypoint changed while planning; run again.')
        version = hashlib.sha256(Path(__file__).read_bytes() + fallback.read_bytes() + content.encode()).hexdigest()[:16]
        directory = root / 'recovery_tools' / f'queued_decode_{version}'
        directory.mkdir(parents=True, exist_ok=True)
        files = {'original_driver.py': content,
                 'queued_eh_vh_decode.py': Path(__file__).read_text(),
                 'decode_eof_fallback.py': fallback.read_text()}
        for name, text in files.items():
            path = directory / name
            if path.exists() and path.read_text() != text:
                raise ValueError(f'Existing override tool differs: {path}')
            atomic_write(path, text)
        manifest = directory / 'manifest.json'
        atomic_json(manifest, dict(root=str(root), training_source=config['source'],
                                  decoder_num_threads=1, installed_at=time.time(),
                                  eof_fallback='opencv_sequential',
                                  files={name: sha(directory / name) for name in files}))
        verify(manifest)
        atomic_write(entrypoint, shim_text(manifest))
        print(f'ENABLED: future worker starts use num_threads=1 + sequential EOF fallback; manifest={manifest}')
        print('Already started Python workers are unchanged. Existing job IDs/dependencies remain intact.')


def dispatch(driver, manifest):
    """Only replace the child CLI command for worker invocations."""
    if '--worker' not in sys.argv:
        driver.main()
        return
    ident = sys.argv[sys.argv.index('--worker') + 1]
    data = verify(manifest)
    if '--root' not in sys.argv or Path(sys.argv[sys.argv.index('--root') + 1]).resolve() != Path(data['root']):
        raise ValueError('Queued worker root differs from override manifest.')
    original_run = driver.subprocess.run

    def run(command, *args, **kwargs):
        if command[1:4] != ['-u', '-m', 'seed_video']:
            raise ValueError('Unexpected frozen worker subprocess; refusing override.')
        command = [command[0], '-u', str(manifest.parent / 'queued_eh_vh_decode.py'),
                   '--root', data['root'], '--worker', ident, '--manifest', str(manifest),
                   '--cli', *command[4:]]
        return original_run(command, *args, **kwargs)

    driver.subprocess.run = run
    try:
        driver.main()
        work = Path(data['root']) / 'work' / ident
        marker = work / 'COMPLETE.json'
        audit = work / 'decoder_override' / f'{os.environ.get("PBS_JOBID", "unknown")}.json'
        if marker.is_file() and audit.is_file():
            result = json.loads(marker.read_text())
            result['decoder_override'] = str(audit.relative_to(work))
            atomic_json(marker, result)
    finally:
        driver.subprocess.run = original_run


def worker(root, ident, manifest, cli_args):
    data = verify(manifest)
    if Path(data['root']) != root:
        raise ValueError('Experiment root mismatch.')
    config = json.loads((root / 'experiment.json').read_text())
    if not any(p['id'] == ident for p in config['points']):
        raise ValueError('Unknown experiment point.')
    sys.path.insert(0, str(root / 'code/src'))
    video = importlib.import_module('seed_video.data.video')
    cli = importlib.import_module('seed_video.cli')
    previous = video.run
    fallback_spec = importlib.util.spec_from_file_location('queued_eof_fallback', manifest.parent / 'decode_eof_fallback.py')
    fallback = importlib.util.module_from_spec(fallback_spec)
    fallback_spec.loader.exec_module(fallback)

    def video_run(runtime):
        previous(runtime)
        runtime.VideoReader = fallback.wrap_reader(runtime.VideoReader, audit, lambda: atomic_json(path, audit))
        print('QUEUED DECODER OVERRIDE: num_threads=1 + sequential EOF fallback; original frame indices and splits', flush=True)

    video.run = video_run
    path = root / 'work' / ident / 'decoder_override' / f'{os.environ.get("PBS_JOBID", "unknown")}.json'
    audit = dict(point=ident, training_source=config['source'], num_threads=1,
                 eof_fallback='opencv_sequential',
                 override_manifest=str(manifest), training_allowed=True,
                 reuse_existing_caches=True, job=os.environ.get('PBS_JOBID'), state='STARTED')
    atomic_json(path, audit)
    try:
        cli.main(cli_args)
        audit['state'] = 'CLI_COMPLETE'
    except BaseException as error:
        audit.update(state='FAILED', error=str(error))
        raise
    finally:
        video.run = previous
        atomic_json(path, audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--manifest', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--cli', nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker:
        if not args.manifest or not args.cli:
            parser.error('Worker requires manifest and original CLI arguments.')
        worker(root, args.worker, args.manifest, args.cli)
    else:
        install(root, args.apply)


if __name__ == '__main__':
    main()
