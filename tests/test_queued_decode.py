import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/queued_eh_vh_decode.py'
spec = importlib.util.spec_from_file_location('queued_decode', SCRIPT)
queued = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queued)


class QueuedDecodeTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'code/src/seed_video'
        source.mkdir(parents=True)
        (source / 'test.py').write_text('training_source = True\n')
        scripts = root / 'code/scripts'
        scripts.mkdir()
        original = scripts / 'eh_vh_experiment.py'
        original.write_text('''
import subprocess
import sys
def command(config, point, work): return [sys.executable, '-u', '-m', 'seed_video', '--stage', 'core']
def worker(config, point, work):
    subprocess.run(command(config, point, work), check=True)
def complete(root, point): return False
def main(): print('ORIGINAL MAIN')
''')
        (root / 'experiment.json').write_text(json.dumps(dict(
            source=queued.source_hash(root), points=[dict(id='point')]
        )))
        return original

    def install(self, root):
        entry = self.fixture(root)
        original = entry.read_bytes()
        before = queued.source_hash(root)
        queued.install(root, True)
        manifest = next((root / 'recovery_tools').rglob('manifest.json'))
        return entry, original, before, manifest

    def test_dry_plan_does_not_change_any_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root)
            before = entry.read_bytes()
            queued.install(root)
            self.assertEqual(entry.read_bytes(), before)
            self.assertFalse((root / 'recovery_tools').exists())

    def test_install_preserves_original_and_source_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry, original, before, manifest = self.install(root)
            self.assertEqual((manifest.parent / 'original_driver.py').read_bytes(), original)
            self.assertEqual(queued.source_hash(root), before)
            self.assertTrue(entry.read_text().startswith(queued.MARKER))
            wrapped = entry.read_bytes()
            queued.install(root, True)
            self.assertEqual(entry.read_bytes(), wrapped)
            self.assertEqual(len(list((root / 'recovery_tools').rglob('manifest.json'))), 1)

    def test_imported_shim_still_exposes_original_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry, _, _, _ = self.install(root)
            spec = importlib.util.spec_from_file_location('frozen_shim_test', entry)
            shim = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(shim)
            self.assertEqual(shim.command({}, {}, root)[1:4], ['-u', '-m', 'seed_video'])
            self.assertFalse(shim.complete(root, {}))

    def test_tampered_original_or_source_is_rejected(self):
        for change in ('original', 'source'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, _, _, manifest = self.install(root)
                if change == 'original':
                    (manifest.parent / 'original_driver.py').write_text('changed')
                else:
                    (root / 'code/src/seed_video/test.py').write_text('changed')
                with self.assertRaises(ValueError):
                    queued.verify(manifest)

    def test_dispatch_replaces_only_worker_child_and_restores_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, manifest = self.install(root)
            calls = []
            real_run = lambda cmd, **kw: calls.append((cmd, kw))
            sub = SimpleNamespace(run=real_run)
            driver = SimpleNamespace(subprocess=sub)
            driver.main = lambda: sub.run(['python', '-u', '-m', 'seed_video', '--original'], check=True)
            with patch.object(sys, 'argv', ['driver', '--root', str(root), '--worker', 'point']):
                queued.dispatch(driver, manifest)
            self.assertIs(sub.run, real_run)
            self.assertEqual(calls[0][0][-2:], ['--cli', '--original'])
            self.assertIn(str(manifest.parent / 'queued_eh_vh_decode.py'), calls[0][0])
            self.assertEqual(calls[0][1], {'check': True})

    def test_worker_override_is_audited_and_does_not_change_frame_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, before, manifest = self.install(root)
            seen = []
            video = SimpleNamespace(run=lambda r: setattr(
                r, 'VideoReader', lambda *a, **kw: seen.append((a, kw))))
            def cli_main(argv):
                self.assertEqual(argv, ['--original'])
                runtime = SimpleNamespace()
                video.run(runtime)
                runtime.VideoReader('same-video', ctx='cpu', num_threads=0)
            modules = {'seed_video.data.video': video,
                       'seed_video.cli': SimpleNamespace(main=cli_main)}
            with patch.object(sys, 'path', list(sys.path)), patch.dict(os.environ, {'PBS_JOBID': '123.gadi-pbs'}), patch.object(
                queued.importlib, 'import_module', side_effect=modules.__getitem__
            ):
                queued.worker(root, 'point', manifest, ['--original'])
            self.assertEqual(seen, [(('same-video',), dict(ctx='cpu', num_threads=1))])
            audit = json.loads((root / 'work/point/decoder_override/123.gadi-pbs.json').read_text())
            self.assertEqual(audit['state'], 'CLI_COMPLETE')
            self.assertTrue(audit['training_allowed'])
            self.assertEqual(queued.source_hash(root), before)

    def test_entrypoint_can_delegate_nonworker_cli_in_a_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, _, _, _ = self.install(Path(directory))
            result = subprocess.run([sys.executable, str(entry), '--collect'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('ORIGINAL MAIN', result.stdout)

    def test_queued_entrypoint_reaches_single_thread_child_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root)
            entry.write_text(entry.read_text().replace(
                "def main(): print('ORIGINAL MAIN')", "def main(): worker({}, {}, None)"))
            package = root / 'code/src/seed_video'
            (package / '__init__.py').write_text('')
            (package / 'data').mkdir()
            (package / 'data/__init__.py').write_text('')
            (package / 'data/video.py').write_text('''
def run(runtime): runtime.VideoReader = lambda **kwargs: kwargs
''')
            (package / 'cli.py').write_text('''
from types import SimpleNamespace
from .data import video
def main(argv):
    assert argv == ['--stage', 'core']
    runtime = SimpleNamespace()
    video.run(runtime)
    assert runtime.VideoReader() == {'num_threads': 1}
    print('SINGLE THREAD CHILD PASS')
''')
            config = json.loads((root / 'experiment.json').read_text())
            config['source'] = queued.source_hash(root)
            (root / 'experiment.json').write_text(json.dumps(config))
            queued.install(root, True)
            result = subprocess.run(
                [sys.executable, str(entry), '--root', str(root), '--worker', 'point'],
                capture_output=True, text=True, env=dict(os.environ, PBS_JOBID='123.gadi-pbs'))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('SINGLE THREAD CHILD PASS', result.stdout)
            audit = json.loads((root / 'work/point/decoder_override/123.gadi-pbs.json').read_text())
            self.assertEqual(audit['state'], 'CLI_COMPLETE')


if __name__ == '__main__':
    unittest.main()
