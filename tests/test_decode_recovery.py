import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/retry_eh_vh_decode.py'
spec = importlib.util.spec_from_file_location('decode_recovery', SCRIPT)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class DecodeRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'code/src/seed_video'
        source.mkdir(parents=True)
        (source / 'test.py').write_text('original = True\n')
        point = dict(id='both_ap0_an0', kind='both', run='adaptive_pos_neg_s0')
        config = dict(source=recovery.snapshot_hash(root), points=[point],
                      model='Qwen/Qwen2.5-VL-7B-Instruct')
        (root / 'experiment.json').write_text(json.dumps(config))
        work = root / 'work' / point['id']
        work.mkdir(parents=True)
        for name in ('VideoHallucer', 'EventHallusion'):
            (work / name).mkdir()
        (work / 'student_adapter.pt').write_bytes(b'weights')
        return point, work

    def test_plan_preserves_frozen_source_and_requires_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point, work = self.fixture(root)
            before = recovery.snapshot_hash(root)
            self.assertEqual(recovery.prepare(root, point['id'])[2], work)
            self.assertEqual(recovery.snapshot_hash(root), before)
            (work / 'student_adapter.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'never trains'):
                recovery.prepare(root, point['id'])

    def test_changed_frozen_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point, _ = self.fixture(root)
            (root / 'code/src/seed_video/test.py').write_text('changed = True\n')
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                recovery.prepare(root, point['id'])

    def test_active_point_blocks_retry_but_not_other_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'jobs.tsv').write_text('123.gadi-pbs\tpoint\t1\tlabel\n')
            with patch.dict(os.environ, {'PBS_JOBID': '456.gadi-pbs'}), patch.object(
                recovery.subprocess, 'check_output', return_value='123.gadi-pbs user\n'
            ):
                with self.assertRaisesRegex(ValueError, 'active job'):
                    recovery.check_inactive(root, 'point')
                recovery.check_inactive(root, 'other')
            with patch.dict(os.environ, {'PBS_JOBID': '123.gadi-pbs'}), patch.object(
                recovery.subprocess, 'check_output', return_value='123.gadi-pbs user\n'
            ):
                recovery.check_inactive(root, 'point')

    def test_hook_forces_one_thread_and_restores_on_error(self):
        seen = []
        def original_video(runtime):
            runtime.VideoReader = lambda *a, **kw: seen.append((a, kw))
        video = SimpleNamespace(run=original_video)
        trainer = SimpleNamespace(train_arm=lambda *a: None)
        original_train = trainer.train_arm
        with self.assertRaisesRegex(RuntimeError, 'expected'):
            with recovery.recovery_hooks(video, trainer, {'run': 'method_s0'}, {}):
                runtime = SimpleNamespace()
                video.run(runtime)
                runtime.VideoReader('clip', ctx='cpu', num_threads=4)
                self.assertEqual(seen, [(('clip',), dict(ctx='cpu', num_threads=1))])
                raise RuntimeError('expected')
        self.assertIs(video.run, original_video)
        self.assertIs(trainer.train_arm, original_train)

    def test_guard_refuses_missing_or_mismatched_adapter_without_calling_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            adapter, metadata = path / 'x_adapter.pt', path / 'x.json'
            calls = []
            trainer = SimpleNamespace(train_arm=lambda *a: calls.append(a))
            video = SimpleNamespace(run=lambda r: None)
            runtime = SimpleNamespace(
                arm_paths=lambda *a: (str(path / 'x.pt'), str(adapter), str(metadata)),
                student_cfg=lambda *a: dict(seed=0, targets=('a', 'b')),
            )
            with recovery.recovery_hooks(video, trainer, {'run': 'method_s0'}, {}):
                with self.assertRaisesRegex(ValueError, 'refusing training'):
                    trainer.train_arm(runtime, 'method', 0)
                adapter.write_bytes(b'weights')
                metadata.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'configuration does not match'):
                    trainer.train_arm(runtime, 'method', 0)
            self.assertEqual(calls, [])

    def test_guard_reuses_exact_adapter_and_audits_unchanged_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            adapter, metadata = path / 'x_adapter.pt', path / 'x.json'
            adapter.write_bytes(b'weights')
            metadata.write_text('{"seed": 0, "targets": ["a", "b"]}')
            calls = []
            def evaluate(*args):
                calls.append(args)
                return {'marg': {'example': 1.0}}
            trainer, video = SimpleNamespace(train_arm=evaluate), SimpleNamespace(run=lambda r: None)
            runtime = SimpleNamespace(
                arm_paths=lambda *a: (str(path / 'x.pt'), str(adapter), str(metadata)),
                student_cfg=lambda *a: dict(seed=0, targets=('a', 'b')),
            )
            audit = {}
            with recovery.recovery_hooks(video, trainer, {'run': 'method_s0'}, audit):
                output = trainer.train_arm(runtime, 'method', 0)
            self.assertEqual(output, {'marg': {'example': 1.0}})
            self.assertEqual(len(calls), 1)
            self.assertTrue(audit['adapter_verified_unchanged'])
            self.assertEqual(audit['adapter_sha256'], recovery.sha(adapter))

    def test_submit_snapshots_tools_and_appends_journal_without_editing_training_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point, _ = self.fixture(root)
            (root / 'logs').mkdir()
            before = recovery.snapshot_hash(root)
            with patch.object(recovery.subprocess, 'check_output', side_effect=[
                '', '123.gadi-pbs\n'
            ]) as command:
                recovery.submit(root, point['id'])
            args = command.call_args.args[0]
            self.assertIn('walltime=01:30:00', ' '.join(args))
            self.assertIn(f'EXPERIMENT_POINT={point["id"]}', ' '.join(args))
            self.assertTrue(Path(args[-1]).is_file())
            self.assertIn('123.gadi-pbs\tboth_ap0_an0\t1\t', (root / 'jobs.tsv').read_text())
            self.assertEqual(before, recovery.snapshot_hash(root))

    def test_execute_keeps_original_arguments_and_records_decoder_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point, work = self.fixture(root)
            adapter = work / 'student_adapter.pt'
            metadata = work / 'student.json'
            metadata.write_text('{"seed": 0}')
            scripts = root / 'code/scripts'
            scripts.mkdir()
            (scripts / 'eh_vh_experiment.py').write_text('''
import json
from pathlib import Path
def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
def complete(root, point): return False
def command(config, point, work): return ['python', '-u', '-m', 'seed_video', '--original']
def results_ready(work, point): return True
''')
            runtime = SimpleNamespace(
                arm_paths=lambda *a: (str(work / 'student.pt'), str(adapter), str(metadata)),
                student_cfg=lambda *a: dict(seed=0),
            )
            video = SimpleNamespace(run=lambda r: setattr(r, 'VideoReader', lambda **kw: kw))
            trainer = SimpleNamespace(train_arm=lambda *a: {'marg': {}})
            def cli_main(argv):
                self.assertEqual(argv, ['--original'])
                video.run(runtime)
                self.assertEqual(runtime.VideoReader(), {'num_threads': 1})
                trainer.train_arm(runtime, 'adaptive_pos_neg', 0)
            modules = {
                'seed_video.data.video': video,
                'seed_video.training.trainer': trainer,
                'seed_video.cli': SimpleNamespace(main=cli_main),
            }
            before = recovery.snapshot_hash(root)
            with patch.object(sys, 'path', list(sys.path)), patch.dict(sys.modules, {
                name: module for name, module in sys.modules.items() if name.startswith('seed_video')
            }), patch.dict(os.environ), patch.object(
                recovery, 'check_inactive'
            ), patch.object(recovery.importlib, 'import_module', side_effect=modules.__getitem__):
                # Other tests import the live package; execution rejects that by design.
                for name in list(sys.modules):
                    if name == 'seed_video' or name.startswith('seed_video.'):
                        del sys.modules[name]
                recovery.execute(root, point['id'])
            marker = json.loads((work / 'COMPLETE.json').read_text())
            audit = json.loads((work / marker['decoder_recovery']).read_text())
            self.assertEqual(audit['state'], 'COMPLETE')
            self.assertFalse(audit['training_allowed'])
            self.assertTrue(audit['adapter_verified_unchanged'])
            self.assertEqual(before, recovery.snapshot_hash(root))


if __name__ == '__main__':
    unittest.main()
