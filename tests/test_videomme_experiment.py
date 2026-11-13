import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import videomme_experiment as driver
import videomme_train as training


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.args = driver.parser().parse_args(['--root', str(self.base / 'experiment')])
        for key in ('env_archive', 'hf_cache', 'data_root', 'vh_data', 'eh_data'):
            setattr(self.args, key, self.base / key)
        self.args.env_archive.write_bytes(b'packed fixture')
        with patch.object(driver, 'preflight', return_value={}):
            self.config = driver.config_for(self.args)

    def quiet_submit(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return driver.submit(self.args, self.config)

    def adapter(self, dataset='mixed', teacher=None):
        work = self.args.root / 'work' / dataset
        path = work / 'students/adaptive_pos_neg_s0_exact_adapter.pt'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'adapter')
        metadata = path.with_name('adaptive_pos_neg_s0_exact.json')
        record = dict(train=['train_item'], hp=dict(model_id=self.config['model'], train_dataset=dataset,
                      max_train_steps=50, epochs=20, n_frames=8, frame_size=448),
                      teacher=teacher or dict(name='T_pos_tcd', spec=dict(a_pos=.9, negs={'tcd': .25}), branches={}))
        driver.atomic_json(metadata, record)
        driver.atomic_json(metadata.with_name(metadata.stem + '_training_budget.json'),
                           dict(max_train_steps=50, optimizer_steps_completed=50,
                                cfg_hash=self.metadata_hash(record)))
        driver.atomic_json(work / 'training_manifest.json',
                           dict(dataset=dataset, splits={'train': [dict(id='train_item')]}))
        return path, metadata

    @staticmethod
    def metadata_hash(record):
        return hashlib.md5(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def save_metadata(self, path, record):
        driver.atomic_json(path, record)
        budget_path = path.with_name(path.stem + '_training_budget.json')
        budget = json.loads(budget_path.read_text())
        driver.atomic_json(budget_path, dict(budget, cfg_hash=self.metadata_hash(record)))

    def test_defaults_and_seven_independent_jobs(self):
        self.assertEqual(self.args.datasets, list(driver.DATASETS))
        self.assertEqual((self.config['ap'], self.config['an'], self.config['max_train_steps']), (.9, .25, 50))
        self.assertEqual(self.config['epsilon_pp'], 1)
        self.assertEqual(len(self.config['jobs']), 7)
        self.assertEqual([p['id'] for p in self.config['jobs'] if p['parent']],
                         ['eval_videohallucer', 'eval_eventhallusion', 'eval_mixed'])

    def test_all_sources_fixed_command_budget_and_eval_paths(self):
        for dataset in driver.DATASETS:
            cmd = driver.train_command(self.args.root, self.config, dataset)
            self.assertEqual(cmd[:4], [sys.executable, '-u', '-m', 'seed_video'])
            for flag, value in (('--stage', 'core'), ('--arms', 'adaptive_pos_neg'),
                                ('--max-train-steps', '50'), ('--teacher-frames', '32'),
                                ('--epochs', '20'), ('--train-dataset', dataset)):
                self.assertEqual(cmd[cmd.index(flag) + 1], value)
            self.assertNotIn('--smoke-pairs', cmd)
            evaluation = driver.eval_command(self.args.root, self.config, dataset, Path('/exact_adapter.pt'))
            self.assertEqual(evaluation[evaluation.index('--output') + 1], str(self.args.root / 'evaluation' / dataset))
            self.assertEqual(evaluation[evaluation.index('--adapter') + 1], '/exact_adapter.pt')
            self.assertNotIn('--limit', evaluation)
        self.assertNotIn('--adapter', driver.eval_command(self.args.root, self.config, 'base'))

    def test_mock_submission_freezes_code_and_afterok_only_source(self):
        commands = []
        def qsub(cmd, text):
            commands.append(cmd)
            return f'{100 + len(commands)}.gadi-pbs\n'
        with patch.object(driver.subprocess, 'check_output', side_effect=qsub):
            self.assertTrue(self.quiet_submit())
        self.assertEqual(len(commands), 7)
        self.assertEqual(len(driver.submitted_jobs(self.args.root)), 7)
        for i, cmd in enumerate(commands):
            self.assertIn('walltime=01:30:00', ' '.join(cmd))
            self.assertIn('storage=gdata/hn98+scratch/hn98+scratch/jp09', cmd)
            if i in (2, 4, 6):
                self.assertEqual(cmd[cmd.index('-W') + 1], f'depend=afterok:{100+i}.gadi-pbs')
            else:
                self.assertNotIn('-W', cmd)
        for name in driver.TOOLS:
            self.assertEqual((self.args.root / 'code/scripts' / name).read_bytes(), (REPO / 'scripts' / name).read_bytes())
        self.assertEqual(driver.code_hashes(self.args.root / 'code'), self.config['code'])

    def test_queue_failure_fill_unsubmitted_no_duplicates(self):
        with patch.object(driver.subprocess, 'check_output',
                          side_effect=['101.gadi-pbs', '102.gadi-pbs', subprocess.CalledProcessError(1, 'qsub')]):
            self.assertFalse(self.quiet_submit())
        before = (self.args.root / 'jobs.tsv').read_text()
        self.args.fill_unsubmitted = True
        commands = []
        def qsub(cmd, text):
            commands.append(cmd)
            return f'{200 + len(commands)}.gadi-pbs'
        with patch.object(driver.subprocess, 'check_output', side_effect=qsub):
            self.assertTrue(self.quiet_submit())
            self.assertTrue(self.quiet_submit())
        self.assertEqual(len(commands), 5)
        self.assertEqual(commands[0][commands[0].index('-W') + 1], 'depend=afterok:102.gadi-pbs')
        self.assertTrue((self.args.root / 'jobs.tsv').read_text().startswith(before))
        self.assertEqual(len(driver.submitted_jobs(self.args.root)), 7)

    def test_no_automatic_retries_config_drift_or_ambiguous_qsub(self):
        with patch.object(driver.subprocess, 'check_output', return_value='not a job id'):
            with self.assertRaisesRegex(ValueError, 'Unexpected qsub'):
                self.quiet_submit()
        with self.assertRaisesRegex(ValueError, 'Ambiguous qsub'):
            self.quiet_submit()
        altered = dict(self.config, ap=.8)
        with self.assertRaisesRegex(ValueError, 'Immutable'):
            driver.validate_root(self.args.root, altered)
        frozen = self.args.root / 'code/scripts/decode_eof_fallback.py'
        frozen.write_text('changed fixture')
        with self.assertRaisesRegex(ValueError, 'Frozen code changed'):
            driver.validate_root(self.args.root, self.config)

    def test_duplicate_submit_requires_explicit_fill(self):
        with patch.object(driver.subprocess, 'check_output', side_effect=[f'{i}.gadi-pbs' for i in range(10, 17)]):
            self.quiet_submit()
        with self.assertRaisesRegex(ValueError, 'fill-unsubmitted'):
            self.quiet_submit()

    def test_dry_run_has_no_writes_or_qsub(self):
        flags = ['--root', str(self.args.root)]
        for flag in ('env-archive', 'hf-cache', 'data-root', 'vh-data', 'eh-data'):
            flags += ['--' + flag, str(getattr(self.args, flag.replace('-', '_')))]
        with patch.object(driver, 'preflight', return_value={}), \
                patch.object(driver.subprocess, 'check_output') as qsub, \
                contextlib.redirect_stdout(io.StringIO()):
            driver.main(flags)
        self.assertFalse(self.args.root.exists())
        qsub.assert_not_called()

    def test_smoke_requires_distinct_root_and_propagates_exact_limit(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            driver.main(['--root', str(self.args.root), '--smoke-pairs', '1',
                         '--env-archive', '/env', '--hf-cache', '/hf', '--data-root', '/mme',
                         '--vh-data', '/vh', '--eh-data', '/eh'])
        config = dict(self.config, smoke=True, smoke_pairs=2, limit=9)
        cmd = driver.eval_command(self.args.root, config, 'base')
        self.assertEqual(cmd[cmd.index('--limit') + 1], '9')
        cmd = driver.train_command(self.args.root, config, 'mixed')
        self.assertEqual(cmd[cmd.index('--smoke-pairs') + 1], '2')

    def test_model_maps_match_evaluator_and_cache_is_not_double_hub(self):
        spec = importlib.util.spec_from_file_location('mme_eval', REPO / 'scripts/eval_videomme.py')
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        self.assertEqual(driver.MODELS, evaluator.MODELS)
        self.assertEqual(self.config['hf_cache'], str(self.args.hf_cache))

    def test_exact_adapter_nested_teacher_and_component_teacher(self):
        path, metadata = self.adapter()
        self.assertEqual(driver.exact_adapter(self.args.root, self.config, 'mixed'), path)
        saved = json.loads(metadata.read_text())
        saved['teacher'] = saved['teacher']['spec']
        self.save_metadata(metadata, saved)
        self.assertEqual(driver.exact_adapter(self.args.root, self.config, 'mixed'), path)
        saved['teacher']['a_pos'] = .5
        self.save_metadata(metadata, saved)
        with self.assertRaisesRegex(ValueError, 'alpha'):
            driver.exact_adapter(self.args.root, self.config, 'mixed')

    def test_adapter_budget_dataset_manifest_and_ambiguity_rejected(self):
        path, metadata = self.adapter()
        budget_path = metadata.with_name(metadata.stem + '_training_budget.json')
        budget = json.loads(budget_path.read_text())
        for change, message in ((dict(optimizer_steps_completed=49), 'budget'),
                                (dict(cfg_hash='wrong'), 'cfg_hash')):
            driver.atomic_json(budget_path, dict(budget, **change))
            with self.assertRaisesRegex(ValueError, message):
                driver.exact_adapter(self.args.root, self.config, 'mixed')
        driver.atomic_json(budget_path, budget)
        saved = json.loads(metadata.read_text())
        saved['hp']['train_dataset'] = 'eventhallusion'
        self.save_metadata(metadata, saved)
        with self.assertRaisesRegex(ValueError, 'dataset'):
            driver.exact_adapter(self.args.root, self.config, 'mixed')
        path.with_name('adaptive_pos_neg_s0_other_adapter.pt').write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, 'found 2'):
            driver.exact_adapter(self.args.root, self.config, 'mixed')

    def test_worker_eval_demands_training_success_and_preserves_protocol_flags(self):
        path, _ = self.adapter('videohallucer')
        driver.atomic_json(self.args.root / 'experiment.json', self.config)
        work = self.args.root / 'work/videohallucer'
        with patch.object(driver, 'validate_root'), self.assertRaises(FileNotFoundError):
            driver.worker(self.args.root, 'eval_videohallucer')
        driver.atomic_json(work / 'TRAIN_COMPLETE.json', dict(dataset='videohallucer', adapter=str(path),
                           adapter_sha256=driver.sha(path), training_manifest_sha256=driver.sha(work / 'training_manifest.json')))
        def evaluate(cmd, check):
            output = Path(cmd[cmd.index('--output') + 1])
            driver.atomic_json(output / 'COMPLETE.json', {'n': 1})
        with patch.object(driver, 'validate_root'), patch.dict(os.environ), \
                patch.object(driver.subprocess, 'run', side_effect=evaluate) as process, \
                contextlib.redirect_stdout(io.StringIO()):
            driver.worker(self.args.root, 'eval_videohallucer')
            self.assertEqual(os.environ['SEED_MODEL_ID'], self.config['model'])
            self.assertEqual(os.environ['HF_HUB_CACHE'], self.config['hf_cache'])
        self.assertNotIn('--limit', process.call_args.args[0])

    def test_smoke_training_only_marks_full_sized_base_evaluation_smoke(self):
        config = dict(self.config, smoke=True, smoke_pairs=2, limit=None)
        driver.atomic_json(self.args.root / 'experiment.json', config)
        def evaluate(cmd, check):
            self.assertNotIn('--limit', cmd)
            driver.atomic_json(self.args.root / 'evaluation/base/COMPLETE.json', dict(n=2700, smoke=False))
        with patch.object(driver, 'validate_root'), patch.dict(os.environ), \
                patch.object(driver.subprocess, 'run', side_effect=evaluate), \
                contextlib.redirect_stdout(io.StringIO()):
            driver.worker(self.args.root, 'eval_base')
        self.assertTrue(json.loads((self.args.root / 'evaluation/base/COMPLETE.json').read_text())['smoke'])

    def test_preflight_missing_env_rejected(self):
        self.args.env_archive.unlink()
        with self.assertRaisesRegex(ValueError, 'environment archive'):
            driver.preflight(self.args)

    def test_preflight_video_resolution_offline_refs_and_model_cache_paths(self):
        target = self.args.data_root / 'videos/target.mp4'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'target video fixture')
        driver.atomic_json(self.args.data_root / 'questions.json', [dict(
            video_id='001', videoID='target', question_id='001-1', duration='short',
            domain='Knowledge', sub_category='Art', task_type='Recognition',
            question='Which action?', options=['A. first', 'B. second', 'C. third', 'D. fourth'], answer='A')])
        for subset in ('temporal', 'object_relation'):
            video = self.args.vh_data / subset / 'videos/source.mp4'
            video.parent.mkdir(parents=True)
            video.write_bytes(b'VH video fixture')
            driver.atomic_json(self.args.vh_data / subset / f'{subset}.json', [
                {role: dict(video=video.name) for role in ('basic', 'hallucination')}])
        for subset, folder in (('entire', 'entire'), ('mix', 'interleave'), ('misleading', 'misleading')):
            video = self.args.eh_data / 'videos' / folder / f'{subset}_1.mp4'
            video.parent.mkdir(parents=True)
            video.write_bytes(b'EH video fixture')
            driver.atomic_json(self.args.eh_data / 'questions' / f'{subset}_questions.json', [dict(id=f'{subset}_1')])
        for model_key in driver.MODELS:
            self.args.model = model_key
            cache = self.base / model_key
            self.args.hf_cache = cache if model_key == 'qwen25' else cache / 'hub'
            for model in (driver.MODELS[model_key], 'ZhengPeng7/BiRefNet'):
                model_cache = self.args.hf_cache / ('models--' + model.replace('/', '--'))
                (model_cache / 'refs').mkdir(parents=True)
                (model_cache / 'refs/main').write_text('revision')
                driver.atomic_json(model_cache / 'snapshots/revision/config.json', {})
                (model_cache / 'snapshots/revision/model.safetensors').write_bytes(b'weights')
            fingerprints = driver.preflight(self.args)
            self.assertIn(str(target), fingerprints['videos'])
            ref = self.args.hf_cache / ('models--' + driver.MODELS[model_key].replace('/', '--')) / 'refs/main'
            ref.unlink()
            with self.assertRaisesRegex(ValueError, 'offline model ref'):
                driver.preflight(self.args)
        missing = self.args.vh_data / 'object_relation/videos/source.mp4'
        missing.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing VH video'):
            driver.preflight(self.args)

    def test_pbs_environment_only_unpacks_to_jobfs(self):
        pbs = (REPO / 'scripts/run_videomme_experiment.pbs').read_text()
        self.assertIn('${PBS_JOBFS:?}/videomme_env', pbs)
        self.assertIn('conda-unpack', pbs)
        self.assertIn('unset PYTHONHOME PYTHONPATH', pbs)
        self.assertIn('PBS_O_WORKDIR', pbs)
        self.assertIn('HF_MODULES_CACHE="$PBS_JOBFS/hf_modules"', pbs)
        self.assertIn('XDG_CACHE_HOME="$PBS_JOBFS/cache"', pbs)
        subprocess.run(['bash', '-n', str(REPO / 'scripts/run_videomme_experiment.pbs')], check=True)


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = SimpleNamespace(CFG={'train_dataset': 'mixed'}, args=SimpleNamespace(smoke_pairs=None),
                                       TRAIN=[], VAL=[], TEST=[], key=lambda item: item['key'])
        self.target = self.root / 'mme/target.mp4'
        self.target.parent.mkdir()
        self.target.write_bytes(b'target video bytes')
        self.mme = [dict(video=str(self.target), videoID='target', answer='SECRET', question='DO NOT SAVE')]

    def item(self, split, name, content):
        path = self.root / split / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        item = dict(video=str(path), key=split + '/' + name)
        getattr(self.runtime, split.upper()).append(item)
        return item

    def audit(self):
        return training.audit_overlap(self.runtime, self.mme, self.root / 'training_manifest.json')

    def test_no_target_labels_in_immutable_manifest_or_unmatched_target_hashing(self):
        self.item('train', 'source.mp4', b'x')
        self.item('val', 'val.mp4', b'vv')
        self.item('test', 'test.mp4', b'ttt')
        with patch.object(training, 'sha', wraps=driver.sha) as hashed:
            result = self.audit()
            self.assertNotIn(self.target, [call.args[0] for call in hashed.call_args_list])
        self.assertEqual(result, self.audit())
        serialized = json.dumps(result)
        self.assertNotIn('SECRET', serialized)
        self.assertNotIn('DO NOT SAVE', serialized)
        self.assertNotIn('answer', serialized)
        self.item('train', 'new.mp4', b'new data')
        with self.assertRaisesRegex(ValueError, 'Immutable'):
            self.audit()

    def test_same_size_byte_identical_target_rejected_different_filename(self):
        self.item('train', 'unrelated.mp4', self.target.read_bytes())
        with self.assertRaisesRegex(ValueError, 'sha256'):
            self.audit()
        self.assertFalse((self.root / 'training_manifest.json').exists())

    def test_youtube_identity_normalizes_v_prefix_and_rejects_val_too(self):
        self.item('val', 'v_target.mp4', b'other encoding')
        with self.assertRaisesRegex(ValueError, 'youtube_identity'):
            self.audit()

    def test_same_resolved_path_rejected(self):
        self.runtime.TEST = [dict(video=str(self.target), key='source/test')]
        with self.assertRaisesRegex(ValueError, 'resolved_path'):
            self.audit()

    def test_cross_source_split_byte_duplicates_rejected(self):
        self.item('train', 'vh.mp4', b'same source bytes')
        self.item('test', 'eh.mp4', b'same source bytes')
        with self.assertRaisesRegex(ValueError, 'source_cross_split_sha256'):
            self.audit()

    def test_runtime_hooks_audit_before_video_stage_and_restore(self):
        self.item('train', 'source.mp4', b'x')
        calls = []
        video = SimpleNamespace(run=lambda r: calls.append('video'))
        split = SimpleNamespace(run=lambda r: calls.append('split'))
        old_video, old_split = video.run, split.run
        reader = object()
        fallback = SimpleNamespace(wrap_reader=lambda factory, audit, report: reader)
        self.runtime.VideoReader = object()
        audit = {}
        with training.training_hooks(video, split, fallback, self.mme, self.root, audit, lambda: calls.append('report')):
            split.run(self.runtime)
            self.assertTrue((self.root / 'training_manifest.json').exists())
            video.run(self.runtime)
            self.assertIs(self.runtime.VideoReader, reader)
        self.assertIs(video.run, old_video)
        self.assertIs(split.run, old_split)
        self.assertEqual(calls, ['split', 'report', 'video'])
        self.assertEqual(audit['overlap_audit'], 'PASSED')


if __name__ == '__main__':
    unittest.main()
