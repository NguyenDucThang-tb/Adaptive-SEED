import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from seed_video.cli import main as cli
from seed_video.data.transfer import VH_TASKS, check_video_overlap, full_vh_evaluation, write_manifest
from seed_video.data.experiment_split import configure_eval_splits
from seed_video.evaluation.inference import eval_base_cached

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/eh_vh_experiment.py'
spec = importlib.util.spec_from_file_location('eh_vh_driver', SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class TransferTests(unittest.TestCase):
    def test_seven_fixed_or_fifty_sweep_configurations(self):
        self.assertEqual(len(driver.entries(False)), 7)
        points = driver.entries(True)
        self.assertEqual(len(points), 50)
        self.assertEqual(sum(p['kind'] == 'both' for p in points), 36)
        self.assertEqual(len({p['id'] for p in points}), 50)

    def test_both_only_grid_has_exactly_forty_configurations(self):
        points = driver.entries(False, both_only=True)
        self.assertEqual(len(points), 40)
        self.assertEqual(sum(p['kind'] == 'base' for p in points), 1)
        self.assertEqual(sum(p['kind'] in ('kd_basic', 'onpolicy_gkd', 'video_opd') for p in points), 3)
        both = [p for p in points if p['kind'] == 'both']
        self.assertEqual(len(both), 36)
        self.assertEqual(len({p['group'] for p in both}), 6)
        self.assertFalse(any(p['kind'] in ('pos', 'neg') for p in points))
        self.assertEqual({(p['ap'], p['an']) for p in both},
                         {(ap, an) for ap in driver.VALUES for an in driver.VALUES})

    def test_all_methods_use_eh_training_and_common_epochs(self):
        for point in driver.entries(False):
            cmd = driver.command({'epochs': 2}, point, Path('/work'))
            self.assertEqual(cmd[cmd.index('--train-dataset') + 1], 'eventhallusion')
            self.assertEqual(cmd[cmd.index('--epochs') + 1], '2')
            self.assertIn('--eval-videohallucer', cmd)
            self.assertEqual(cmd[cmd.index('--vh-eval-tasks') + 1:cmd.index('--eh-train-fraction')], list(VH_TASKS))
            self.assertNotIn('--smoke-pairs', cmd)

    def test_full_vh_is_target_only_and_task_reports_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            items = [dict(pair_id=f'{subset}/0', role=role, subset=subset,
                          question=role, answer='yes' if role == 'basic' else 'no')
                     for subset in VH_TASKS.values() for role in ('basic', 'hallucination')]
            runtime = SimpleNamespace(CFG={'vh_eval_tasks': list(VH_TASKS), 'hf_revision': 'pinned'},
                                      WORK=Path(directory), ITEMS=items,
                                      key=lambda i: i['pair_id'] + '|' + i['role'])
            full_vh_evaluation(runtime)
            self.assertEqual(runtime.TRAIN, [])
            self.assertEqual(runtime.VAL, [])
            self.assertEqual(runtime.VH_VAL, [])
            self.assertEqual(len(runtime.VH_TEST), 10)
            runtime.args = SimpleNamespace(train_dataset='eventhallusion', eval_videohallucer=True)
            runtime.EH_VAL, runtime.EH_TEST = [], [{'subset': 'eh_mix'}]
            splits = configure_eval_splits(runtime)
            self.assertNotIn('VH_VAL', splits)
            for task in VH_TASKS:
                self.assertEqual(len(splits[f'VH_{task}_TEST']), 2)
            self.assertEqual(sum(len(splits[f'VH_{t}_TEST']) for t in VH_TASKS), len(splits['VH_TEST']))

    def test_manifest_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            write_manifest(path, {'keys': ['one']})
            write_manifest(path, {'keys': ['one']})
            with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
                write_manifest(path, {'keys': ['two']})

    def test_exact_video_overlap_fails_without_resampling(self):
        runtime = SimpleNamespace(TRAIN=[{'video': 'eh/train'}], VAL=[], TEST=[],
                                  VH_TEST=[{'video': 'vh/test'}],
                                  VIDEO_SHA={'eh/train': 'identical', 'vh/test': 'identical'})
        with self.assertRaisesRegex(ValueError, 'video SHA256'):
            check_video_overlap(runtime)
        self.assertEqual(len(runtime.VH_TEST), 1)

    def test_task_cli_cannot_change_legacy_training_implicitly(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli(['--dry-run', '--vh-eval-tasks', 'ORH'])
        with contextlib.redirect_stdout(io.StringIO()):
            cli(['--dry-run', '--train-dataset', 'eventhallusion', '--eval-videohallucer',
                 '--vh-eval-tasks', *VH_TASKS, '--epochs', '2'])

    def test_preflight_rejects_partial_vh_download_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(vh_data=root / 'vh', eh_data=root / 'eh',
                                   hf_cache=root / 'hf', env_archive=root / 'env.tar.gz', model='qwen25')
            with self.assertRaises(ValueError) as error:
                driver.preflight(args)
            self.assertIn('semantic_detail', str(error.exception))
            self.assertIn('external_factual', str(error.exception))
            self.assertIn('external_nonfactual', str(error.exception))

    def test_preflight_and_snapshot_submission_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(root=root / 'experiment', model='qwen25', epochs=2, segments=1,
                                   vh_data=root / 'vh', eh_data=root / 'eh',
                                   hf_cache=root / 'hf', env_archive=root / 'env.tar.gz')
            args.env_archive.write_bytes(b'environment fixture')
            for subset in VH_TASKS.values():
                video = args.vh_data / subset / 'videos/one.mp4'
                video.parent.mkdir(parents=True)
                video.write_bytes(b'video fixture')
                (video.parent.parent / f'{subset}.json').write_text(json.dumps([
                    {role: dict(video='one.mp4', question=role, answer='yes' if role == 'basic' else 'no')
                     for role in ('basic', 'hallucination')}]))
            for subset, folder in (('entire', 'entire'), ('mix', 'interleave'), ('misleading', 'misleading')):
                driver.atomic_json(args.eh_data / 'questions' / f'{subset}_questions.json', [])
                video = args.eh_data / 'videos' / folder / 'one.mp4'
                video.parent.mkdir(parents=True)
                video.write_bytes(b'video fixture')
            for model in (driver.MODELS['qwen25'], 'ZhengPeng7/BiRefNet'):
                snapshot = args.hf_cache / ('models--' + model.replace('/', '--')) / 'snapshots/fixed'
                driver.atomic_json(snapshot / 'config.json', {})
                (snapshot / 'model.safetensors').write_bytes(b'weight fixture')
            hashes = driver.preflight(args)
            self.assertIn('VH/semantic_detail', hashes)
            points = driver.entries(False)
            config = driver.config_for(args, points, hashes, SCRIPT.parents[1])
            commands = []
            def qsub(cmd, text):
                commands.append(cmd)
                return f'{3000 + len(commands)}.gadi-pbs\n'
            with patch.object(driver.subprocess, 'check_output', side_effect=qsub), \
                    contextlib.redirect_stdout(io.StringIO()):
                driver.submit(args, config, SCRIPT.parents[1])
            self.assertEqual(len(commands), 7)
            self.assertEqual(json.loads((args.root / 'experiment.json').read_text()), config)
            self.assertTrue((args.root / 'code/src/seed_video/data/transfer.py').exists())
            self.assertEqual(config['epochs'], 2)
            (args.vh_data / 'semantic_detail/videos/one.mp4').unlink()
            with self.assertRaisesRegex(ValueError, 'missing videos'):
                driver.preflight(args)

    def test_success_requires_all_five_tasks_not_legacy_310_row(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            point = driver.entry('both', 0.9, 0.25)
            path = work / point['result']
            path.parent.mkdir(parents=True)
            def write(splits):
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['run', 'split', 'n', 'q_acc', 'pair_acc'])
                    writer.writeheader()
                    writer.writerows(dict(run=point['run'], split=s,
                                          n=50 if s == 'VH_TEST' else 30 if s == 'EH_TEST' else 10,
                                          q_acc=.5, pair_acc=.4) for s in splits)
            splits = ['EH_VAL', 'EH_TEST', 'eh_entire_TEST', 'eh_misleading_TEST', 'eh_mix_TEST', 'VH_TEST']
            write(splits)
            self.assertFalse(driver.results_ready(work, point))
            write(splits + [f'VH_{t}_TEST' for t in VH_TASKS])
            self.assertTrue(driver.results_ready(work, point))

    def test_confusion_and_balanced_accuracy_do_not_hide_no_bias(self):
        import numpy as np
        from seed_video.evaluation.reporting import result_rows
        items = [dict(pair_id=str(i), role='eh', answer=y) for i, y in enumerate(['yes', 'yes', 'no', 'no'])]
        runtime = SimpleNamespace(SPLITS={'EH_TEST': items}, np=np, key=lambda i: i['pair_id'],
                                  MARG={'base': {'0': 1, '1': -1, '2': 1, '3': -1}})
        row = result_rows(runtime, 'student', {str(i): -1 for i in range(4)})[0]
        self.assertEqual((row['tp'], row['tn'], row['fp'], row['fn']), (0, 2, 0, 2))
        self.assertEqual(row['balanced_acc'], .5)
        self.assertEqual((row['wrong_to_correct'], row['correct_to_wrong']), (1, 1))

    def test_single_class_subset_does_not_invent_balanced_accuracy(self):
        import numpy as np
        from seed_video.evaluation.reporting import result_rows
        items = [dict(pair_id='one', role='eh', answer='no')]
        runtime = SimpleNamespace(SPLITS={'eh_entire_TEST': items}, np=np,
                                  key=lambda i: i['pair_id'], MARG={'base': {'one': 1}})
        row = result_rows(runtime, 'student', {'one': -1})[0]
        self.assertTrue(np.isnan(row['balanced_acc']))
        self.assertTrue(np.isnan(row['recall_yes']))

    def test_resume_segments_same_point_use_afterany_and_one_hour(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(root=root, model='qwen25', segments=3,
                                   eh_data=root / 'eh', vh_data=root / 'vh')
            args.eh_data.mkdir()
            args.vh_data.mkdir()
            config = {'points': [driver.entry('base', .9, .25)]}
            driver.atomic_json(root / 'experiment.json', config)
            commands = []
            def qsub(cmd, text):
                commands.append(cmd)
                return f'{2000 + len(commands)}.gadi-pbs\n'
            with patch.object(driver.subprocess, 'check_output', side_effect=qsub), \
                    contextlib.redirect_stdout(io.StringIO()):
                driver.submit(args, config, SCRIPT.parents[1])
            self.assertEqual(len(commands), 3)
            self.assertNotIn('-W', commands[0])
            self.assertEqual(commands[1][commands[1].index('-W') + 1], 'depend=afterany:2001.gadi-pbs')
            self.assertTrue(all('walltime=01:00:00' in ' '.join(cmd) for cmd in commands))
            self.assertEqual(len({cmd[cmd.index('-v') + 1] for cmd in commands}), 1)

    def test_base_evaluation_resumes_after_interruption(self):
        class Torch:
            @staticmethod
            def save(obj, path):
                with open(path, 'wb') as stream:
                    pickle.dump(obj, stream)
            @staticmethod
            def load(path, **kwargs):
                with open(path, 'rb') as stream:
                    return pickle.load(stream)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = [dict(id=str(i), video=f'/data/v{i}', question='q', answer='yes') for i in range(3)]
            runtime = SimpleNamespace(CFG={}, SOURCE_HASH='test', torch=Torch,
                                      branch_cfg=lambda b: {'model': 'qwen'},
                                      key=lambda i: i['id'], cfg_hash=lambda m: 'digest')
            calls = []
            def interrupted(batch):
                calls.extend(i['id'] for i in batch)
                if batch[0]['id'] == '1':
                    raise RuntimeError('simulated timeout')
                return {i['id']: 1.0 for i in batch}
            runtime.eval_items = interrupted
            with patch.dict(os.environ, {'SEED_EVAL_CHECKPOINT_EVERY': '1'}):
                with self.assertRaisesRegex(RuntimeError, 'timeout'):
                    eval_base_cached(runtime, items, root / 'base.pt')
                calls.clear()
                runtime.eval_items = lambda batch: (calls.extend(i['id'] for i in batch) or
                                                    {i['id']: 1.0 for i in batch})
                result = eval_base_cached(runtime, items, root / 'base.pt')
            self.assertEqual(set(result), {'0', '1', '2'})
            self.assertEqual(calls, ['1', '2'])

    def test_collection_selects_eh_validation_not_vh_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = driver.entry('base', .9, .25)
            a, b = driver.entry('both', 0, .5), driver.entry('both', .75, .25)
            driver.atomic_json(root / 'experiment.json', {'model': 'qwen25', 'points': [base, a, b]})
            for point, val, target in ((base, .5, .5), (a, .8, .4), (b, .6, .9)):
                work = root / 'work' / point['id']
                path = work / point['result']
                path.parent.mkdir(parents=True)
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['run', 'split', 'n', 'q_acc', 'pair_acc'])
                    writer.writeheader()
                    for split in ('EH_VAL', 'EH_TEST', 'eh_entire_TEST', 'eh_misleading_TEST', 'eh_mix_TEST',
                                  'VH_TEST', *[f'VH_{t}_TEST' for t in VH_TASKS]):
                        writer.writerow(dict(run=point['run'], split=split,
                                             n=50 if split == 'VH_TEST' else 30 if split == 'EH_TEST' else 10,
                                             q_acc=val if split == 'EH_VAL' else target, pair_acc=target))
                driver.atomic_json(work / 'COMPLETE.json', {'test': True})
            with contextlib.redirect_stdout(io.StringIO()):
                driver.collect(root)
            rows = driver.read_results(root / 'selected_by_eh_val.csv')
            selected = next(r for r in rows if r['method'] == 'both')
            self.assertEqual(selected['id'], a['id'])
            self.assertEqual(selected['selection_complete'], 'True')
            (root / 'work' / b['id'] / 'COMPLETE.json').unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                driver.collect(root)
            rows = driver.read_results(root / 'selected_by_eh_val.csv')
            self.assertEqual(next(r for r in rows if r['method'] == 'both')['selection_complete'], 'False')


if __name__ == '__main__':
    unittest.main()
