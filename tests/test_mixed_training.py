import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from seed_video.cli import main as cli
from seed_video.data.experiment_split import run as run_split, split_eventhallusion
from seed_video.data.videohallucer import key, run as run_vh, take
from seed_video.evaluation.inference import eval_set
from seed_video.training.trainer import arm_paths, student_cfg, train_arm

try:
    import numpy as np
    import pandas as pd
    import torch
except ImportError:
    torch = None


class MixedSplitTests(unittest.TestCase):
    def runtime(self, directory, smoke=None):
        root = Path(directory)

        def item(source, video, role='eh', subset='eh_mix'):
            return dict(pair_id=f'{video}', role=role, subset=subset,
                        video=str(root / source / f'{video}.mp4'),
                        question=f'Question {video}', answer='yes')

        vh = [[item('VH', video, role, 'temporal') for role in ('basic', 'hallucination')]
              for video in ('train', 'val', 'test')]
        eh = [item('EH', f'{subset}/{video}', subset=subset)
              for subset in ('eh_mix', 'eh_entire', 'eh_misleading') for video in range(12)]
        runtime = SimpleNamespace(
            args=SimpleNamespace(train_dataset='mixed', smoke_pairs=smoke, no_eh=False,
                                 eh_train_fraction=0.7, eh_val_fraction=0.1, eh_split_seed=42),
            CFG={'eval_train': True}, WORK=root, TRAIN=vh[0], VAL=vh[1], TEST=vh[2],
            EH=eh, ITEMS=sum(vh, []), os=os, json=json,
        )
        runtime.key = MethodType(key, runtime)
        return runtime

    def test_train_union_heldouts_and_unique_namespaced_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            run_split(runtime)
            self.assertEqual(runtime.TRAIN, runtime.VH_TRAIN + runtime.EH_TRAIN)
            self.assertEqual(len(runtime.TRAIN), len({runtime.key(i) for i in runtime.TRAIN}))
            splits = runtime.configure_eval_splits()
            self.assertEqual(set(splits), {'VH_VAL', 'VH_TEST', 'EH_VAL', 'EH_TEST',
                                         'eh_mix_TEST', 'eh_entire_TEST', 'eh_misleading_TEST'})
            self.assertNotIn('EH_ALL', splits)
            self.assertEqual(runtime.EH, runtime.EH_TEST)
            train_keys = {runtime.key(i) for i in runtime.TRAIN}
            self.assertFalse(train_keys & {runtime.key(i) for i in eval_set(runtime)})
            for items in splits.values():
                self.assertFalse(train_keys & {runtime.key(i) for i in items})
            video_sets = [{i['video'] for i in items} for items in
                          (runtime.TRAIN, runtime.VAL, runtime.TEST)]
            self.assertTrue(all(not video_sets[a] & video_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))))
            self.assertEqual(set(runtime.ALL_VIDEOS), set.union(*video_sets))

    def test_mixed_keys_disambiguate_sources_and_preserve_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            item = dict(pair_id='same', role='same', subset='same', source='VH')
            self.assertNotEqual(runtime.key(item), runtime.key(dict(item, source='EH')))
            for dataset in ('videohallucer', 'eventhallusion'):
                runtime.args.train_dataset = dataset
                self.assertEqual(runtime.key(item), 'same|same')
            self.assertEqual(key(SimpleNamespace(), item), 'same|same')

    def test_mixed_uses_the_same_eh_partition_as_single_source(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            expected = split_eventhallusion(runtime.EH)
            run_split(runtime)
            actual = (runtime.EH_TRAIN, runtime.EH_VAL, runtime.EH_TEST)
            self.assertEqual([[i['pair_id'] for i in items] for items in actual],
                             [[i['pair_id'] for i in items] for items in expected])

    def test_cross_source_shared_resolved_path_fails_before_smoke(self):
        for smoke in (None, 1):
            with self.subTest(smoke=smoke), tempfile.TemporaryDirectory() as directory:
                runtime = self.runtime(directory, smoke)
                eh_val = split_eventhallusion(runtime.EH)[1]
                target = Path(eh_val[-1]['video'])
                target.parent.mkdir(parents=True)
                target.touch()
                alias = Path(directory) / 'alias.mp4'
                alias.symlink_to(target)
                runtime.TRAIN[0]['video'] = str(alias)
                with self.assertRaisesRegex(ValueError, 'resolved video'):
                    run_split(runtime)
                self.assertFalse((Path(directory) / 'mixed_train_val_test.json').exists())

    def test_resolved_aliases_inside_vh_splits_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            runtime.VAL[0]['video'] = str(Path(runtime.TRAIN[0]['video']).parent / '.' / 'train.mp4')
            with self.assertRaisesRegex(ValueError, 'video leakage'):
                run_split(runtime)

    def test_eh_aliases_are_grouped_before_splitting(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            original = runtime.EH[0]
            target = Path(original['video'])
            target.parent.mkdir(parents=True)
            target.touch()
            alias = Path(directory) / 'alias.mp4'
            alias.symlink_to(target)
            runtime.EH.append(dict(original, pair_id='alias-question', video=str(alias)))
            run_split(runtime)
            owners = [items for items in (runtime.EH_TRAIN, runtime.EH_VAL, runtime.EH_TEST)
                      if any(i['pair_id'] == original['pair_id'] for i in items)]
            self.assertEqual(len(owners), 1)
            self.assertIn('alias-question', {i['pair_id'] for i in owners[0]})

    def test_manifest_is_repeatable_and_rejects_split_or_content_drift(self):
        for change in ('seed', 'content', 'smoke'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                run_split(self.runtime(directory))
                path = Path(directory) / 'mixed_train_val_test.json'
                original = path.read_bytes()
                run_split(self.runtime(directory))
                self.assertEqual(path.read_bytes(), original)
                runtime = self.runtime(directory)
                if change == 'seed':
                    runtime.args.eh_split_seed += 1
                elif change == 'content':
                    runtime.EH[0]['question'] = 'Changed annotation'
                else:
                    runtime.args.smoke_pairs = 1
                with self.assertRaisesRegex(ValueError, 'Immutable manifest mismatch'):
                    run_split(runtime)
                self.assertEqual(path.read_bytes(), original)

    def test_duplicate_source_items_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            runtime.EH.append(dict(runtime.EH[0]))
            with self.assertRaisesRegex(ValueError, 'Duplicate mixed item key'):
                run_split(runtime)

    def test_no_eh_and_empty_vh_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            runtime.args.no_eh = True
            with self.assertRaisesRegex(ValueError, '--no-eh'):
                run_split(runtime)
            runtime.args.no_eh = False
            runtime.TRAIN = []
            with self.assertRaisesRegex(ValueError, 'nonempty VH'):
                run_split(runtime)

    def test_vh_pair_selection_uses_resolved_paths_in_mixed_only(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            target = Path(directory) / 'video.mp4'
            target.touch()
            alias = Path(directory) / 'alias.mp4'
            alias.symlink_to(target)
            runtime.pairs = {'pair': [{'video': str(alias)}]}
            self.assertEqual(take(runtime, ['pair'], 1, {str(target)})[0], [])
            runtime.args.train_dataset = 'videohallucer'
            self.assertEqual(take(runtime, ['pair'], 1, {str(target)})[0], ['pair'])

    def test_cli_accepts_mixed_and_validates_budget_without_ml(self):
        with contextlib.redirect_stdout(io.StringIO()):
            cli(['--dry-run', '--train-dataset', 'mixed', '--max-train-steps', '3'])
        for flags in (['--no-eh'], ['--max-train-steps', '0'], ['--max-train-steps', '-1'],
                      ['--eval-videohallucer', '--vh-eval-tasks', 'TPH']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli(['--dry-run', '--train-dataset', 'mixed', *flags])


@unittest.skipIf(torch is None, 'CPU torch/numpy/pandas required')
class MixedRuntimeTests(unittest.TestCase):
    runtime = MixedSplitTests.runtime

    def test_smoke_limits_both_sources_with_offline_vh_loader(self):
        from seed_video.config import run as run_config

        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory, smoke=2)
            runtime.args.work_dir = Path(directory)
            runtime.args.max_train_steps = 3
            with patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.cuda, 'is_bf16_supported', return_value=True):
                run_config(runtime)
            self.assertTrue(runtime.CFG['eh']['enable'])
            self.assertEqual(runtime.CFG['max_train_steps'], 3)
            root = Path(runtime.CFG['data_root']) / 'temporal'
            root.mkdir(parents=True)
            records = []
            for video in range(12):
                (root / f'{video}.mp4').touch()
                records.append({role: dict(video=f'{video}.mp4', question=role, answer='yes')
                                for role in ('basic', 'hallucination')})
            (root / 'temporal.json').write_text(json.dumps(records))
            hub = SimpleNamespace(HfApi=Mock(), hf_hub_download=Mock(), snapshot_download=Mock())
            with patch.dict(sys.modules, {'huggingface_hub': hub}), \
                    patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}):
                run_vh(runtime)
            run_split(runtime)
            self.assertEqual(len({i['pair_id'] for i in runtime.VH_TRAIN}), 2)
            self.assertEqual(len({i['video'] for i in runtime.EH_TRAIN}), 2)
            for items in (runtime.VH_VAL, runtime.VH_TEST, runtime.EH_VAL, runtime.EH_TEST):
                self.assertEqual(len({i['video'] for i in items}), 1)
            self.assertFalse({runtime.key(i) for i in runtime.TRAIN}
                             & {runtime.key(i) for i in eval_set(runtime)})


@unittest.skipIf(torch is None, 'CPU torch/numpy/pandas required')
class OptimizerBudgetTests(unittest.TestCase):
    def runtime(self, directory, budget=3, count=3, epochs=5, grad_accum=2, skip=False):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        model.config = SimpleNamespace(use_cache=True)
        runtime = SimpleNamespace(
            CFG=dict(epochs=epochs, grad_accum=grad_accum, lr=0.1, grad_ckpt=False,
                     train_dataset='videohallucer'),
            TRAIN=[{'id': str(i)} for i in range(count)], TEST=[{'id': 'test'}],
            model=model, os=os, torch=torch, np=np, pd=pd, math=math, random=random,
            time=time, json=json, STUDENT_DIR=directory, STUDENT_HP=['epochs', 'grad_accum'],
            CODE_VERSION='test', LORA_TARGETS=[], calls=[], schedulers=[],
            key=lambda i: i['id'], eval_set=lambda: [], _wrap=lambda: None, _unwrap=lambda: None,
            set_seed=torch.manual_seed, get_peft_model_state_dict=lambda model: model.state_dict(),
            set_peft_model_state_dict=lambda model, state: model.load_state_dict(state),
            eval_items=lambda items: {}, margin=lambda value: value,
            student_last=lambda *a, **kw: 0, MARG={'base': {'test': 0}},
            cfg_hash=lambda cfg: hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10],
        )
        if budget is not None:
            runtime.CFG['max_train_steps'] = budget

        def step(item, ctx):
            runtime.calls.append((item['id'], dict(ctx)))
            return (None, {}) if skip else ((model.weight - 1).square().sum(), {})

        def scheduler(opt, warmup, total):
            runtime.schedule_total = total
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
            runtime.schedulers.append(sched)
            return sched

        runtime.METHODS = {'arm': (lambda: step, None, None)}
        runtime.get_cosine_schedule_with_warmup = scheduler
        runtime.student_cfg = MethodType(student_cfg, runtime)
        runtime.arm_paths = MethodType(arm_paths, runtime)
        return runtime

    def test_exact_budget_counts_optimizer_updates_across_epochs_and_partial_batches(self):
        for count in (3, 10):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                runtime = self.runtime(directory, count=count)
                out = train_arm(runtime, 'arm', 0)
                self.assertEqual(runtime.schedulers[0].last_epoch, 3)
                self.assertEqual(runtime.schedule_total, 3)
                self.assertEqual(len(runtime.calls), 5 if count == 3 else 6)
                self.assertTrue(Path(runtime.arm_paths('arm', 0)[1]).exists())
                self.assertEqual(out['optimizer_steps_completed'], 3)
                jpath = Path(runtime.arm_paths('arm', 0)[2])
                self.assertEqual(json.loads(jpath.read_text()), runtime.student_cfg('arm', 0))
                budget = json.loads(jpath.with_name(jpath.stem + '_training_budget.json').read_text())
                self.assertEqual(budget['optimizer_steps_completed'], 3)

    def test_impossible_epoch_budget_fails_without_publishing_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory, budget=3, epochs=1)
            with self.assertRaisesRegex(ValueError, 'Cannot meet max_train_steps'):
                train_arm(runtime, 'arm', 0)
            self.assertEqual(runtime.calls, [])
            self.assertFalse(Path(runtime.arm_paths('arm', 0)[1]).exists())

    def test_budget_applies_per_adapter_for_all_training_datasets(self):
        for dataset in ('videohallucer', 'eventhallusion', 'mixed'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                runtime = self.runtime(directory, budget=1, count=1, grad_accum=8)
                runtime.args = SimpleNamespace(train_dataset=dataset)
                runtime.CFG['train_dataset'] = dataset
                runtime.METHODS['second'] = runtime.METHODS['arm']
                for name in ('arm', 'second'):
                    self.assertEqual(train_arm(runtime, name, 0)['optimizer_steps_completed'], 1)
                self.assertEqual(len(runtime.calls), 2)
                self.assertEqual([s.last_epoch for s in runtime.schedulers], [1, 1])

    def test_filtered_epochs_count_actual_steps_not_original_data_length(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            factory, teacher, _ = runtime.METHODS['arm']
            runtime.METHODS['arm'] = (factory, teacher, lambda: lambda items, ep: items[:1])
            train_arm(runtime, 'arm', 0)
            self.assertEqual([ctx['ep'] for _, ctx in runtime.calls], [0, 1, 2])
            self.assertEqual(runtime.schedulers[0].last_epoch, 3)

    def test_skipped_losses_do_not_count_as_steps_and_fail_unmet_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory, skip=True)
            with self.assertRaisesRegex(RuntimeError, 'completed 0/3'):
                train_arm(runtime, 'arm', 0)
            self.assertEqual(runtime.schedulers[0].last_epoch, 0)
            self.assertFalse(Path(runtime.arm_paths('arm', 0)[1]).exists())

    def test_resume_restores_step_count_and_does_not_restart_budget(self):
        from seed_video.training import trainer

        for interrupt_at in (1, 3):
            with self.subTest(interrupt_at=interrupt_at), tempfile.TemporaryDirectory() as directory:
                runtime = self.runtime(directory)
                original_save = trainer._atomic_torch_save

                def interrupt(runtime, value, path):
                    original_save(runtime, value, path)
                    if str(path).endswith('.train_checkpoint.pt') and value['opt_steps'] == interrupt_at:
                        raise InterruptedError('test interruption after durable checkpoint')

                with patch.dict(os.environ, {'SEED_CHECKPOINT_EVERY_STEPS': '1'}), \
                        patch.object(trainer, '_atomic_torch_save', side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        train_arm(runtime, 'arm', 0)
                previous_calls = len(runtime.calls)
                resumed = self.runtime(directory)
                train_arm(resumed, 'arm', 0)
                self.assertEqual(resumed.schedulers[0].last_epoch, 3)
                self.assertEqual(previous_calls + len(resumed.calls), 5)
                if interrupt_at == 3:
                    self.assertEqual(resumed.calls, [])
                self.assertFalse(Path(resumed.arm_paths('arm', 0)[0] + '.train_checkpoint.pt').exists())
                with tempfile.TemporaryDirectory() as baseline_dir:
                    baseline = self.runtime(baseline_dir)
                    train_arm(baseline, 'arm', 0)
                    torch.testing.assert_close(resumed.model.weight, baseline.model.weight)

    def test_no_budget_preserves_epoch_training_and_legacy_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory, budget=None, epochs=2)
            legacy = dict(code='test', name='arm', seed=0, lora_targets=[],
                          hp={'epochs': 2, 'grad_accum': 2}, train=['0', '1', '2'])
            self.assertEqual(runtime.student_cfg('arm', 0), legacy)
            legacy_path = runtime.arm_paths('arm', 0)
            out = train_arm(runtime, 'arm', 0)
            self.assertNotIn('optimizer_steps_completed', out)
            self.assertEqual(list(Path(directory).glob('*_training_budget.json')), [])
            self.assertEqual(runtime.schedulers[0].last_epoch, 4)
            runtime.CFG['max_train_steps'] = 3
            self.assertNotEqual(runtime.arm_paths('arm', 0), legacy_path)
            self.assertEqual(runtime.student_cfg('arm', 0)['hp']['max_train_steps'], 3)

    def test_dataset_provenance_changes_only_explicit_budget_metadata_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory, budget=None)
            legacy_cfg = runtime.student_cfg('arm', 0)
            legacy_paths = runtime.arm_paths('arm', 0)
            budget_paths = []
            for dataset in ('videohallucer', 'eventhallusion', 'mixed'):
                runtime.CFG['train_dataset'] = dataset
                self.assertEqual(runtime.student_cfg('arm', 0), legacy_cfg)
                self.assertEqual(runtime.arm_paths('arm', 0), legacy_paths)
                runtime.CFG['max_train_steps'] = 3
                self.assertEqual(runtime.student_cfg('arm', 0)['hp']['train_dataset'], dataset)
                budget_paths.append(runtime.arm_paths('arm', 0))
                del runtime.CFG['max_train_steps']
            self.assertEqual(len(set(budget_paths)), 3)

    def test_adapter_reuse_reads_budget_without_rewriting_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            train_arm(runtime, 'arm', 0)
            path, apath, jpath = runtime.arm_paths('arm', 0)
            budget_path = Path(jpath).with_name(Path(jpath).stem + '_training_budget.json')
            tracked = [Path(apath), Path(jpath), budget_path]
            before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in tracked]
            Path(path).unlink()
            resumed = self.runtime(directory)
            out = train_arm(resumed, 'arm', 0)
            self.assertEqual(resumed.calls, [])
            self.assertEqual(out['optimizer_steps_completed'], 3)
            self.assertEqual([(p.read_bytes(), p.stat().st_mtime_ns) for p in tracked], before)
            Path(path).unlink()
            budget_path.write_text(json.dumps({'optimizer_steps_completed': 2}))
            with self.assertRaisesRegex(RuntimeError, 'budget marker mismatch'):
                train_arm(self.runtime(directory), 'arm', 0)
            budget_path.unlink()
            with self.assertRaisesRegex(RuntimeError, 'Missing optimizer budget marker'):
                train_arm(self.runtime(directory), 'arm', 0)

    def test_student_hp_includes_budget_only_when_explicit(self):
        from seed_video.training.trainer import run as run_trainer

        peft = SimpleNamespace(get_peft_model_state_dict=Mock(), set_peft_model_state_dict=Mock())
        for budget in (None, 3):
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as directory:
                runtime = self.runtime(directory, budget=budget)
                runtime.CACHE = directory
                with patch.dict(sys.modules, {'peft': peft}):
                    run_trainer(runtime)
                self.assertEqual('max_train_steps' in runtime.STUDENT_HP, budget is not None)


if __name__ == '__main__':
    unittest.main()
