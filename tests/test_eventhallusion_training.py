import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from seed_video.data.experiment_split import (
    configure_eval_splits,
    run as run_split,
    split_eventhallusion,
)
from seed_video.evaluation.inference import eval_set
from seed_video.experiments.components import resolve_arm_run


class EventHallusionTrainingSplitTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            {
                "subset": subset,
                "pair_id": f"{subset}/{video}/{question}",
                "role": "eh",
                "video": f"/videos/{subset}/{video}.mp4",
                "question": f"question {question}",
                "answer": "yes" if question % 2 else "no",
            }
            for subset, n_videos in (("eh_entire", 20), ("eh_mix", 30), ("eh_misleading", 18))
            for video in range(n_videos)
            for question in range(2)
        ]

    def test_split_is_deterministic_video_disjoint_and_complete(self):
        first = split_eventhallusion(self.items, seed=123)
        second = split_eventhallusion(self.items, seed=123)
        self.assertEqual(
            [[item["pair_id"] for item in split] for split in first],
            [[item["pair_id"] for item in split] for split in second],
        )
        video_sets = [{item["video"] for item in split} for split in first]
        self.assertFalse(video_sets[0] & video_sets[1])
        self.assertFalse(video_sets[0] & video_sets[2])
        self.assertFalse(video_sets[1] & video_sets[2])
        self.assertEqual(sum(map(len, first)), len(self.items))
        self.assertTrue(all(len(split) for split in first))

    def test_all_source_subsets_are_preserved_in_each_split(self):
        train, val, test = split_eventhallusion(self.items, seed=42)
        expected = {"eh_entire", "eh_mix", "eh_misleading"}
        for split in (train, val, test):
            self.assertEqual({item["subset"] for item in split}, expected)

    def test_rejects_invalid_fractions(self):
        with self.assertRaisesRegex(ValueError, "sum to less than 1"):
            split_eventhallusion(self.items, train_fraction=0.8, val_fraction=0.2)

    def test_eventhallusion_training_can_add_held_out_videohallucer_eval(self):
        eh_val = [{"subset": "eh_mix", "pair_id": "eh/val/0", "role": "eh"}]
        eh_test = [{"subset": "eh_entire", "pair_id": "eh/test/0", "role": "eh"}]
        vh_val = [{"subset": "temporal", "pair_id": "vh/val", "role": "basic"}]
        vh_test = [{"subset": "temporal", "pair_id": "vh/test", "role": "basic"}]
        runtime = SimpleNamespace(
            args=SimpleNamespace(train_dataset="eventhallusion", eval_videohallucer=True),
            EH_VAL=eh_val,
            EH_TEST=eh_test,
            EH_SPLITS=["eh_entire"],
            VH_VAL=vh_val,
            VH_TEST=vh_test,
        )

        splits = configure_eval_splits(runtime)

        self.assertEqual(splits["EH_VAL"], eh_val)
        self.assertEqual(splits["EH_TEST"], eh_test)
        self.assertEqual(splits["VH_VAL"], vh_val)
        self.assertEqual(splits["VH_TEST"], vh_test)

    def test_eval_set_includes_vh_heldout_but_not_vh_train(self):
        eh_val = {"pair_id": "eh/val", "role": "eh"}
        eh_test = {"pair_id": "eh/test", "role": "eh"}
        vh_val = {"pair_id": "vh/val", "role": "basic"}
        vh_test = {"pair_id": "vh/test", "role": "basic"}
        runtime = SimpleNamespace(
            args=SimpleNamespace(eval_videohallucer=True),
            CFG={"eval_train": False},
            TRAIN=[{"pair_id": "vh/train", "role": "basic"}],
            VAL=[eh_val],
            TEST=[eh_test],
            EH=[eh_test],
            VH_VAL=[vh_val],
            VH_TEST=[vh_test],
            EXTRA_EVAL=[],
            key=lambda item: f"{item['pair_id']}|{item['role']}",
        )

        selected = eval_set(runtime)

        self.assertEqual(
            {item["pair_id"] for item in selected},
            {"eh/val", "eh/test", "vh/val", "vh/test"},
        )


    def _runtime(self, work, smoke_pairs=None):
        all_videos = sorted({item["video"] for item in self.items})
        return SimpleNamespace(
            args=SimpleNamespace(
                train_dataset="eventhallusion", eh_train_fraction=0.7,
                eh_val_fraction=0.1, eh_split_seed=42, smoke_pairs=smoke_pairs,
            ),
            EH=list(self.items), ALL_VIDEOS=all_videos, CFG={"eval_train": True},
            WORK=Path(work), os=os, json=json,
            key=lambda item: f"{item['pair_id']}|{item['role']}",
        )

    def test_smoke_limits_videos_to_selected_splits(self):
        with tempfile.TemporaryDirectory() as work:
            runtime = self._runtime(work, smoke_pairs=1)
            run_split(runtime)
            used = {i["video"] for i in runtime.TRAIN + runtime.VAL + runtime.TEST}
            self.assertEqual(set(runtime.ALL_VIDEOS), used)
            self.assertEqual(len(runtime.ALL_VIDEOS), 3)

    def test_eval_set_scores_test_items_once(self):
        with tempfile.TemporaryDirectory() as work:
            runtime = self._runtime(work)
            run_split(runtime)
            keys = [runtime.key(i) for i in eval_set(runtime)]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(len(keys), len(runtime.VAL) + len(runtime.TEST))


class ArmSelectionTests(unittest.TestCase):
    ARMS = {"seed_base": (0, 0, 0, 0), "pair_only": (0, 1, 0, 0), "adaptive_pos_only": (0, 0, 1, 0)}

    def test_default_arms_skip_pair_arms_without_coverage(self):
        self.assertEqual(resolve_arm_run(self.ARMS, None, 0), ["seed_base", "adaptive_pos_only"])

    def test_default_arms_keep_pair_arms_with_coverage(self):
        self.assertEqual(resolve_arm_run(self.ARMS, None, 3), list(self.ARMS))

    def test_explicit_pair_arm_without_coverage_fails(self):
        with self.assertRaisesRegex(ValueError, "pair_only"):
            resolve_arm_run(self.ARMS, ["seed_base", "pair_only"], 0)


if __name__ == "__main__":
    unittest.main()
