import importlib
import unittest
from argparse import Namespace
from seed_video.pipeline import plan, source_hash
from seed_video.runtime import Runtime


class StructureTests(unittest.TestCase):
    def test_all_modules_import_without_gpu_side_effects(self):
        for name in plan("all"):
            module = importlib.import_module("seed_video." + name)
            self.assertTrue(callable(module.run))
            self.assertFalse(hasattr(module, "model"))

    def test_dependency_order(self):
        stages = plan("full")
        self.assertLess(stages.index("experiments.components"), stages.index("experiments.caption"))
        self.assertLess(stages.index("experiments.caption"), stages.index("experiments.factorial"))
        self.assertNotIn("experiments.caption", plan("no-caption"))

    def test_run_state_isolation(self):
        first, second = Runtime(Namespace()), Runtime(Namespace())
        first.model = object()
        self.assertFalse(hasattr(second, "model"))

    def test_fingerprint(self):
        self.assertEqual(len(source_hash()), 64)
        self.assertEqual(source_hash(), source_hash())
