import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from seed_video.training.trainer import _atomic_torch_save, _checkpoint_interval


class FakeTorch:
    @staticmethod
    def save(value, path):
        Path(path).write_text(str(value), encoding='utf-8')


class TrainingCheckpointTests(unittest.TestCase):
    def test_atomic_save_replaces_target_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, 'state.pt')
            Path(target).write_text('old', encoding='utf-8')
            runtime = SimpleNamespace(os=os, torch=FakeTorch)

            _atomic_torch_save(runtime, 'new', target)

            self.assertEqual(Path(target).read_text(encoding='utf-8'), 'new')
            self.assertEqual(list(Path(directory).iterdir()), [Path(target)])

    def test_checkpoint_intervals_use_defaults_and_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_checkpoint_interval('TEST_CHECKPOINT_INTERVAL', 5), 5)
        with patch.dict(os.environ, {'TEST_CHECKPOINT_INTERVAL': '3'}):
            self.assertEqual(_checkpoint_interval('TEST_CHECKPOINT_INTERVAL', 5), 3)

    def test_checkpoint_interval_rejects_non_positive_values(self):
        for value in ('0', '-2'):
            with self.subTest(value=value), patch.dict(
                os.environ, {'TEST_CHECKPOINT_INTERVAL': value}
            ):
                with self.assertRaises(ValueError):
                    _checkpoint_interval('TEST_CHECKPOINT_INTERVAL', 5)


if __name__ == '__main__':
    unittest.main()
