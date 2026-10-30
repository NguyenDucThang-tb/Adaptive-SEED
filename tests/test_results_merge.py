import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import pandas as pd
except ImportError:  # pandas chưa cài ở môi trường kiểm tra cấu trúc
    pd = None

from seed_video.experiments.components import save_results


def rows(*runs):
    return [dict(run=run, split='VH_TEST', q_acc=0.5) for run in runs]


@unittest.skipIf(pd is None, "pandas not installed")
class SaveResultsTests(unittest.TestCase):
    def runtime(self, directory, *runs):
        return SimpleNamespace(pd=pd, os=os, RES=Path(directory), ROWS=rows(*runs))

    def test_second_arm_keeps_first_arm_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            save_results(self.runtime(directory, 'base', 'seed_base_s0', 'adaptive_pos_only_s0'))
            save_results(self.runtime(directory, 'base', 'seed_base_s0', 'adaptive_neg_only_s0'))
            frame = pd.read_csv(Path(directory) / 'results.csv')
            self.assertEqual(
                sorted(frame['run']),
                ['adaptive_neg_only_s0', 'adaptive_pos_only_s0', 'base', 'seed_base_s0'],
            )

    def test_rerun_replaces_same_run_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.runtime(directory, 'base', 'seed_base_s0')
            save_results(first)
            second = self.runtime(directory, 'base', 'seed_base_s0')
            second.ROWS[1]['q_acc'] = 0.9
            save_results(second)
            frame = pd.read_csv(Path(directory) / 'results.csv')
            self.assertEqual(len(frame), 2)
            self.assertEqual(float(frame[frame['run'] == 'seed_base_s0']['q_acc'].iloc[0]), 0.9)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['results.csv'])


if __name__ == '__main__':
    unittest.main()
