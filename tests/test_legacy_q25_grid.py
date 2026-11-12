import csv
import importlib.util
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/submit_legacy_q25_grid.py'
spec = importlib.util.spec_from_file_location('legacy_grid', SCRIPT)
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


def rows(run):
    return [dict(run=run, split='VH_TEST', n=310, q_acc=0.77, pair_acc=0.58),
            dict(run=run, split='EH_ALL', n=409, q_acc=0.64, pair_acc='')]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


class LegacyGridTests(unittest.TestCase):
    def test_requires_both_full_splits_and_valid_metrics(self):
        data = rows('seed_base_s0')
        self.assertTrue(grid.complete(data, 'seed_base_s0'))
        self.assertFalse(grid.complete(data[:1], 'seed_base_s0'))
        data[0]['n'] = 2
        self.assertFalse(grid.complete(data, 'seed_base_s0'))
        data[0]['n'], data[0]['q_acc'] = 310, 'nan'
        self.assertFalse(grid.complete(data, 'seed_base_s0'))

    def test_duplicate_split_is_not_complete(self):
        data = rows('seed_base_s0')
        self.assertFalse(grid.complete(data + data[:1], 'seed_base_s0'))

    def test_legacy_diagonal_reuses_five_and_submits_31(self):
        with tempfile.TemporaryDirectory() as directory:
            old, root = Path(directory) / 'old', Path(directory) / 'new'
            for value in grid.VALUES[1:]:
                save(grid.result_path(old, value, value),
                     rows('seed_base_s0') + rows('adaptive_pos_neg_s0'))
            # Only Pos has the same parameters here but a different adaptation rule.
            save(grid.result_path(old, 0.5, 0.25),
                 rows('seed_base_s0') + rows('adaptive_pos_only_s0'))
            points = grid.plan(old, root)
            self.assertEqual(len(points), 36)
            self.assertEqual(sum(all(done.values()) for _, _, done in points), 5)
            point = next(done for ap, an, done in points if (ap, an) == (0.5, 0.25))
            self.assertTrue(point['seed_base_s0'])
            self.assertFalse(point['adaptive_pos_neg_s0'])

    def test_new_results_are_reused_without_modifying_old(self):
        with tempfile.TemporaryDirectory() as directory:
            old, root = Path(directory) / 'old', Path(directory) / 'new'
            current = grid.result_path(grid.work_for(root, 0, 1), 0, 1)
            save(current, rows('seed_base_s0') + rows('adaptive_pos_neg_s0'))
            done = next(done for ap, an, done in grid.plan(old, root) if (ap, an) == (0, 1))
            self.assertTrue(all(done.values()))
            self.assertFalse(old.exists())

    def test_worker_only_trains_missing_adaptive(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'protocol.json').write_text('{"model": "Qwen/Qwen2.5-VL-7B-Instruct"}')
            work = grid.work_for(root, 0.5, 0.25)
            dest = grid.result_path(work, 0.5, 0.25)
            save(dest, rows('seed_base_s0'))
            def train(cmd, check):
                self.assertEqual(cmd[cmd.index('--arms') + 1:], ['adaptive_pos_neg'])
                self.assertEqual(cmd[cmd.index('--train-dataset') + 1], 'videohallucer')
                save(dest, rows('seed_base_s0') + rows('adaptive_pos_neg_s0'))
            with patch.object(grid.subprocess, 'run', side_effect=train) as run:
                grid.worker(SimpleNamespace(root=root, point=(0.5, 0.25)))
                self.assertEqual(run.call_count, 1)
            self.assertTrue((work / 'COMPLETE.json').exists())

    def test_worker_rejects_zero_exit_without_full_results(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'protocol.json').write_text('{"model": "q25"}')
            with patch.object(grid.subprocess, 'run'):
                with self.assertRaisesRegex(RuntimeError, 'results are missing'):
                    grid.worker(SimpleNamespace(root=root, point=(0, 0)))

    def test_submission_uses_six_chains_and_one_hour_jobs(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = SimpleNamespace(root=base / 'new', old_work=base / 'old',
                                   vh_data=base / 'vh', eh_data=base / 'eh')
            args.vh_data.mkdir()
            args.eh_data.mkdir()
            for value in grid.VALUES[1:]:
                save(grid.result_path(args.old_work, value, value),
                     rows('seed_base_s0') + rows('adaptive_pos_neg_s0'))
            commands = []
            def qsub(cmd, text):
                commands.append(cmd)
                return f'{1000 + len(commands)}.gadi-pbs\n'
            with patch.object(grid, 'require_inputs'), patch.object(grid, 'initialize'), \
                    patch.object(grid.subprocess, 'check_output', side_effect=qsub), \
                    contextlib.redirect_stdout(io.StringIO()):
                grid.submit(SCRIPT.parents[1], args, grid.plan(args.old_work, args.root))
            self.assertEqual(len(commands), 31)
            self.assertEqual(sum('-W' not in cmd for cmd in commands), 6)
            self.assertTrue(all('walltime=01:00:00' in ' '.join(cmd) for cmd in commands))
            self.assertTrue(all('afterany:' in cmd[cmd.index('-W') + 1]
                                for cmd in commands if '-W' in cmd))


if __name__ == '__main__':
    unittest.main()
