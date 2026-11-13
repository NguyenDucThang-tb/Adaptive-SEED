import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from seed_video.evaluation.videomme import compare, summarize

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/eval_videomme.py'
spec = importlib.util.spec_from_file_location('eval_videomme_tests', SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def row(qid, video, answer='A', prediction='A'):
    return dict(question_id=qid, videoID=video, answer=answer, prediction=prediction,
                question='What?', options=['A. One', 'B. Two', 'C. Three', 'D. Four'],
                duration='short', domain='Knowledge', sub_category='Art', task_type='Recognition')


class VideoMMEEvaluationTests(unittest.TestCase):
    def test_summary_invalid_predictions_stay_in_denominator(self):
        result = summarize([row('1', 'v1'), row('2', 'v2', prediction='')])
        overall = next(r for r in result if r['group'] == 'overall')
        self.assertEqual(overall['n'], 2)
        self.assertEqual(overall['accuracy'], 0.5)
        self.assertEqual(overall['invalid'], 1)

    def test_paired_cluster_comparison_and_direction(self):
        base = [row('1', 'v1'), row('2', 'v1'), row('3', 'v2', prediction='D')]
        student = [row('1', 'v1', prediction='B'), row('2', 'v1'), row('3', 'v2')]
        a = compare(base, student, iterations=200)
        self.assertEqual(a, compare(base, student, iterations=200))
        self.assertEqual(a['n_videos'], 2)
        self.assertEqual(a['delta_pp'], 0)
        self.assertEqual(a['wrong_to_correct'], 1)
        self.assertEqual(a['correct_to_wrong'], 1)
        self.assertFalse(a['noninferior'])

    def test_identical_results_noninferior_not_improvement(self):
        data = [row('1', 'v1'), row('2', 'v2')]
        result = compare(data, data, iterations=100)
        self.assertEqual(result['lower_95_one_sided_pp'], 0)
        self.assertTrue(result['noninferior'])

    def test_comparison_rejects_missing_or_mismatched_examples(self):
        for student in ([], [row('1', 'v2')], [row('1', 'v1', answer='B')],
                        [row('1', 'v1'), row('1', 'v1')]):
            with self.subTest(student=student), self.assertRaises(ValueError):
                compare([row('1', 'v1')], student, iterations=100)

    def test_adapter_must_match_model_and_frame_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adaptive_pos_neg_s0_hash_adapter.pt'
            path.touch()
            metadata = path.with_name('adaptive_pos_neg_s0_hash.json')
            config = dict(hp=dict(model_id='Qwen/Qwen2.5-VL-7B-Instruct', n_frames=8,
                                  frame_size=448, lora_r=16, lora_alpha=32, lora_dropout=0.05),
                          lora_targets=['q_proj'], train=['train_only'])
            metadata.write_text(json.dumps(config))
            result, _ = evaluation.adapter_config(path, 'Qwen/Qwen2.5-VL-7B-Instruct', 8, 448)
            self.assertEqual(result, config)
            with self.assertRaisesRegex(ValueError, 'model_id'):
                evaluation.adapter_config(path, 'Qwen/Qwen3-VL-8B-Instruct', 8, 448)
            with self.assertRaisesRegex(ValueError, 'n_frames'):
                evaluation.adapter_config(path, 'Qwen/Qwen2.5-VL-7B-Instruct', 32, 448)

    def test_partial_write_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.json'
            evaluation.atomic_json(path, {'first': 1})
            evaluation.atomic_json(path, {'next': 2})
            self.assertEqual(json.loads(path.read_text()), {'next': 2})
            self.assertEqual([p.name for p in path.parent.iterdir()], ['predictions.json'])

    def test_generation_uses_mcq_prompt_not_binary_margin(self):
        try:
            import torch
        except ImportError:
            self.skipTest('CPU torch required')
        import contextlib

        class Inputs(dict):
            def to(self, device):
                return self

        inputs = Inputs(input_ids=torch.tensor([[1, 2]]))
        model = Mock()
        model.modules.return_value = []
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        runtime = SimpleNamespace(
            CFG={'n_frames': 8, 'frame_size': 448}, DEV='cpu', torch=torch,
            _decode=Mock(return_value=([object()] * 8, 20)),
            build_inputs=Mock(return_value=inputs), model=model,
            keep_last=lambda n: contextlib.nullcontext(),
            tok=SimpleNamespace(decode=Mock(return_value='A')))
        item = dict(row('1', 'v1'), video='clip.mp4')
        self.assertEqual(evaluation.generate_answer(runtime, item), 'A')
        self.assertFalse(runtime.build_inputs.call_args.kwargs['yn'])
        prompt = runtime.build_inputs.call_args.args[1]
        self.assertIn('A. One', prompt)
        self.assertNotIn('Answer with yes or no', prompt)
        self.assertFalse(model.generate.call_args.kwargs['do_sample'])
        self.assertEqual(runtime.tok.decode.call_args.args[0].tolist(), [3])

    def test_collector_rejects_protocol_drift_and_never_calls_smoke_retention(self):
        sys.path.insert(0, str(SCRIPT.parent))
        try:
            from collect_videomme import collect
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ('base', 'mixed'):
                    target = root / 'evaluation' / name
                    target.mkdir(parents=True)
                    predictions = target / 'predictions.json'
                    evaluation.atomic_json(predictions, dict(meta=dict(protocol={'frames': 8}),
                        predictions=[row('1', 'v1'), row('2', 'v2')]))
                    evaluation.atomic_json(target / 'COMPLETE.json', dict(n=2, smoke=True,
                        predictions_sha=evaluation.sha(predictions)))
                with patch('collect_videomme.compare', side_effect=AssertionError('Smoke cannot claim retention')):
                    results = collect(root, iterations=100)
                self.assertEqual(results[-1]['status'], 'SMOKE_COMPLETE')
                self.assertNotIn('noninferior', results[-1])
                path = root / 'evaluation/mixed/predictions.json'
                payload = json.loads(path.read_text())
                payload['meta']['protocol']['frames'] = 32
                evaluation.atomic_json(path, payload)
                evaluation.atomic_json(path.parent / 'COMPLETE.json', dict(n=2, smoke=True,
                    predictions_sha=evaluation.sha(path)))
                with self.assertRaisesRegex(ValueError, 'protocols differ'):
                    collect(root, iterations=100)
        finally:
            sys.path.remove(str(SCRIPT.parent))


if __name__ == '__main__':
    unittest.main()
