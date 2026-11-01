import unittest
from types import SimpleNamespace

from seed_video.experiments.distillation import method_registry
from seed_video.teachers.composition import coefs
from seed_video.training.losses import (
    _teacher_answer_correct,
    _tvdf_rank_reliable,
    response_tokens,
)


class DistillationMethodTests(unittest.TestCase):
    def test_response_tokens_handles_full_sequence_and_continuation_only(self):
        class Tensor:
            def __init__(self, values):
                self.values = values
                self.shape = (len(values),)

            def __getitem__(self, item):
                return Tensor(self.values[item]) if isinstance(item, slice) else self.values[item]

        class Torch:
            @staticmethod
            def equal(left, right):
                return left.values == right.values

        prompt = Tensor([10, 11])
        self.assertEqual(response_tokens([Tensor([10, 11, 12])], [prompt], Torch).values, [12])
        self.assertEqual(response_tokens([Tensor([12, 13])], [prompt], Torch).values, [12, 13])

    def test_registry_exposes_kd_and_video_opd_adaptation(self):
        runtime = SimpleNamespace(
            step_kd_basic=lambda teacher: f"kd:{teacher}",
            step_onpolicy=lambda teacher, divergence: f"opd:{teacher}:{divergence}",
            tvdf_select=lambda teacher: f"tvdf:{teacher}",
        )
        registry = method_registry(runtime, ["kd_basic", "video_opd"])

        self.assertEqual(registry["kd_basic@T_pos_tcd"][0](), "kd:T_pos_tcd")
        self.assertEqual(registry["video_opd@T_pos_tcd"][0](), "opd:T_pos_tcd:rkl")
        self.assertEqual(registry["video_opd@T_pos_tcd"][2](), "tvdf:T_pos_tcd")
        self.assertFalse(registry["video_opd@T_pos_tcd"][3])

    def test_registry_does_not_claim_unavailable_seed_implementation(self):
        runtime = SimpleNamespace(step_kd_basic=lambda teacher: None)
        with self.assertRaisesRegex(ValueError, "Unsupported distillation method: seed"):
            method_registry(runtime, ["seed"])

    def test_tvdf_reliability_uses_teacher_answer_against_label(self):
        self.assertTrue(_teacher_answer_correct(0.2, "yes"))
        self.assertTrue(_teacher_answer_correct(-0.2, "no"))
        self.assertFalse(_teacher_answer_correct(-0.2, "yes"))

    def test_tvdf_keeps_high_disagreement_only_from_reliable_rows(self):
        rows = [(0.1, "low"), (0.8, "high"), (0.4, "mid")]
        self.assertEqual(_tvdf_rank_reliable(rows, 0.5), [(0.8, "high"), (0.4, "mid")])
        self.assertEqual(_tvdf_rank_reliable([], 0.5), [])

    def test_teacher_coefficients_allow_single_branch_edges(self):
        pos_only = coefs(SimpleNamespace(), {
            "pos": "orig+fg", "a_pos": 1.0, "negs": {"tcd": 0.0}
        })
        neg_only = coefs(SimpleNamespace(), {
            "pos": "orig+fg", "a_pos": 0.0, "negs": {"tcd": 0.75}
        })
        self.assertAlmostEqual(sum(pos_only.values()), 1.0)
        self.assertAlmostEqual(sum(neg_only.values()), 1.0)
        self.assertEqual(pos_only["orig+fg"], 1.0)
        self.assertEqual(neg_only["tcd"], -0.75)


if __name__ == "__main__":
    unittest.main()
