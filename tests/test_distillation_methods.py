import unittest
from types import SimpleNamespace

from seed_video.experiments.distillation import method_registry
from seed_video.teachers.composition import coefs


class DistillationMethodTests(unittest.TestCase):
    def test_registry_exposes_kd_video_opd_and_seed(self):
        runtime = SimpleNamespace(
            step_kd_basic=lambda teacher: f"kd:{teacher}",
            step_onpolicy=lambda teacher, divergence: f"opd:{teacher}:{divergence}",
            step_seed=lambda teacher, use_gt: f"seed:{teacher}:{use_gt}",
            tvdf_select=lambda teacher: f"tvdf:{teacher}",
        )
        registry = method_registry(runtime, ["kd_basic", "video_opd", "seed"])

        self.assertEqual(registry["kd_basic@T_pos_tcd"][0](), "kd:T_pos_tcd")
        self.assertEqual(registry["video_opd@T_pos_tcd"][0](), "opd:T_pos_tcd:rkl")
        self.assertEqual(registry["video_opd@T_pos_tcd"][2](), "tvdf:T_pos_tcd")
        self.assertEqual(registry["seed@T_pos_tcd"][0](), "seed:T_pos_tcd:True")

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
