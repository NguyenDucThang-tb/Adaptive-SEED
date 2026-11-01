import unittest
from types import SimpleNamespace

from seed_video.models import model_spec
from seed_video.models.internvl import get_feature_grid, model_spec as internvl_model_spec


class InternVLModelSpecTests(unittest.TestCase):
    def test_internvl35_8b_uses_one_frame_per_visual_grid(self):
        model_id = "OpenGVLab/InternVL3_5-8B-Instruct"
        self.assertEqual(model_spec(model_id), ("InternVLChatModel", 28))
        self.assertEqual(internvl_model_spec(model_id), ("InternVLChatModel", 28))

    def test_other_internvl_variants_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported InternVL model"):
            internvl_model_spec("OpenGVLab/InternVL3_5-14B-Instruct")

    def test_visual_grid_uses_one_token_grid_per_frame(self):
        runtime = SimpleNamespace(CFG={"frame_size": 448, "spatial_stride": 28})
        inp = {"image_flags": SimpleNamespace(shape=(32, 1))}
        self.assertEqual(get_feature_grid(runtime, inp), (32, 32, 32))


if __name__ == "__main__":
    unittest.main()
