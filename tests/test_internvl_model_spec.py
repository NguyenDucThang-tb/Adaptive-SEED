import unittest
from types import SimpleNamespace

from seed_video.models import model_spec
from seed_video.models.internvl import (
    generate_response,
    get_feature_grid,
    model_spec as internvl_model_spec,
)
from seed_video.models.foreground import temporal_grid_indices


class InternVLModelSpecTests(unittest.TestCase):
    def test_foreground_temporal_grid_resamples_teacher_to_student_frames(self):
        self.assertEqual(temporal_grid_indices(32, 8), [0, 4, 9, 13, 18, 22, 27, 31])
        self.assertEqual(temporal_grid_indices(16, 4), [0, 5, 10, 15])
        self.assertEqual(temporal_grid_indices(8, 1), [0])

    def test_generation_drops_image_flags_but_preserves_forward_inputs(self):
        class Model:
            def generate(self, **kwargs):
                return kwargs

        runtime = SimpleNamespace(model=Model())
        inputs = {
            "input_ids": [1, 2],
            "pixel_values": [3],
            "attention_mask": [1, 1],
            "image_flags": [1],
        }

        result = generate_response(runtime, inputs, max_new_tokens=4)

        self.assertNotIn("image_flags", result)
        self.assertEqual(result["input_ids"], [1, 2])
        self.assertEqual(result["pixel_values"], [3])
        self.assertEqual(result["max_new_tokens"], 4)
        self.assertIn("image_flags", inputs)

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
