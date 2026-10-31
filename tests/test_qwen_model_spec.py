import unittest

from seed_video.models.qwen import model_spec


class QwenModelSpecTests(unittest.TestCase):
    def test_qwen25_uses_legacy_class_and_stride(self):
        self.assertEqual(
            model_spec("Qwen/Qwen2.5-VL-7B-Instruct"),
            ("Qwen2_5_VLForConditionalGeneration", 28),
        )

    def test_qwen3_uses_qwen3_class_and_stride(self):
        self.assertEqual(
            model_spec("Qwen/Qwen3-VL-8B-Instruct"),
            ("Qwen3VLForConditionalGeneration", 32),
        )

    def test_unknown_model_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Qwen model"):
            model_spec("Qwen/unknown")


if __name__ == "__main__":
    unittest.main()
