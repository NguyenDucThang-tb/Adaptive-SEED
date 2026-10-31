import unittest
from types import SimpleNamespace

from seed_video.models.qwen import build_inputs, model_spec


class FakeTensor:
    def permute(self, *dims):
        return self


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        return "rendered prompt"

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return kwargs


class FakeNumpy:
    @staticmethod
    def asarray(value):
        return value

    @staticmethod
    def stack(values):
        return values


class FakeTorch:
    @staticmethod
    def from_numpy(value):
        return FakeTensor()


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

    def test_qwen3_keeps_pre_sampled_frames_and_metadata(self):
        processor = FakeProcessor()
        runtime = SimpleNamespace(
            CFG={"model_id": "Qwen/Qwen3-VL-8B-Instruct"},
            processor=processor,
            torch=FakeTorch(),
            np=FakeNumpy(),
        )

        build_inputs(runtime, [object() for _ in range(32)], "question", 2.0)

        self.assertNotIn("fps", processor.kwargs)
        self.assertFalse(processor.kwargs["do_sample_frames"])
        self.assertEqual(
            processor.kwargs["video_metadata"],
            [{
                "total_num_frames": 32,
                "fps": 2.0,
                "duration": 16.0,
                "frames_indices": list(range(32)),
            }],
        )


if __name__ == "__main__":
    unittest.main()
