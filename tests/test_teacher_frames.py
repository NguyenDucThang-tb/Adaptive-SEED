import unittest
from argparse import Namespace
from types import MethodType

try:
    import numpy as np
except ImportError:  # numpy chưa cài ở môi trường kiểm tra cấu trúc
    np = None

from seed_video.runtime import Runtime
from seed_video.data.video import view_frames


@unittest.skipIf(np is None, "numpy not installed")
class TeacherFramesTests(unittest.TestCase):
    def make_runtime(self):
        state = Runtime(Namespace())
        state.np = np
        state.CFG = dict(n_frames=8, frame_size=448, tcd_frames=4)
        state.calls = []

        def decode(path, n, size):
            state.calls.append(n)
            return tuple(range(n)), 4.0
        state._decode = decode
        state.view_frames = MethodType(view_frames, state)
        return state

    def test_student_default_uses_n_frames(self):
        state = self.make_runtime()
        frames, _, _ = state.view_frames({'video': 'v'}, 'orig')
        self.assertEqual(len(frames), 8)

    def test_teacher_frames_override(self):
        state = self.make_runtime()
        frames, fps, _ = state.view_frames({'video': 'v'}, 'orig', 32)
        self.assertEqual((len(frames), state.calls), (32, [32]))
        self.assertEqual(fps, 8.0)

    def test_tcd_samples_four_from_teacher_frames(self):
        state = self.make_runtime()
        frames, _, idx = state.view_frames({'video': 'v'}, 'tcd', 32)
        self.assertEqual((len(frames), idx), (4, [0, 10, 21, 31]))


if __name__ == '__main__':
    unittest.main()
