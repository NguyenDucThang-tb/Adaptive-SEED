import copy
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/decode_eof_fallback.py'
spec = importlib.util.spec_from_file_location('eof_fallback_test', SCRIPT)
fallback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fallback)


class Reader:
    def __init__(self, error=None):
        self.error = error

    def __len__(self):
        return 8

    def get_avg_fps(self):
        return 24.0

    def get_batch(self, indices):
        if self.error:
            raise self.error
        return 'original batch'


class EOFFallbackTests(unittest.TestCase):
    def test_success_keeps_decord_and_does_not_invoke_fallback(self):
        reader = fallback.EOFReader(Reader(), 'clip', {}, lambda: None)
        with patch.object(fallback, 'decode_in_child') as child:
            self.assertEqual(reader.get_batch([0, 7]), 'original batch')
            child.assert_not_called()
        self.assertEqual(len(reader), 8)
        self.assertEqual(reader.get_avg_fps(), 24.0)

    def test_eof_uses_exact_indices_and_audits_backend(self):
        audit, reports = {}, []
        array = np.zeros((3, 4, 5, 3), dtype=np.uint8)
        reader = fallback.EOFReader(Reader(RuntimeError('Unable to handle EOF')), 'clip',
                                    audit, lambda: reports.append(copy.deepcopy(audit)))
        with patch.object(fallback, 'decode_in_child', return_value=(array, {'backend': 'FFMPEG'})) as child:
            result = reader.get_batch(np.array([7, 0, 7]))
        child.assert_called_once_with('clip', [7, 0, 7])
        self.assertIs(result.asnumpy(), array)
        self.assertEqual(audit['fallbacks'][0]['state'], 'COMPLETE')
        self.assertEqual(reports[0]['fallbacks'][0]['state'], 'STARTED')
        self.assertEqual(reports[-1]['fallbacks'][0]['state'], 'COMPLETE')

    def test_non_eof_error_is_not_hidden(self):
        reader = fallback.EOFReader(Reader(ValueError('other error')), 'clip', {}, lambda: None)
        with patch.object(fallback, 'decode_in_child') as child:
            with self.assertRaisesRegex(ValueError, 'other error'):
                reader.get_batch([0])
            child.assert_not_called()

    def test_fallback_failure_is_reported_not_padded(self):
        audit = {}
        reader = fallback.EOFReader(Reader(RuntimeError('Unable to handle EOF')), 'clip', audit, lambda: None)
        with patch.object(fallback, 'decode_in_child', side_effect=RuntimeError('Missing decoded frame')):
            with self.assertRaisesRegex(RuntimeError, 'Missing decoded frame'):
                reader.get_batch([7])
        self.assertEqual(audit['fallbacks'][0]['state'], 'FAILED')

    def test_timeout_is_bounded(self):
        with patch.object(fallback.subprocess, 'run', side_effect=subprocess.TimeoutExpired('child', 1)):
            with self.assertRaisesRegex(RuntimeError, 'exceeded 1s'):
                fallback.decode_in_child('clip', [0], timeout=1)

    def test_invalid_indices_are_rejected(self):
        for values in ([], [-1], [1.5], ['1']):
            with self.subTest(values=values), self.assertRaises(ValueError):
                fallback.frame_indices(values)

    def clip(self, folder):
        try:
            import cv2
        except ImportError:
            self.skipTest('OpenCV not installed locally')
        path = Path(folder) / 'clip.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 24.0, (32, 24))
        if not writer.isOpened():
            self.skipTest('MJPG test encoder unavailable')
        for index in range(8):
            writer.write(np.full((24, 32, 3), (index * 20, 30, 200), dtype=np.uint8))
        writer.release()
        return path

    def test_real_child_decoder_preserves_order_duplicates_and_rgb(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.clip(folder)
            import cv2
            cap = cv2.VideoCapture(str(path))
            reference = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                reference.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            array, metadata = fallback.decode_in_child(path, [7, 0, 7, 2], timeout=30)
            np.testing.assert_array_equal(array, np.stack([reference[i] for i in [7, 0, 7, 2]]))
            self.assertEqual(metadata['decoded_frames'], 8)
            self.assertEqual(metadata['backend'], 'FFMPEG')

    def test_real_decoder_rejects_missing_last_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'Missing decoded frame 8'):
                fallback.sequential_frames(self.clip(folder), [0, 8])


if __name__ == '__main__':
    unittest.main()
