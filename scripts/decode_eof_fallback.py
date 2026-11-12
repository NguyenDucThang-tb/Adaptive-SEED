"""Explicit, bounded sequential OpenCV fallback for Decord EOF failures.

Keep Decord's requested zero-based frame indices. Never pad missing frames,
resample the clip, rewrite a video, or silently omit a benchmark example.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace


def frame_indices(values):
    indices = [int(i) for i in values]
    if not indices or min(indices) < 0 or any(i != value for i, value in zip(indices, values)):
        raise ValueError('Expected nonnegative, nonempty integer frame indices.')
    return indices


def sequential_frames(path, indices):
    import cv2
    import numpy as np

    indices = frame_indices(indices)
    cv2.setNumThreads(1)
    params = [cv2.CAP_PROP_N_THREADS, 1] if hasattr(cv2, 'CAP_PROP_N_THREADS') else []
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, params)
    try:
        if not cap.isOpened():
            raise RuntimeError(f'OpenCV/FFmpeg could not open {path}')
        requested = set(indices)
        selected = {}
        for position in range(max(indices) + 1):
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f'Missing decoded frame {position}; refusing padding/skipping: {path}')
            if position in requested:
                selected[position] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        array = np.stack([selected[i] for i in indices])
        return array, dict(opencv_version=cv2.__version__, backend=cap.getBackendName(),
                           decoded_frames=max(indices) + 1)
    finally:
        cap.release()


def decode_in_child(path, indices, timeout=300):
    import numpy as np

    indices = frame_indices(indices)
    with tempfile.TemporaryDirectory(prefix='decode_eof_', dir=os.environ.get('PBS_JOBFS')) as directory:
        output = Path(directory) / 'frames.npy'
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--video', str(path), '--indices', json.dumps(indices), '--output', str(output)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f'Sequential decode exceeded {timeout}s: {path}') from error
        if result.returncode:
            raise RuntimeError(f'Sequential decode failed: {path}\n{result.stderr[-4000:]}')
        array = np.load(output, allow_pickle=False)
        if array.ndim != 4 or array.shape[0] != len(indices) or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise RuntimeError('Fallback returned an invalid RGB batch.')
        metadata = json.loads(output.with_suffix('.json').read_text())
        return array, metadata


class EOFReader:
    def __init__(self, reader, path, audit, report):
        self._reader, self._path = reader, str(path)
        self._audit, self._report = audit, report

    def __len__(self):
        return len(self._reader)

    def __getattr__(self, name):
        return getattr(self._reader, name)

    def get_batch(self, indices):
        try:
            return self._reader.get_batch(indices)
        except Exception as error:
            if 'Unable to handle EOF' not in str(error):
                raise
            requested = frame_indices(indices)
            event = dict(video=self._path, indices=requested, original_error=str(error),
                         decoder='opencv_sequential', state='STARTED', timeout_seconds=300)
            self._audit.setdefault('fallbacks', []).append(event)
            self._report()
            print(f'DECODER EOF FALLBACK: {self._path}; indices={requested}', flush=True)
            try:
                array, metadata = decode_in_child(self._path, requested)
                event.update(state='COMPLETE', shape=list(array.shape), **metadata)
                print(f'DECODER EOF FALLBACK COMPLETE: {self._path}', flush=True)
                return SimpleNamespace(asnumpy=lambda: array)
            except Exception as fallback_error:
                event.update(state='FAILED', error=str(fallback_error))
                raise
            finally:
                self._report()


def wrap_reader(factory, audit, report=lambda: None):
    def create(*args, **kwargs):
        kwargs['num_threads'] = 1
        reader = factory(*args, **kwargs)
        if not hasattr(reader, 'get_batch'):
            return reader
        path = args[0] if args else kwargs['uri']
        return EOFReader(reader, path, audit, report)
    return create


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--indices', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    frames, metadata = sequential_frames(args.video, json.loads(args.indices))
    np.save(args.output, frames, allow_pickle=False)
    args.output.with_suffix('.json').write_text(json.dumps(metadata))


if __name__ == '__main__':
    main()
