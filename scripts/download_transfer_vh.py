"""Download the five pinned VH tasks into a separate, resumable local dataset.

Run inside the packed environment on copyq, not a GPU job. Never replaces an
unmarked existing dataset used by other experiments.
"""
import argparse
import json
from pathlib import Path

from eh_vh_experiment import REVISION, TASKS, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    root = args.destination.resolve()
    marker = root / 'transfer_download.json'
    expected = {'repo': 'bigai-nlco/VideoHallucer', 'revision': REVISION, 'tasks': list(TASKS)}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise SystemExit('Download revision/task mismatch; use a new destination.')
    elif root.exists() and any(root.iterdir()):
        raise SystemExit('Refusing to modify an existing unmarked dataset; use a new destination.')
    atomic_json(marker, expected)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=expected['repo'], repo_type='dataset', revision=REVISION,
                      allow_patterns=[f'{subset}/*' for subset in TASKS.values()],
                      local_dir=str(root), max_workers=4)
    print('Downloaded pinned task files:', root)
    print('Next: eh_vh_experiment.py preflight verifies annotations and all referenced videos.')


if __name__ == '__main__':
    main()
