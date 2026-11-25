#!/usr/bin/env python3
"""Freeze nested CounterVid training subsets and audit validation separation."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from seed_video.data.countervid import load_freeform


def build_manifest(train, validation, sizes, seed):
    ids = [row['id'] for row in train]
    val_ids = [row['id'] for row in validation]
    if len(set(ids)) != len(ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError('Duplicate IDs in dataset')
    train_anchors = {row['base_identifier'] for row in train}
    val_anchors = {row['base_identifier'] for row in validation}
    if None in train_anchors or None in val_anchors:
        raise ValueError('Missing base_identifier')
    if set(ids) & set(val_ids) or train_anchors & val_anchors:
        raise ValueError('Train and validation overlap')
    if any(size < 1 or size > len(ids) for size in sizes):
        raise ValueError(f'Subset sizes must be between 1 and {len(ids)}')
    ids.sort()
    random.Random(seed).shuffle(ids)
    return {'schema': 'countervid-freeform-subsets-v1', 'seed': seed,
            'train_count': len(ids), 'validation_ids': sorted(val_ids),
            'subsets': {str(size): ids[:size] for size in sorted(set(sizes))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 2000])
    args = parser.parse_args()
    root = Path(args.data_root)
    train = load_freeform(root, 'train')
    validation = load_freeform(root, 'validation')
    manifest = build_manifest(train, validation, args.sizes + [len(train)], args.seed)
    manifest['metadata_sha256'] = {
        split: hashlib.sha256((root / split / 'metadata.parquet').read_bytes()).hexdigest()
        for split in ('train', 'validation')
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    print(f'Wrote {path}: sizes={list(manifest["subsets"])}, validation={len(validation)}')


if __name__ == '__main__':
    main()
