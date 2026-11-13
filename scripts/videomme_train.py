"""Frozen-source worker with runtime-only decoder and pre-model leakage audit."""
import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import sys

from videomme_experiment import atomic_json, exact_adapter, load_module, sha, train_command


def identity(value):
    return Path(str(value)).stem.removeprefix('v_')


def audit_overlap(runtime, mme_items, path):
    """No benchmark questions/labels are serialized or used for source selection."""
    targets = {}
    for item in mme_items:
        video = Path(item['video']).resolve()
        targets.setdefault(video, set()).add(identity(item['videoID']))
        targets[video].add(identity(video.name))
    target_ids = set().union(*targets.values())
    by_size = {}
    for video in targets:
        by_size.setdefault(video.stat().st_size, []).append(video)
    source_hashes, target_hashes, split_owners = {}, {}, {}
    manifest = dict(version='videomme_source_audit_v1', dataset=runtime.CFG['train_dataset'],
                    smoke_pairs=getattr(runtime.args, 'smoke_pairs', None),
                    policy='reject_train_val_test_path_identity_or_same_size_sha256_overlap',
                    splits={}, target_videos=[dict(path=str(p), identities=sorted(ids), size=p.stat().st_size)
                                              for p, ids in sorted(targets.items())])
    overlaps = []
    for name, items in (('train', runtime.TRAIN), ('val', runtime.VAL), ('test', runtime.TEST)):
        manifest['splits'][name] = []
        for item in items:
            video = Path(item['video']).resolve()
            size = video.stat().st_size
            # All selected source files get content provenance; only same-size MME files are read.
            if video not in source_hashes:
                source_hashes[video] = sha(video)
            record = dict(id=runtime.key(item), video=str(video), identity=identity(video.name),
                          size=size, sha256=source_hashes[video])
            manifest['splits'][name].append(record)
            reasons = []
            owners = split_owners.setdefault(record['sha256'], set())
            if owners and name not in owners:
                reasons.append('source_cross_split_sha256:' + ','.join(sorted(owners)))
            owners.add(name)
            if video in targets:
                reasons.append('resolved_path')
            if record['identity'] in target_ids:
                reasons.append('youtube_identity')
            for target in by_size.get(size, []):
                if target not in target_hashes:
                    target_hashes[target] = sha(target)
                if target_hashes[target] == record['sha256']:
                    reasons.append('sha256:' + str(target))
            if reasons:
                overlaps.append(dict(split=name, source=str(video), reasons=reasons))
    manifest['target_sha256_checked'] = {str(p): digest for p, digest in sorted(target_hashes.items())}
    if overlaps:
        atomic_json(path.with_name('overlap_rejected.json'), dict(dataset=manifest['dataset'], overlaps=overlaps))
        raise ValueError(f'Source/VideoMME overlap in train/val/test: {overlaps[:3]}')
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Immutable training_manifest.json changed; refusing resume')
    else:
        atomic_json(path, manifest)
    return manifest


@contextlib.contextmanager
def training_hooks(video, split, fallback, mme_items, work, audit, report):
    original_video, original_split = video.run, split.run

    def split_run(runtime):
        original_split(runtime)
        audit_overlap(runtime, mme_items, work / 'training_manifest.json')
        audit['overlap_audit'] = 'PASSED'
        report()

    def video_run(runtime):
        original_video(runtime)
        runtime.VideoReader = fallback.wrap_reader(runtime.VideoReader, audit, report)

    split.run, video.run = split_run, video_run
    try:
        yield
    finally:
        split.run, video.run = original_split, original_video


def execute(root, dataset):
    config = json.loads((root / 'experiment.json').read_text())
    work = root / 'work' / dataset
    work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / 'code/src'))
    os.environ['SEED_MODEL_ID'] = config['model']
    from seed_video.data.videomme import load_videomme
    video = importlib.import_module('seed_video.data.video')
    split = importlib.import_module('seed_video.data.experiment_split')
    cli = importlib.import_module('seed_video.cli')
    fallback = load_module(Path(__file__).with_name('decode_eof_fallback.py'), '_mme_eof')
    audit = dict(job=os.environ.get('PBS_JOBID'), dataset=dataset, state='STARTED',
                 decoder='decord_single_thread_opencv_sequential_eof',
                 fallback_sha256=sha(Path(fallback.__file__)), training_code=config['code'])
    audit_path = work / 'decoder_override' / f'{os.environ.get("PBS_JOBID", "local")}.json'
    report = lambda: atomic_json(audit_path, audit)
    report()
    try:
        # The normal core pipeline (including binary evaluation) must return successfully.
        with training_hooks(video, split, fallback, load_videomme(Path(config['data_root'])), work, audit, report):
            cli.main(train_command(root, config, dataset)[4:])
        adapter = exact_adapter(root, config, dataset)
        atomic_json(work / 'TRAIN_COMPLETE.json', dict(dataset=dataset, adapter=str(adapter),
                    adapter_sha256=sha(adapter), training_manifest_sha256=sha(work / 'training_manifest.json'),
                    max_train_steps=config['max_train_steps'], smoke=config['smoke'],
                    decoder_audit=str(audit_path), binary_pipeline_complete=True))
        audit['state'] = 'COMPLETE'
    except BaseException as error:
        audit.update(state='FAILED', error=str(error))
        raise
    finally:
        report()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--dataset', required=True, choices=('videohallucer', 'eventhallusion', 'mixed'))
    args = p.parse_args()
    execute(args.root.resolve(), args.dataset)


if __name__ == '__main__':
    main()
