"""Create deterministic, video-disjoint splits for EventHallusion training."""
from collections import defaultdict
import hashlib
import random
from types import MethodType


def split_eventhallusion(items, train_fraction=0.7, val_fraction=0.1, seed=42):
    if not items:
        raise ValueError('EventHallusion contains no resolved QA items.')
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1
            and train_fraction + val_fraction < 1):
        raise ValueError('train_fraction and val_fraction must be positive and sum to less than 1.')

    by_video = defaultdict(list)
    video_subsets = defaultdict(set)
    for item in items:
        by_video[item['video']].append(item)
        video_subsets[item['video']].add(item['subset'])

    strata = defaultdict(list)
    for video, subsets in video_subsets.items():
        strata[tuple(sorted(subsets))].append(video)

    train_videos, val_videos, test_videos = (set(), set(), set())
    for stratum, videos in sorted(strata.items()):
        if len(videos) < 3:
            raise ValueError(
                f'Need at least 3 distinct EventHallusion videos per stratum; '
                f'{stratum} has {len(videos)}.'
            )
        stratum_seed = int(hashlib.sha256(repr((seed, stratum)).encode()).hexdigest()[:8], 16)
        videos.sort()
        random.Random(stratum_seed).shuffle(videos)
        n_train = max(1, int(round(len(videos) * train_fraction)))
        n_val = max(1, int(round(len(videos) * val_fraction)))
        if n_train + n_val >= len(videos):
            n_train = len(videos) - 2
            n_val = 1
        train_videos.update(videos[:n_train])
        val_videos.update(videos[n_train:n_train + n_val])
        test_videos.update(videos[n_train + n_val:])

    if train_videos & val_videos or train_videos & test_videos or val_videos & test_videos:
        raise AssertionError('EventHallusion video leakage between splits.')

    def rows(video_ids):
        return [item for item in items if item['video'] in video_ids]

    train, val, test = rows(train_videos), rows(val_videos), rows(test_videos)
    if not train or not val or not test:
        raise ValueError('EventHallusion split produced an empty train, validation, or test set.')
    return train, val, test


def _take_videos(items, count):
    selected = []
    videos = set()
    for item in items:
        if item['video'] in videos:
            continue
        videos.add(item['video'])
        selected.extend(row for row in items if row['video'] == item['video'])
        if len(videos) == count:
            break
    return selected


def configure_eval_splits(runtime):
    if getattr(runtime.args, 'train_dataset', 'videohallucer') == 'eventhallusion':
        splits = {'EH_VAL': runtime.EH_VAL, 'EH_TEST': runtime.EH_TEST}
        for subset in sorted({item['subset'] for item in runtime.EH_TEST}):
            short = subset.removeprefix('eh_')
            splits[f'eh_{short}_TEST'] = [item for item in runtime.EH_TEST if item['subset'] == subset]
        if getattr(runtime.args, 'eval_videohallucer', False):
            full_tasks = getattr(runtime, 'CFG', {}).get('vh_eval_tasks', [])
            if not getattr(runtime, 'VH_TEST', None) or (not full_tasks and not getattr(runtime, 'VH_VAL', None)):
                raise ValueError('VideoHallucer evaluation was requested, but VH_VAL/VH_TEST are unavailable.')
            if runtime.VH_VAL:
                splits['VH_VAL'] = runtime.VH_VAL
            splits['VH_TEST'] = runtime.VH_TEST
            if full_tasks:
                from seed_video.data.transfer import VH_TASKS
                for task in full_tasks:
                    splits[f'VH_{task}_TEST'] = [i for i in runtime.VH_TEST if i['subset'] == VH_TASKS[task]]
        return splits

    splits = {'VH_VAL': runtime.VAL, 'VH_TEST': runtime.TEST, 'EH_ALL': runtime.EH}
    for subset in sorted({item['subset'] for item in runtime.EH}):
        splits[subset] = [item for item in runtime.EH if item['subset'] == subset]
    return splits


def run(runtime):
    runtime.configure_eval_splits = MethodType(configure_eval_splits, runtime)
    if getattr(runtime.args, 'train_dataset', 'videohallucer') != 'eventhallusion':
        return
    args = runtime.args
    all_items = list(runtime.EH)
    train, val, test = split_eventhallusion(
        all_items,
        train_fraction=args.eh_train_fraction,
        val_fraction=args.eh_val_fraction,
        seed=args.eh_split_seed,
    )
    smoke_count = getattr(args, 'smoke_pairs', None)
    if runtime.CFG.get('vh_eval_tasks') and smoke_count is None:
        from seed_video.data.transfer import content_digest, write_manifest
        from pathlib import Path
        portable = dict(version='eh_to_vh_full_v1', seed=args.eh_split_seed,
                        fractions=[args.eh_train_fraction, args.eh_val_fraction],
                        digest=content_digest(all_items),
                        train=sorted(runtime.key(i) for i in train),
                        val=sorted(runtime.key(i) for i in val),
                        test=sorted(runtime.key(i) for i in test))
        write_manifest(runtime.WORK / 'eh_transfer_manifest.json', portable)
        shared = runtime.os.environ.get('SEED_PROTOCOL_DIR')
        if shared:
            write_manifest(Path(shared) / 'eh_transfer_manifest.json', portable)
    if smoke_count is not None:
        train = _take_videos(train, smoke_count)
        val = _take_videos(val, 1)
        test = _take_videos(test, 1)

    runtime.EH_ALL_ITEMS = all_items
    runtime.EH_TRAIN, runtime.EH_VAL, runtime.EH_TEST = train, val, test
    runtime.TRAIN, runtime.VAL, runtime.TEST = train, val, test
    runtime.EH = test
    runtime.EH_SPLITS = sorted({item['subset'] for item in test})
    runtime.CFG['eval_train'] = False
    runtime.EXTRA_EVAL = []

    train_videos = {item['video'] for item in train}
    val_videos = {item['video'] for item in val}
    test_videos = {item['video'] for item in test}
    # Foreground masks are only needed for videos in the (possibly smoke-trimmed) splits.
    runtime.ALL_VIDEOS = sorted(train_videos | val_videos | test_videos)
    manifest = {
        'dataset': 'eventhallusion',
        'seed': args.eh_split_seed,
        'train_fraction': args.eh_train_fraction,
        'val_fraction': args.eh_val_fraction,
        'test_fraction': 1.0 - args.eh_train_fraction - args.eh_val_fraction,
        'train_items': [runtime.key(item) for item in train],
        'val_items': [runtime.key(item) for item in val],
        'test_items': [runtime.key(item) for item in test],
        'train_videos': sorted(train_videos),
        'val_videos': sorted(val_videos),
        'test_videos': sorted(test_videos),
    }
    path = runtime.WORK / 'eventhallusion_train_val_test.json'
    temp = path.with_name(path.name + f'.tmp.{runtime.os.getpid()}')
    with temp.open('w', encoding='utf-8') as handle:
        runtime.json.dump(manifest, handle, indent=2)
    runtime.os.replace(temp, path)

    print(
        'EventHallusion video-disjoint split:',
        f"train={len(train)} QA/{len(train_videos)} videos,",
        f"val={len(val)} QA/{len(val_videos)} videos,",
        f"test={len(test)} QA/{len(test_videos)} videos",
    )
    print('Held-out test subsets:', runtime.EH_SPLITS)
