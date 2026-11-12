"""Explicit full-benchmark transfer protocol; no changes to legacy VH splits."""
import hashlib
import json
import os
from pathlib import Path

VH_TASKS = {
    'ORH': 'object_relation', 'TPH': 'temporal', 'SDH': 'semantic_detail',
    'EFH': 'external_factual', 'ENFH': 'external_nonfactual',
}


def write_manifest(path, manifest):
    """Reject split drift rather than replacing an established experiment."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f'Immutable manifest mismatch: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def content_digest(items):
    records = sorted((i['pair_id'], i['role'], i['question'], i['answer']) for i in items)
    return hashlib.sha256(json.dumps(records, ensure_ascii=True).encode()).hexdigest()


def check_video_overlap(runtime):
    """Reject exact-byte train/evaluation overlap; never silently remove samples."""
    train = {runtime.VIDEO_SHA[i['video']] for i in runtime.TRAIN}
    heldout = {runtime.VIDEO_SHA[i['video']] for i in runtime.VAL + runtime.TEST + runtime.VH_TEST}
    if train & heldout:
        raise ValueError('EH train overlaps EH/VH evaluation by video SHA256. '
                         'Review the datasets and create a new explicit manifest; no samples were removed.')


def full_vh_evaluation(runtime):
    """Use all requested VH pairs as target-only evaluation, never as train/val."""
    tasks = runtime.CFG['vh_eval_tasks']
    for task in tasks:
        subset = VH_TASKS[task]
        if not any(i['subset'] == subset for i in runtime.ITEMS):
            raise ValueError(f'VideoHallucer task {task} has no complete local pairs.')
    runtime.TRAIN, runtime.VAL, runtime.TEST = [], [], []
    runtime.VH_VAL, runtime.VH_TEST = [], list(runtime.ITEMS)
    runtime.ALL_VIDEOS = []  # Foreground features are needed only for EH teachers.
    runtime.failed = []
    manifest = dict(version='eh_to_vh_full_v1', configured_revision=runtime.CFG['hf_revision'],
                    tasks=tasks, digest=content_digest(runtime.ITEMS),
                    keys=[runtime.key(i) for i in runtime.ITEMS])
    write_manifest(runtime.WORK / 'videohallucer_eval_manifest.json', manifest)
    shared = os.environ.get('SEED_PROTOCOL_DIR')
    if shared:
        write_manifest(Path(shared) / 'videohallucer_eval_manifest.json', manifest)
    print('Full target-only VideoHallucer evaluation:',
          {task: sum(i['subset'] == VH_TASKS[task] for i in runtime.ITEMS) for task in tasks},
          flush=True)
