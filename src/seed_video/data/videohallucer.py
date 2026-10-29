"""data: videohallucer stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType

def load_videohallucer(runtime, subsets):
    items, n_raw = ([], 0)
    for sub in subsets:
        seen = set()
        for jp in sorted(runtime.glob.glob(f'{runtime.ROOT}/{sub}/*.json')):
            for ex in runtime.json.load(open(jp)):
                n_raw += 1
                b, h = (ex['basic'], ex['hallucination'])
                sig = (b['video'], b['question'], h['video'], h['question'])
                if sig in seen:
                    continue
                seen.add(sig)
                if any(((sub, runtime.os.path.basename(q['video'])) not in runtime.VIDEO_INDEX for q in (b, h))):
                    continue
                pid = f'{sub}/{len(seen) - 1}'
                for role, q in (('basic', b), ('hallucination', h)):
                    rp = runtime.VIDEO_INDEX[sub, runtime.os.path.basename(q['video'])]
                    items.append(dict(subset=sub, pair_id=pid, role=role, repo_path=rp, video=runtime.os.path.join(runtime.ROOT, rp), question=q['question'], answer=q['answer'].strip().lower()))
    print(f"{n_raw} cặp gốc → {len({i['pair_id'] for i in items})} cặp sau khử trùng")
    return items

def key(runtime, it):
    return f"{it['pair_id']}|{it['role']}"

def take(runtime, cands, n, banned):
    ids, vids = ([], set())
    for pid in cands:
        if len(ids) >= n:
            break
        v = {i['video'] for i in runtime.pairs[pid]}
        if v & banned:
            continue
        ids.append(pid)
        vids |= v
    return (ids, vids)

def _local_repo_files(runtime):
    files = []
    for sub in runtime.CFG['subsets']:
        for dirpath, _, names in runtime.os.walk(f'{runtime.ROOT}/{sub}'):
            for n in names:
                files.append(runtime.os.path.relpath(runtime.os.path.join(dirpath, n), runtime.ROOT).replace('\\', '/'))
    return files

def _dl(runtime, p):
    return runtime.hf_hub_download(runtime.CFG['hf_repo'], p, repo_type='dataset', revision=runtime.CFG['hf_revision'], local_dir=runtime.ROOT, token=runtime.CFG['hf_token'])

def run(runtime):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    runtime.ThreadPoolExecutor = ThreadPoolExecutor
    runtime.as_completed = as_completed
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    runtime.HfApi = HfApi
    runtime.hf_hub_download = hf_hub_download
    runtime.snapshot_download = snapshot_download
    runtime.ROOT = runtime.CFG['data_root']
    runtime.os.makedirs(runtime.ROOT, exist_ok=True)
    runtime.OFFLINE = runtime.os.environ.get('HF_HUB_OFFLINE') == '1'
    if runtime.OFFLINE:
        runtime.REPO_FILES = _local_repo_files(runtime)
        print(f'HF_HUB_OFFLINE=1: dùng dữ liệu local trong {runtime.ROOT} ({len(runtime.REPO_FILES)} file)')
    else:
        runtime.snapshot_download(repo_id=runtime.CFG['hf_repo'], repo_type='dataset', revision=runtime.CFG['hf_revision'], allow_patterns=[f'{s}/*.json' for s in runtime.CFG['subsets']], local_dir=runtime.ROOT, token=runtime.CFG['hf_token'])
        runtime.REPO_FILES = runtime.HfApi().list_repo_files(runtime.CFG['hf_repo'], repo_type='dataset', revision=runtime.CFG['hf_revision'], token=runtime.CFG['hf_token'])
    runtime.VIDEO_INDEX = {}
    for runtime.f in runtime.REPO_FILES:
        if runtime.f.lower().endswith(('.mp4', '.mkv', '.webm', '.avi', '.mov')):
            runtime.VIDEO_INDEX.setdefault((runtime.f.split('/')[0], runtime.os.path.basename(runtime.f)), runtime.f)
    runtime.load_videohallucer = MethodType(load_videohallucer, runtime)
    runtime.key = MethodType(key, runtime)
    runtime.ITEMS = runtime.load_videohallucer(runtime.CFG['subsets'])
    runtime.pairs = {}
    for runtime.it in runtime.ITEMS:
        runtime.pairs.setdefault(runtime.it['pair_id'], []).append(runtime.it)
    runtime.pair_ids = sorted(runtime.pairs)
    runtime.random.Random(runtime.CFG['split_seed']).shuffle(runtime.pair_ids)
    runtime.take = MethodType(take, runtime)
    runtime.train_ids, runtime.train_videos = runtime.take(runtime.pair_ids, runtime.CFG['n_train_pairs'], set())
    runtime.rest = [p for p in runtime.pair_ids if p not in set(runtime.train_ids)]
    runtime.val_ids, runtime.val_videos = runtime.take(runtime.rest, runtime.CFG['n_val_pairs'], runtime.train_videos)
    runtime.rest = [p for p in runtime.rest if p not in set(runtime.val_ids)]
    runtime.test_ids, runtime._ = runtime.take(runtime.rest, runtime.CFG['n_test_pairs'], runtime.train_videos | runtime.val_videos)
    runtime.TRAIN = [i for p in runtime.train_ids for i in runtime.pairs[p]]
    runtime.VAL = [i for p in runtime.val_ids for i in runtime.pairs[p]]
    runtime.TEST = [i for p in runtime.test_ids for i in runtime.pairs[p]]
    runtime._v = lambda S: {i['video'] for i in S}
    assert not runtime._v(runtime.TRAIN) & runtime._v(runtime.VAL) and (not runtime._v(runtime.TRAIN) & runtime._v(runtime.TEST)) and (not runtime._v(runtime.VAL) & runtime._v(runtime.TEST)), 'train/val/test chung video'
    if len(runtime.val_ids) < runtime.CFG['n_val_pairs']:
        print(f"⚠ VAL chỉ có {len(runtime.val_ids)} cặp (yêu cầu {runtime.CFG['n_val_pairs']}) do trùng video với TRAIN")
    assert len(runtime.VAL) and len(runtime.TEST), 'VAL hoặc TEST rỗng → giảm n_train_pairs / n_val_pairs hoặc thêm subset'
    runtime.need = sorted({i['repo_path'] for i in runtime.TRAIN + runtime.VAL + runtime.TEST})
    runtime.todo = [p for p in runtime.need if not runtime.os.path.exists(runtime.os.path.join(runtime.ROOT, p))]
    if runtime.OFFLINE and runtime.todo:
        raise RuntimeError(f'Offline nhưng thiếu {len(runtime.todo)} video, ví dụ {runtime.todo[:3]}')
    runtime._dl = MethodType(_dl, runtime)
    runtime.failed = []
    with runtime.ThreadPoolExecutor(runtime.CFG['download_workers']) as runtime.ex:
        runtime.futs = {runtime.ex.submit(runtime._dl, p): p for p in runtime.todo}
        for runtime.fu in runtime.as_completed(runtime.futs):
            try:
                runtime.fu.result()
            except Exception as e:
                runtime.e = e
                runtime.failed.append(runtime.futs[runtime.fu])
    runtime.bad = {runtime.os.path.join(runtime.ROOT, p) for p in runtime.failed}
    if runtime.bad:
        runtime.badp = {i['pair_id'] for i in runtime.TRAIN + runtime.VAL + runtime.TEST if i['video'] in runtime.bad}
        runtime.TRAIN, runtime.VAL, runtime.TEST = ([i for i in S if i['pair_id'] not in runtime.badp] for S in (runtime.TRAIN, runtime.VAL, runtime.TEST))
    runtime.ALL_VIDEOS = sorted({i['video'] for i in runtime.TRAIN + runtime.VAL + runtime.TEST})
    print(f'train {len(runtime.TRAIN) // 2} cặp | val {len(runtime.VAL) // 2} cặp | test {len(runtime.TEST) // 2} cặp | {len(runtime.ALL_VIDEOS)} video | tải lỗi {len(runtime.failed)}')
    print('GT yes-rate val / test:', runtime.np.mean([i['answer'] == 'yes' for i in runtime.VAL]).round(3), '/', runtime.np.mean([i['answer'] == 'yes' for i in runtime.TEST]).round(3))
