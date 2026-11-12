def eval_items(runtime, items):
    runtime.model.eval()
    with runtime.torch.no_grad():
        return {runtime.key(i): runtime.margin(runtime.student_last(i, grad=False)) for i in items}


def eval_base_cached(runtime, items, path):
    """Resumable base evaluation, optionally shared by transfer jobs of one model."""
    import fcntl
    import os
    from pathlib import Path

    meta = dict(source=runtime.SOURCE_HASH, branch=runtime.branch_cfg('orig'),
                items=[[runtime.key(i), os.path.realpath(i['video']), i['question'], i['answer']]
                       for i in items])
    shared = os.environ.get('SEED_BASE_CACHE_DIR') if runtime.CFG.get('vh_eval_tasks') else None
    path = Path(shared) / f'base_{runtime.cfg_hash(meta)}.pt' if shared else Path(str(path) + '.partial.pt')
    path.parent.mkdir(parents=True, exist_ok=True)

    def read():
        if not path.exists():
            return {}
        state = runtime.torch.load(path, map_location='cpu', weights_only=False)
        if state['meta'] != meta:
            raise RuntimeError(f'Base cache metadata mismatch: {path}')
        return dict(state['marg'])

    marg = read()
    interval = int(os.environ.get('SEED_EVAL_CHECKPOINT_EVERY', '25'))
    if interval < 1:
        raise ValueError('SEED_EVAL_CHECKPOINT_EVERY must be positive')
    for start in range(0, len(items), interval):
        marg.update(read())
        pending = [i for i in items[start:start + interval] if runtime.key(i) not in marg]
        if not pending:
            continue
        computed = runtime.eval_items(pending)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            marg.update(read())
            for key, value in computed.items():
                marg.setdefault(key, value)
            temp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
            runtime.torch.save(dict(meta=meta, marg=marg), temp)
            os.replace(temp, path)
        print(f'BASE EVAL saved {len(marg)}/{len(items)}: {path}', flush=True)
    return marg

def eval_set(runtime):
    """Return each evaluation item once, including optional VideoHallucer holdouts."""
    items = ((runtime.TRAIN if runtime.CFG['eval_train'] else [])
             + runtime.VAL + runtime.TEST + runtime.EH
             + getattr(runtime, 'EXTRA_EVAL', []))
    if getattr(runtime.args, 'eval_videohallucer', False):
        items += getattr(runtime, 'VH_VAL', []) + getattr(runtime, 'VH_TEST', [])
    seen, out = set(), []
    for item in items:
        k = runtime.key(item)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out
