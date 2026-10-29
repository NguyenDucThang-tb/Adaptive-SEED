def pair_vec(runtime, marg, items):
    ok = {}
    for i in items:
        ok.setdefault(i['pair_id'], []).append((marg[runtime.key(i)] > 0) == (i['answer'] == 'yes'))
    pids = sorted(ok)
    return (pids, runtime.np.array([all(ok[p]) for p in pids], dtype=int))

def pair_acc(runtime, marg, items):
    return runtime.pair_vec(marg, items)[1].mean()
