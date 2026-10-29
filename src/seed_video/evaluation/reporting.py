def result_rows(runtime, run, m):
    rows = []
    for split, items in runtime.SPLITS.items():
        if not items:
            continue
        old = runtime.np.array([(runtime.MARG['base'][runtime.key(i)] > 0) == (i['answer'] == 'yes') for i in items])
        new = runtime.np.array([(m[runtime.key(i)] > 0) == (i['answer'] == 'yes') for i in items])
        rows.append(dict(run=run, split=split, n=len(items), q_acc=float(new.mean()), pair_acc=runtime.pair_acc(m, items) if split.startswith('VH') else runtime.np.nan, yes_rate=float(runtime.np.mean([m[runtime.key(i)] > 0 for i in items])), delta_pp=100 * float(new.mean() - old.mean()), wrong_to_correct=int((~old & new).sum()), correct_to_wrong=int((old & ~new).sum()), mean_abs_margin_change=float(runtime.np.mean([abs(m[runtime.key(i)] - runtime.MARG['base'][runtime.key(i)]) for i in items]))))
    return rows
