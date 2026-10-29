"""experiments: answer_tokens stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType

def sr_target(runtime, o, p, n, beta, answer_only, agreement, pair_factor):
    o = o.detach().float()
    p = p.detach().float()
    n = n.detach().float()
    dp = p - o
    dn = o - n
    up = float(dp[runtime.YES] - dp[runtime.NO])
    un = float(dn[runtime.YES] - dn[runtime.NO])
    product = up * un
    agree = max(product, 0.0) / (abs(product) + runtime.SR['eps'])
    ngate = (agree if agreement else 1.0) * pair_factor
    residual = runtime.ap0 * dp + runtime.an0 * ngate * dn
    if answer_only:
        dm = beta * (residual[runtime.YES] - residual[runtime.NO])
        target = o.clone()
        target[runtime.YES] += dm / 2
        target[runtime.NO] -= dm / 2
    else:
        target = o + beta * residual
    assert runtime.torch.isfinite(target).all()
    return (target, {'u_pos': up, 'u_neg': un, 'agreement': agree, 'negative_gate': ngate})

def run(runtime):
    from pathlib import Path
    runtime.Path = Path
    import numpy as np
    runtime.np = np
    import pandas as pd
    runtime.pd = pd
    import torch
    runtime.torch = torch
    import gc
    runtime.gc = gc
    runtime.required = ['SHARED', 'TARGETS_ABL', 'BASE_TEACHER', 'ARMS', 'METHODS', 'component_step', 'train_arm', 'result_rows', 'MARG', 'STUDENT_DIR', 'seed_beta', 'TRAIN', 'CFG']
    runtime.missing = [x for x in runtime.required if x not in vars(runtime)]
    assert not runtime.missing, f'Runtime thiếu {runtime.missing}. Cần nạp notebook ablation trước.'
    assert not hasattr(runtime.model, 'peft_config'), 'Cần unload adapter trước.'
    assert 'base' in runtime.MARG
    assert runtime.YES != runtime.NO
    runtime.SR = {'version': 'selective_residual_v1', 'seed': 0, 'eps': 1e-08, 'negative': 'tcd'}
    runtime.SR_ARMS = {'answer_token_only': (True, False, False), 'agreement_only': (False, True, False), 'answer_token_agreement': (True, True, False), 'answer_token_agreement_pair': (True, True, True)}
    runtime.sr_root = runtime.Path(runtime.CACHE) / runtime.SR['version']
    runtime.sr_root.mkdir(parents=True, exist_ok=True)
    runtime.ap0 = float(runtime.BASE_TEACHER['a_pos'])
    runtime.an0 = float(runtime.BASE_TEACHER['negs']['tcd'])
    print('Positive: orig+fg | Negative: TCD')
    print('a_pos:', runtime.ap0, '| a_neg:', runtime.an0)
    print('Epochs:', runtime.CFG['epochs'], '| seed:', runtime.SR['seed'])
    print('New arms:', list(runtime.SR_ARMS))
    runtime.sr_target = MethodType(sr_target, runtime)
    for runtime.it in runtime.TRAIN[:5]:
        runtime.k = runtime.key(runtime.it)
        runtime.r = runtime.SHARED[runtime.k]
        runtime.o, runtime.p, runtime.n = (runtime.r['orig'], runtime.r['orig+fg'], runtime.r['tcd'])
        runtime.beta = runtime.seed_beta(runtime.k)
        runtime.ref, runtime._ = runtime.sr_target(runtime.o, runtime.p, runtime.n, runtime.beta, False, False, 1.0)
        assert runtime.torch.allclose(runtime.ref, runtime.TARGETS_ABL['seed_base'][runtime.k], atol=1e-05, rtol=1e-05), 'SEED target hiện tại không khớp cache/config.'
        runtime.selected, runtime._ = runtime.sr_target(runtime.o, runtime.p, runtime.n, runtime.beta, True, False, 1.0)
        assert runtime.torch.allclose(runtime.selected[runtime.YES] - runtime.selected[runtime.NO], runtime.ref[runtime.YES] - runtime.ref[runtime.NO], atol=0.0001, rtol=1e-05)
        runtime.outside = runtime.torch.ones_like(runtime.o, dtype=runtime.torch.bool)
        runtime.outside[runtime.YES] = False
        runtime.outside[runtime.NO] = False
        assert runtime.torch.equal(runtime.selected[runtime.outside], runtime.o.float()[runtime.outside])
    print('Target algebra checks: OK')
    runtime.sr_audit = []
    for runtime.name, runtime.flags in runtime.SR_ARMS.items():
        runtime.answer_only, runtime.agreement, runtime.use_pair = runtime.flags
        runtime.TARGETS_ABL[runtime.name] = {}
        runtime.ARMS[runtime.name] = {**runtime.SR, 'answer_only': runtime.answer_only, 'agreement': runtime.agreement, 'pair': runtime.use_pair, 'positive': 'orig+fg', 'a_pos': runtime.ap0, 'a_neg': runtime.an0}
        for runtime.it in runtime.TRAIN:
            runtime.k = runtime.key(runtime.it)
            runtime.r = runtime.SHARED[runtime.k]
            runtime.eligible = bool(runtime.r.get('pair_eligible', False))
            runtime.pair_factor = float(runtime.r['pair'][False][0]) if runtime.use_pair and runtime.eligible else 1.0
            assert runtime.np.isfinite(runtime.pair_factor) and 0 <= runtime.pair_factor <= 1
            runtime.q_logits, runtime.info = runtime.sr_target(runtime.r['orig'], runtime.r['orig+fg'], runtime.r['tcd'], runtime.seed_beta(runtime.k), runtime.answer_only, runtime.agreement, runtime.pair_factor)
            runtime.TARGETS_ABL[runtime.name][runtime.k] = runtime.q_logits
            runtime.sr_audit.append({'arm': runtime.name, 'id': runtime.k, 'pair_eligible': runtime.eligible, 'pair_factor': runtime.pair_factor, **runtime.info, 'base_margin': runtime.margin(runtime.r['orig']), 'target_margin': runtime.margin(runtime.q_logits), 'mean_abs_target_change': float((runtime.q_logits - runtime.r['orig']).abs().mean())})
        runtime.METHODS[runtime.name] = (lambda name=runtime.name: runtime.component_step(name), None, None, True)
    runtime.sr_audit = runtime.pd.DataFrame(runtime.sr_audit)
    runtime.sr_audit.to_csv(runtime.sr_root / 'target_diagnostics_train.csv', index=False)
    runtime.display(runtime.sr_audit.groupby('arm')[['agreement', 'negative_gate', 'mean_abs_target_change']].mean())
    runtime.seed = runtime.SR['seed']
    runtime.baseline_tag = f'seed_base_s{runtime.seed}'
    if runtime.baseline_tag not in runtime.MARG:
        runtime.baseline_out = runtime.train_arm('seed_base', runtime.seed)
        runtime.MARG[runtime.baseline_tag] = runtime.baseline_out['marg']
    runtime.sr_rows = runtime.result_rows('base', runtime.MARG['base']) + runtime.result_rows(runtime.baseline_tag, runtime.MARG[runtime.baseline_tag])
    runtime.original_student_dir = runtime.STUDENT_DIR
    try:
        runtime.STUDENT_DIR = str(runtime.sr_root / 'students')
        runtime.Path(runtime.STUDENT_DIR).mkdir(parents=True, exist_ok=True)
        for runtime.name in runtime.SR_ARMS:
            runtime.out = runtime.train_arm(runtime.name, runtime.seed)
            runtime.tag = f'{runtime.name}_s{runtime.seed}'
            runtime.MARG[runtime.tag] = runtime.out['marg']
            runtime.current_rows = runtime.result_rows(runtime.tag, runtime.out['marg'])
            runtime.sr_rows.extend(runtime.current_rows)
            runtime.display(runtime.pd.DataFrame(runtime.current_rows))
            runtime.pd.DataFrame(runtime.sr_rows).to_csv(runtime.sr_root / 'results.csv', index=False)
            runtime.pd.DataFrame([{'id': runtime.key(it), 'video': it['video'], 'question': it['question'], 'answer': it['answer'], 'base_margin': runtime.MARG['base'][runtime.key(it)], 'seed_margin': runtime.MARG[runtime.baseline_tag][runtime.key(it)], 'student_margin': runtime.out['marg'][runtime.key(it)]} for it in runtime.eval_set()]).to_csv(runtime.sr_root / f'{runtime.tag}_predictions.csv', index=False)
            runtime.gc.collect()
            runtime.torch.cuda.empty_cache()
    finally:
        runtime.STUDENT_DIR = runtime.original_student_dir
    runtime.sr_report = runtime.pd.DataFrame(runtime.sr_rows)
    runtime.seed_reference = runtime.sr_report[runtime.sr_report['run'] == runtime.baseline_tag].set_index('split')
    for runtime.metric in ['q_acc', 'pair_acc']:
        runtime.sr_report[f'{runtime.metric}_delta_vs_seed_pp'] = 100 * (runtime.sr_report[runtime.metric] - runtime.sr_report['split'].map(runtime.seed_reference[runtime.metric]))
    runtime.sr_report.to_csv(runtime.sr_root / 'results.csv', index=False)
    runtime.display(runtime.sr_report[runtime.sr_report['split'].isin(['VH_VAL', 'VH_TEST', 'EH_ALL'])][['run', 'split', 'n', 'q_acc', 'pair_acc', 'yes_rate', 'q_acc_delta_vs_seed_pp', 'pair_acc_delta_vs_seed_pp', 'wrong_to_correct', 'correct_to_wrong']])
    print('Saved:', runtime.sr_root)
