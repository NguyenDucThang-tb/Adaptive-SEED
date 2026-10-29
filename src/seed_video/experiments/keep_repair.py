"""experiments: keep_repair stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType

def atpk_cleanup(runtime):
    runtime.gc.collect()
    if runtime.torch.cuda.is_available():
        runtime.torch.cuda.empty_cache()

def atpk_margin(runtime, z):
    return float((z[runtime.YES] - z[runtime.NO]).detach().cpu())

def atpk_clip01(runtime, value):
    return min(1.0, max(0.0, float(value)))

def atpk_spec(runtime, adaptive_pos=False, adaptive_neg=False):
    return {'answer_token': True, 'visual': True, 'pair': True, 'keep_repair': True, 'adaptive_pos': adaptive_pos, 'adaptive_neg': adaptive_neg, 'caption': False, 'null': False, 'reverse': False, 'agreement': False}

def atpk_pair_gate(runtime, rec):
    if not rec['pair_eligible']:
        return 1.0
    value = rec['pair'][True]
    if isinstance(value, runtime.torch.Tensor):
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, (list, tuple, runtime.np.ndarray)):
        value = value[0]
    value = float(value)
    assert runtime.np.isfinite(value) and 0 <= value <= 1, value
    return value

def atpk_target(runtime, item, spec):
    k = runtime.key(item)
    rec = runtime.SHARED[k]
    o = rec['orig'].detach().float().cpu()
    p = rec['selected'].detach().float().cpu()
    n = rec['tcd'].detach().float().cpu()
    assert o.ndim == 1 and o.shape == p.shape == n.shape
    assert 0 <= runtime.YES < o.numel() and 0 <= runtime.NO < o.numel()
    assert all((runtime.torch.isfinite(z).all().item() for z in (o, p, n)))
    gt = runtime.YES if item['answer'] == 'yes' else runtime.NO
    other = runtime.NO if gt == runtime.YES else runtime.YES
    dp, dn = (p - o, o - n)
    pos_gain = float(dp[gt] - dp[other])
    neg_gain = float(dn[gt] - dn[other])
    pos_scale = runtime.atpk_clip01(pos_gain / runtime.GAIN_TAU) if spec['adaptive_pos'] else 1.0
    neg_scale = runtime.atpk_clip01(runtime.AN0 * neg_gain / runtime.GAIN_TAU) if spec['adaptive_neg'] else 1.0
    ap = runtime.AP0 * pos_scale
    an_before_gate = runtime.AN0 * neg_scale
    gate = runtime.atpk_pair_gate(rec)
    an = an_before_gate * gate
    residual = ap * dp + an * dn
    beta = float(runtime.seed_beta(k))
    assert runtime.np.isfinite(beta)
    dm = beta * float(residual[runtime.YES] - residual[runtime.NO])
    candidate = o.clone()
    candidate[runtime.YES] += dm / 2
    candidate[runtime.NO] -= dm / 2
    gt_yes = item['answer'] == 'yes'
    base_ok = (runtime.atpk_margin(o) > 0) == gt_yes
    candidate_ok = (runtime.atpk_margin(candidate) > 0) == gt_yes
    if base_ok:
        target, weight, route = (o.clone(), 1.0, 'keep')
    elif candidate_ok:
        target, weight, route = (candidate, 1.0, 'repair')
    else:
        target, weight, route = (candidate, 0.0, 'ce_only')
    final_ok = (runtime.atpk_margin(target) > 0) == gt_yes
    assert runtime.torch.isfinite(target).all()
    if route == 'keep':
        assert runtime.torch.equal(target, o)
    audit = {'id': k, 'route': route, 'kd_weight': weight, 'a_pos': ap, 'a_neg_before_gate': an_before_gate, 'a_neg': an, 'pair_gate': gate, 'pair_eligible': bool(rec['pair_eligible']), 'pos_scale': pos_scale, 'neg_scale': neg_scale, 'base_correct': base_ok, 'candidate_correct': candidate_ok, 'target_correct': final_ok, 'base_margin': runtime.atpk_margin(o), 'candidate_margin': runtime.atpk_margin(candidate), 'target_margin': runtime.atpk_margin(target)}
    return (target, weight, audit)

def atpk_step(runtime, arm):

    def step(item, ctx):
        k = runtime.key(item)
        s = runtime.student_last(item)
        t = runtime.ATPK_TARGETS[arm][k].to(s.device)
        weight = runtime.ATPK_KD_WEIGHT[arm, k]
        kd = runtime.rkl(t[None], s[None]) * weight if weight > 0 else s.sum() * 0
        rng_seed = int(runtime.seed_of(k) + ctx['g'])
        draw = runtime.random.Random(rng_seed).random()
        if weight > 0 and draw < runtime.CFG['seed_gamma']:
            generator = runtime.torch.Generator(device=s.device)
            generator.manual_seed(rng_seed)
            y = runtime.torch.multinomial(t.softmax(0), 1, generator=generator)
        else:
            y = runtime.torch.tensor([runtime.gt_tok(item)], device=s.device, dtype=runtime.torch.long)
        ce = runtime.F.cross_entropy(s[None].float(), y.view(1))
        loss = kd + ce
        return (loss, {'rkl': float(kd.detach()), 'ce': float(ce.detach())})
    return step

def atpk_student_cfg(runtime, arm, seed):
    return {'version': runtime.ATPK['version'], 'code': runtime.ATPK_CODE_HASH, 'source': runtime.SOURCE_HASH, 'config': runtime.ATPK, 'spec': runtime.ATPK_SPECS[arm], 'seed': seed, 'teacher': runtime.BASE_TEACHER, 'abl': runtime.ABL, 'lora_targets': runtime.LORA_TARGETS, 'hp': {name: runtime.CFG[name] for name in runtime.STUDENT_HP}, 'seed_gamma': runtime.CFG['seed_gamma'], 'train': [runtime.item_sig(i) for i in runtime._ATPK_ITEMS], 'train_answers': [(runtime.key(i), i['answer']) for i in runtime._ATPK_ITEMS], 'evaluation': [runtime.item_sig(i) for i in runtime._ATPK_EVAL], 'targets': runtime.ATPK_TARGET_HASH[arm], 'weights': [runtime.ATPK_KD_WEIGHT[arm, runtime.key(i)] for i in runtime._ATPK_ITEMS], 'betas': [float(runtime.seed_beta(runtime.key(i))) for i in runtime._ATPK_ITEMS]}

def run(runtime):
    from pathlib import Path
    runtime.Path = Path
    import gc
    runtime.gc = gc
    import hashlib
    runtime.hashlib = hashlib
    import json
    runtime.json = json
    import random
    runtime.random = random
    import time
    runtime.time = time
    import warnings
    runtime.warnings = warnings
    import numpy as np
    runtime.np = np
    import pandas as pd
    runtime.pd = pd
    import torch
    runtime.torch = torch
    import torch.nn.functional as F
    runtime.F = F
    from tqdm.auto import tqdm
    runtime.tqdm = tqdm
    runtime._REQUIRED = ['model', 'TRAIN', 'SHARED', 'BASE_TEACHER', 'ABL', 'METHODS', 'MARG', 'train_arm', 'result_rows', 'student_last', 'rkl', 'gt_tok', 'seed_beta', 'seed_of', 'key', 'item_sig', 'eval_set', 'SOURCE_HASH', 'LORA_TARGETS', 'CFG', 'STUDENT_HP', 'STUDENT_DIR', 'student_cfg', 'YES', 'NO', 'CACHE']
    runtime._missing = [name for name in runtime._REQUIRED if name not in vars(runtime)]
    assert not runtime._missing, 'Cần chạy các cell setup/SEED trước. Thiếu: ' + ', '.join(runtime._missing)
    assert runtime.TRAIN, 'TRAIN rỗng.'
    assert 'base' in runtime.MARG, 'Thiếu kết quả base trong MARG.'
    assert not hasattr(runtime.model, 'peft_config'), 'Model còn adapter. Hãy hoàn tất/unload run trước.'
    assert not any(('lora_' in n for n, _ in runtime.model.named_parameters())), 'Model vẫn chứa LoRA; không bắt đầu arm mới trên adapter cũ.'
    assert runtime.YES != runtime.NO
    runtime.atpk_cleanup = MethodType(atpk_cleanup, runtime)
    runtime.atpk_margin = MethodType(atpk_margin, runtime)
    runtime.atpk_clip01 = MethodType(atpk_clip01, runtime)
    runtime._ATPK_ITEMS = list(runtime.TRAIN)
    runtime._ATPK_EVAL = list(runtime.eval_set())
    runtime._train_keys = [runtime.key(i) for i in runtime._ATPK_ITEMS]
    runtime._eval_keys = [runtime.key(i) for i in runtime._ATPK_EVAL]
    assert len(set(runtime._train_keys)) == len(runtime._train_keys), 'Trùng key TRAIN.'
    assert len(set(runtime._eval_keys)) == len(runtime._eval_keys), 'Trùng key evaluation.'
    assert all((k in runtime.MARG['base'] for k in runtime._eval_keys)), 'Base margins chưa phủ đủ evaluation.'
    runtime.AP0 = float(runtime.BASE_TEACHER['a_pos'])
    runtime.AN0 = float(runtime.BASE_TEACHER['negs']['tcd'])
    runtime.GAIN_TAU = float(runtime.ABL['gain_tau'])
    assert runtime.np.isfinite([runtime.AP0, runtime.AN0, runtime.GAIN_TAU]).all()
    assert runtime.AP0 >= 0 and runtime.AN0 >= 0 and (runtime.GAIN_TAU > 0)
    assert 0 <= float(runtime.CFG['seed_gamma']) <= 1
    for runtime.item in runtime._ATPK_ITEMS:
        runtime.k = runtime.key(runtime.item)
        assert runtime.item.get('answer') in ('yes', 'no'), f'Thiếu nhãn: {runtime.k}'
        assert runtime.k in runtime.SHARED, f'Thiếu SHARED: {runtime.k}'
        runtime.rec = runtime.SHARED[runtime.k]
        for runtime.field in ('orig', 'selected', 'tcd', 'pair_eligible'):
            assert runtime.field in runtime.rec, f'{runtime.k}: thiếu {runtime.field}'
        if runtime.rec['pair_eligible']:
            assert 'pair' in runtime.rec and True in runtime.rec['pair'], f'{runtime.k}: thiếu pair gate cho selected positive'
    runtime.ATPK = {'version': 'answer_pair_keep_no_caption_v2', 'seeds': [0], 'caption': False, 'null': False, 'reverse': False, 'agreement': False, 'adaptive_neg_formula': 'AN0 * clip01(AN0 * gain / tau)'}
    runtime.atpk_spec = MethodType(atpk_spec, runtime)
    runtime.ATPK_SPECS = {'atpk_fixed': runtime.atpk_spec(), 'atpk_adaptive_pos': runtime.atpk_spec(adaptive_pos=True), 'atpk_adaptive_neg': runtime.atpk_spec(adaptive_neg=True), 'atpk_adaptive_both': runtime.atpk_spec(True, True)}
    runtime.ATPK_RUN = list(runtime.ATPK_SPECS)
    runtime.ATPK_ROOT = runtime.Path(runtime.CACHE) / runtime.ATPK['version']
    runtime.ATPK_ROOT.mkdir(parents=True, exist_ok=True)
    print('Output:', runtime.ATPK_ROOT)
    print('Student frames:', runtime.CFG.get('n_frames', 'theo CFG'))
    print("Positive: SHARED['selected']; negative: TCD")
    print('Caption / null / reverse: OFF')
    print('Pair mode:', runtime.ABL.get('pair_mode', 'theo cached SHARED'))
    print('a_pos:', runtime.AP0, '| a_neg:', runtime.AN0, '| gain_tau:', runtime.GAIN_TAU)
    print('Epochs:', runtime.CFG['epochs'], '| Seeds:', runtime.ATPK['seeds'])
    runtime.display(runtime.pd.DataFrame.from_dict(runtime.ATPK_SPECS, orient='index'))
    runtime.atpk_pair_gate = MethodType(atpk_pair_gate, runtime)
    runtime.atpk_target = MethodType(atpk_target, runtime)
    runtime.ATPK_TARGETS, runtime.ATPK_KD_WEIGHT, runtime.ATPK_AUDIT = ({}, {}, [])
    for runtime.arm in runtime.ATPK_RUN:
        runtime.ATPK_TARGETS[runtime.arm] = {}
        for runtime.item in runtime.tqdm(runtime._ATPK_ITEMS, desc=f'Targets {runtime.arm}'):
            runtime.k = runtime.key(runtime.item)
            runtime.target, runtime.weight, runtime.audit = runtime.atpk_target(runtime.item, runtime.ATPK_SPECS[runtime.arm])
            runtime.ATPK_TARGETS[runtime.arm][runtime.k] = runtime.target
            runtime.ATPK_KD_WEIGHT[runtime.arm, runtime.k] = runtime.weight
            runtime.ATPK_AUDIT.append({'arm': runtime.arm, **runtime.audit})
    runtime.audit_df = runtime.pd.DataFrame(runtime.ATPK_AUDIT)
    runtime.audit_df.to_csv(runtime.ATPK_ROOT / 'target_diagnostics_train.csv', index=False)
    runtime.display(runtime.audit_df.groupby(['arm', 'route'], as_index=False).agg(n=('id', 'size'), mean_a_pos=('a_pos', 'mean'), mean_a_neg=('a_neg', 'mean'), mean_pair_gate=('pair_gate', 'mean'), base_acc=('base_correct', 'mean'), candidate_acc=('candidate_correct', 'mean'), final_target_acc=('target_correct', 'mean')))
    print('Target checks: OK')
    runtime.atpk_step = MethodType(atpk_step, runtime)
    import marshal
    runtime.marshal = marshal
    runtime.ATPK_CODE_HASH = runtime.hashlib.sha256(b''.join((runtime.marshal.dumps(fn.__code__) for fn in (runtime.atpk_target, runtime.atpk_pair_gate, runtime.atpk_step)))).hexdigest()
    runtime.ATPK_TARGET_HASH = {}
    for runtime.arm in runtime.ATPK_RUN:
        runtime.digest = runtime.hashlib.sha256()
        for runtime.item in runtime._ATPK_ITEMS:
            runtime.digest.update(runtime.ATPK_TARGETS[runtime.arm][runtime.key(runtime.item)].contiguous().numpy().tobytes())
        runtime.ATPK_TARGET_HASH[runtime.arm] = runtime.digest.hexdigest()
    runtime.atpk_student_cfg = MethodType(atpk_student_cfg, runtime)
    runtime.ATPK_ROWS = []
    runtime.REFERENCE_RUNS = ['base', 'seed_base_s0', 'token_pair_s0', 'answer_token_only_s0', 'nx_vpc_keep_repair_s0', 'nx_vpc_keep_repair_adaptive_pos_s0', 'nx_vpc_keep_repair_adaptive_neg_s0', 'nx_vpc_keep_repair_adaptive_both_s0']
    for runtime.tag in runtime.REFERENCE_RUNS:
        if runtime.tag in runtime.MARG and all((k in runtime.MARG[runtime.tag] for k in runtime._eval_keys)):
            runtime.ATPK_ROWS.extend(runtime.result_rows(runtime.tag, runtime.MARG[runtime.tag]))
        else:
            print('Bỏ reference chưa đủ margins:', runtime.tag)
    runtime._previous_cfg = runtime.student_cfg
    runtime._previous_dir = runtime.STUDENT_DIR
    runtime._previous_methods = {name: runtime.METHODS.get(name) for name in runtime.ATPK_RUN}
    try:
        runtime.student_cfg = runtime.atpk_student_cfg
        runtime.STUDENT_DIR = str(runtime.ATPK_ROOT / 'students')
        runtime.Path(runtime.STUDENT_DIR).mkdir(parents=True, exist_ok=True)
        for runtime.arm in runtime.ATPK_RUN:
            runtime.METHODS[runtime.arm] = (lambda arm=runtime.arm: runtime.atpk_step(arm), None, None, True)
            for runtime.seed in runtime.ATPK['seeds']:
                assert not hasattr(runtime.model, 'peft_config'), f'Adapter chưa unload trước {runtime.arm}'
                runtime.atpk_cleanup()
                runtime.started = runtime.time.time()
                runtime.output = runtime.train_arm(runtime.arm, runtime.seed)
                assert all((k in runtime.output['marg'] for k in runtime._eval_keys))
                runtime.tag = f'{runtime.arm}_s{runtime.seed}'
                runtime.MARG[runtime.tag] = runtime.output['marg']
                runtime.rows = runtime.result_rows(runtime.tag, runtime.output['marg'])
                runtime.ATPK_ROWS.extend(runtime.rows)
                runtime.pd.DataFrame(runtime.ATPK_ROWS).to_csv(runtime.ATPK_ROOT / 'results.csv', index=False)
                runtime.pd.DataFrame([{'id': runtime.key(i), 'video': i['video'], 'question': i['question'], 'answer': i['answer'], 'base_margin': runtime.MARG['base'][runtime.key(i)], 'student_margin': runtime.output['marg'][runtime.key(i)]} for i in runtime._ATPK_EVAL]).to_csv(runtime.ATPK_ROOT / f'{runtime.tag}_predictions.csv', index=False)
                print(f'{runtime.tag}: {(runtime.time.time() - runtime.started) / 60:.1f} phút')
                runtime.display(runtime.pd.DataFrame(runtime.rows))
                runtime.atpk_cleanup()
    finally:
        runtime.student_cfg = runtime._previous_cfg
        runtime.STUDENT_DIR = runtime._previous_dir
        for runtime.name, runtime.old in runtime._previous_methods.items():
            if runtime.old is None:
                runtime.METHODS.pop(runtime.name, None)
            else:
                runtime.METHODS[runtime.name] = runtime.old
        runtime.atpk_cleanup()
    runtime.atpk_report = runtime.pd.DataFrame(runtime.ATPK_ROWS)
    runtime.atpk_report.to_csv(runtime.ATPK_ROOT / 'results.csv', index=False)
    runtime.main = runtime.atpk_report[runtime.atpk_report['split'].isin(['VH_VAL', 'VH_TEST', 'EH_ALL'])].copy()
    runtime.base_scores = runtime.main[runtime.main['run'] == 'base'].set_index('split')
    runtime.main['q_delta_base_pp'] = runtime.main.apply(lambda r: 100 * (r['q_acc'] - runtime.base_scores.loc[r['split'], 'q_acc']), axis=1)
    runtime.main['pair_delta_base_pp'] = runtime.main.apply(lambda r: 100 * (r['pair_acc'] - runtime.base_scores.loc[r['split'], 'pair_acc']), axis=1)
    runtime.main['net_corrections'] = runtime.main['wrong_to_correct'] - runtime.main['correct_to_wrong']
    runtime.main.to_csv(runtime.ATPK_ROOT / 'main_comparison.csv', index=False)
    runtime.display(runtime.main[['run', 'split', 'n', 'q_acc', 'pair_acc', 'q_delta_base_pp', 'pair_delta_base_pp', 'wrong_to_correct', 'correct_to_wrong', 'net_corrections', 'yes_rate']])
    runtime.compact = runtime.main.pivot(index='run', columns='split', values=['q_acc', 'pair_acc']) * 100
    print('Accuracy / Pair accuracy (%):')
    runtime.display(runtime.compact.round(2))
    runtime.compact.to_csv(runtime.ATPK_ROOT / 'accuracy_percent.csv')
    print('Hoàn thành:', runtime.ATPK_ROOT)
    print('Giữ toàn bộ checkpoint và kết quả cũ.')
