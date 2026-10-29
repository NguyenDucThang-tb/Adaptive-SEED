"""experiments: seednoise stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType

def run(runtime):
    from pathlib import Path
    runtime.Path = Path
    from tqdm.auto import tqdm
    runtime.tqdm = tqdm
    assert 'SHARED' in vars(runtime), 'Phải chạy xong cell ablation 8-arm trước.'
    assert 'TARGETS_ABL' in vars(runtime)
    assert 'BASE_TEACHER' in vars(runtime)
    assert 'component_step' in vars(runtime)
    assert 'result_rows' in vars(runtime)
    assert not hasattr(runtime.model, 'peft_config'), 'Model vẫn đang gắn adapter.'
    runtime.NAME = 'full_seednoise'
    runtime.NOISE_OUT = runtime.Path(runtime.CACHE) / 'seednoise_negative_compare'
    runtime.NOISE_OUT.mkdir(parents=True, exist_ok=True)
    runtime.AP0 = float(runtime.BASE_TEACHER['a_pos'])
    runtime.AN0 = float(runtime.BASE_TEACHER['negs']['tcd'])
    print('Positive coefficient:', runtime.AP0)
    print('Matched negative coefficient:', runtime.AN0)
    print('SeedNoise alpha:', runtime.CFG['seed_noise_alpha'])
    print('Epochs:', runtime.CFG['epochs'])
    runtime.SEEDNOISE_LOGITS = {}
    for runtime.it in runtime.tqdm(runtime.TRAIN, desc='SeedNoise teacher TRAIN'):
        runtime.k = runtime.key(runtime.it)
        runtime.meta = {'source': runtime.SOURCE_HASH, 'item': runtime.item_sig(runtime.it), 'branch': runtime.branch_cfg('seednoise')}
        runtime.path = runtime.NOISE_OUT / ('seednoise_' + runtime.cfg_hash(runtime.meta) + '.pt')
        if runtime.path.exists():
            runtime.obj = runtime.torch.load(runtime.path, map_location='cpu', weights_only=False)
            assert runtime.obj['meta'] == runtime.meta
            runtime.z_noise = runtime.obj['logits']
        else:
            runtime.model.eval()
            with runtime.adapter_off(), runtime.torch.no_grad():
                runtime.z_noise = runtime.fwd_view(runtime.it, 'seednoise')[0].logits[0, -1].detach().float().cpu()
            runtime.tmp = str(runtime.path) + '.tmp'
            runtime.torch.save({'meta': runtime.meta, 'logits': runtime.z_noise}, runtime.tmp)
            runtime.os.replace(runtime.tmp, runtime.path)
        assert runtime.torch.isfinite(runtime.z_noise).all()
        runtime.SEEDNOISE_LOGITS[runtime.k] = runtime.z_noise
    print('SeedNoise cached:', len(runtime.SEEDNOISE_LOGITS))
    runtime.TARGETS_ABL[runtime.NAME] = {}
    runtime.noise_audit = []
    for runtime.it in runtime.TRAIN:
        runtime.k = runtime.key(runtime.it)
        runtime.r = runtime.SHARED[runtime.k]
        runtime.o = runtime.r['orig']
        runtime.p = runtime.r['selected']
        runtime.n = runtime.SEEDNOISE_LOGITS[runtime.k]
        runtime.gt = runtime.gt_tok(runtime.it)
        runtime.gain_pos = runtime.correct_margin(runtime.p, runtime.gt) - runtime.correct_margin(runtime.o, runtime.gt)
        runtime.ap = runtime.AP0
        runtime.ap *= min(1.0, max(0.0, runtime.gain_pos / runtime.ABL['gain_tau']))
        runtime.ref = runtime.o + runtime.ap * (runtime.p - runtime.o)
        runtime.gain_neg = runtime.correct_margin(runtime.ref + runtime.AN0 * (runtime.o - runtime.n), runtime.gt) - runtime.correct_margin(runtime.ref, runtime.gt)
        runtime.an = runtime.AN0
        runtime.an *= min(1.0, max(0.0, runtime.gain_neg / runtime.ABL['gain_tau']))
        runtime.gate, runtime.sensitivity, runtime.equivariance_error = runtime.r['pair'][True]
        runtime.an *= runtime.gate
        runtime.z = runtime.o + runtime.seed_beta(runtime.k) * (runtime.ap * (runtime.p - runtime.o) + runtime.an * (runtime.o - runtime.n))
        assert runtime.torch.isfinite(runtime.z).all()
        runtime.TARGETS_ABL[runtime.NAME][runtime.k] = runtime.z
        runtime.noise_audit.append({'id': runtime.k, 'arm': runtime.NAME, 'negative': 'seednoise', 'a_pos': runtime.ap, 'a_neg': runtime.an, 'gain_pos': runtime.gain_pos, 'gain_neg': runtime.gain_neg, 'pair_gate': runtime.gate, 'sensitivity': runtime.sensitivity, 'equivariance_error': runtime.equivariance_error, 'pair_eligible': runtime.r['pair_eligible'], 'base_margin': runtime.margin(runtime.o), 'target_margin': runtime.margin(runtime.z), 'answer': runtime.it['answer']})
    runtime.noise_audit = runtime.pd.DataFrame(runtime.noise_audit)
    runtime.display(runtime.noise_audit.describe(include='all'))
    runtime.noise_audit.to_csv(runtime.NOISE_OUT / 'target_diagnostics_train.csv', index=False)
    runtime.ARMS[runtime.NAME] = {'token_selection': True, 'pair_gate': True, 'adaptive_positive': True, 'adaptive_negative': True, 'negative': 'seednoise', 'a_pos_base': runtime.AP0, 'a_neg_base': runtime.AN0, 'noise_alpha': runtime.CFG['seed_noise_alpha']}
    runtime.METHODS[runtime.NAME] = (lambda: runtime.component_step(runtime.NAME), None, None, True)
    runtime.mean_target_difference = float(runtime.np.mean([(runtime.TARGETS_ABL[runtime.NAME][runtime.key(it)] - runtime.TARGETS_ABL['seed_base'][runtime.key(it)]).abs().mean().item() for it in runtime.TRAIN]))
    print('Mean |full_seednoise target - seed_base target|:', runtime.mean_target_difference)
    assert runtime.mean_target_difference > 0
    runtime.seed = 0
    if 'seed_base_s0' not in runtime.MARG:
        runtime.seed_out = runtime.train_arm('seed_base', runtime.seed)
        runtime.MARG['seed_base_s0'] = runtime.seed_out['marg']
    runtime.noise_out = runtime.train_arm(runtime.NAME, runtime.seed)
    runtime.MARG[f'{runtime.NAME}_s{runtime.seed}'] = runtime.noise_out['marg']
    runtime.compare_rows = []
    runtime.compare_rows += runtime.result_rows('base', runtime.MARG['base'])
    runtime.compare_rows += runtime.result_rows('seed_base_s0', runtime.MARG['seed_base_s0'])
    runtime.compare_rows += runtime.result_rows(f'{runtime.NAME}_s{runtime.seed}', runtime.MARG[f'{runtime.NAME}_s{runtime.seed}'])
    runtime.noise_report = runtime.pd.DataFrame(runtime.compare_rows)
    runtime.keep_splits = ['VH_VAL', 'VH_TEST', 'EH_ALL', 'eh_entire', 'eh_misleading', 'eh_mix']
    runtime.noise_report = runtime.noise_report[runtime.noise_report['split'].isin(runtime.keep_splits)].reset_index(drop=True)
    runtime.display(runtime.noise_report)
    runtime.noise_report.to_csv(runtime.NOISE_OUT / 'results.csv', index=False)
    runtime.summary = runtime.noise_report.pivot_table(index='run', columns='split', values=['q_acc', 'pair_acc'], aggfunc='first')
    runtime.display(runtime.summary)
    runtime.summary.to_csv(runtime.NOISE_OUT / 'summary.csv')
    print('Saved:', runtime.NOISE_OUT)
