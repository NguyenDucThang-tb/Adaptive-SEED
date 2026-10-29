"""experiments: factorial stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType

def run(runtime):
    from pathlib import Path
    runtime.Path = Path
    import gc
    runtime.gc = gc
    import numpy as np
    runtime.np = np
    import pandas as pd
    runtime.pd = pd
    import torch
    runtime.torch = torch
    runtime.required = ['NX', 'NX_ROOT', 'NX_SPECS', 'NX_TARGETS', 'NX_KD_WEIGHT', 'NX_CAPTION_LOGITS', 'SHARED', 'BASE_TEACHER', 'nx_spec', 'nx_target', 'nx_step', 'nx_student_cfg', 'METHODS', 'MARG', 'TRAIN', 'VAL', 'TEST', 'EH', 'train_arm', 'result_rows', 'student_cfg', 'STUDENT_DIR', 'seed_beta', 'key', 'margin']
    runtime.missing = [name for name in runtime.required if name not in vars(runtime)]
    assert not runtime.missing, 'Thiếu biến từ cell teacher32: ' + ', '.join(runtime.missing)
    assert not hasattr(runtime.model, 'peft_config'), 'Model vẫn đang gắn adapter.'
    assert runtime.CFG['n_frames'] == 8
    assert runtime.NX['caption_frames'] == 32
    assert runtime.NX['teacher_frames'] == 32
    assert runtime.NX['version'] == 'seed_teacher32_components_v2'
    runtime.AP0 = float(runtime.BASE_TEACHER['a_pos'])
    runtime.AN0 = float(runtime.BASE_TEACHER['negs']['tcd'])
    print('Student:', runtime.CFG['n_frames'], 'frames')
    print('Teacher caption:', runtime.NX['teacher_frames'], 'frames')
    print('a_pos:', runtime.AP0, '| a_neg:', runtime.AN0)
    print('Epochs:', runtime.CFG['epochs'])
    runtime.FACTORIAL_SPECS = {'nx_vpc_adaptive_both': runtime.nx_spec(visual=True, pair=True, caption=True, adaptive_pos=True, adaptive_neg=True, keep_repair=False), 'nx_vpc_keep_repair_adaptive_pos': runtime.nx_spec(visual=True, pair=True, caption=True, adaptive_pos=True, adaptive_neg=False, keep_repair=True), 'nx_vpc_keep_repair_adaptive_neg': runtime.nx_spec(visual=True, pair=True, caption=True, adaptive_pos=False, adaptive_neg=True, keep_repair=True), 'nx_vpc_keep_repair_adaptive_both': runtime.nx_spec(visual=True, pair=True, caption=True, adaptive_pos=True, adaptive_neg=True, keep_repair=True)}
    runtime.FACTORIAL_RUN = list(runtime.FACTORIAL_SPECS)
    runtime.NX_SPECS.update(runtime.FACTORIAL_SPECS)
    print('Arms sẽ chạy:')
    for runtime.name in runtime.FACTORIAL_RUN:
        print(' -', runtime.name, runtime.NX_SPECS[runtime.name])
    runtime.factorial_audit = []
    for runtime.arm in runtime.FACTORIAL_RUN:
        runtime.spec = runtime.NX_SPECS[runtime.arm]
        runtime.NX_TARGETS[runtime.arm] = {}
        for runtime.it in runtime.TRAIN:
            runtime.item_key = runtime.key(runtime.it)
            runtime.record = runtime.SHARED[runtime.item_key]
            runtime.positive = runtime.record['selected']
            runtime.pair_factor = float(runtime.record['pair'][True][0]) if runtime.record.get('pair_eligible', False) else 1.0
            assert runtime.np.isfinite(runtime.pair_factor)
            assert 0.0 <= runtime.pair_factor <= 1.0
            assert runtime.item_key in runtime.NX_CAPTION_LOGITS
            runtime.caption_logits = runtime.NX_CAPTION_LOGITS[runtime.item_key]
            runtime.target, runtime.kd_weight, runtime.route, runtime.ap, runtime.an = runtime.nx_target(orig=runtime.record['orig'], positive=runtime.positive, negative=runtime.record['tcd'], caption_logits=runtime.caption_logits, beta=runtime.seed_beta(runtime.item_key), spec=runtime.spec, pair_factor=runtime.pair_factor, answer=runtime.it['answer'])
            assert runtime.torch.isfinite(runtime.target).all()
            runtime.NX_TARGETS[runtime.arm][runtime.item_key] = runtime.target
            runtime.NX_KD_WEIGHT[runtime.arm, runtime.item_key] = runtime.kd_weight
            runtime.target_margin = runtime.margin(runtime.target)
            runtime.base_margin = runtime.margin(runtime.record['orig'])
            runtime.factorial_audit.append({'arm': runtime.arm, 'id': runtime.item_key, 'route': runtime.route, 'kd_weight': runtime.kd_weight, 'a_pos': runtime.ap, 'a_neg': runtime.an, 'pair_gate': runtime.pair_factor, 'base_margin': runtime.base_margin, 'target_margin': runtime.target_margin, 'base_correct': (runtime.base_margin > 0) == (runtime.it['answer'] == 'yes'), 'target_correct': (runtime.target_margin > 0) == (runtime.it['answer'] == 'yes')})
    runtime.factorial_audit = runtime.pd.DataFrame(runtime.factorial_audit)
    runtime.factorial_audit.to_csv(runtime.NX_ROOT / 'factorial_adaptive_keep_repair_targets.csv', index=False)
    runtime.target_summary = runtime.factorial_audit.groupby(['arm', 'route']).agg(n=('id', 'size'), mean_a_pos=('a_pos', 'mean'), mean_a_neg=('a_neg', 'mean'), mean_pair_gate=('pair_gate', 'mean'), base_acc=('base_correct', 'mean'), target_acc=('target_correct', 'mean')).reset_index()
    runtime.display(runtime.target_summary)
    runtime.full_arm = 'nx_vpc_keep_repair_adaptive_both'
    runtime.full_spec = runtime.NX_SPECS[runtime.full_arm]
    assert runtime.full_spec['visual']
    assert runtime.full_spec['pair']
    assert runtime.full_spec['caption']
    assert runtime.full_spec['adaptive_pos']
    assert runtime.full_spec['adaptive_neg']
    assert runtime.full_spec['keep_repair']
    assert not runtime.full_spec['null']
    assert not runtime.full_spec['reverse']
    for runtime.arm in runtime.FACTORIAL_RUN:
        assert len(runtime.NX_TARGETS[runtime.arm]) == len(runtime.TRAIN)
        for runtime.it in runtime.TRAIN[:3]:
            runtime.target = runtime.NX_TARGETS[runtime.arm][runtime.key(runtime.it)]
            assert runtime.torch.isfinite(runtime.target).all()
    print('Target checks: OK')
    for runtime.arm in runtime.FACTORIAL_RUN:
        runtime.METHODS[runtime.arm] = (lambda arm=runtime.arm: runtime.nx_step(arm), None, None, True)
    runtime.REFERENCE_RUNS = ['base', 'seed_base_s0', 'token_pair_s0', 'answer_token_only_s0', 'nx_visual_pair_caption32_s0', 'nx_vpc_adaptive_pos_s0', 'nx_vpc_adaptive_neg_s0', 'nx_vpc_keep_repair_s0', 'nx_vpc_null_s0']
    runtime.factorial_rows = []
    for runtime.run in runtime.REFERENCE_RUNS:
        if runtime.run in runtime.MARG:
            runtime.factorial_rows.extend(runtime.result_rows(runtime.run, runtime.MARG[runtime.run]))
        else:
            print('Reference không có trong RAM:', runtime.run)
    runtime.previous_student_cfg = runtime.student_cfg
    runtime.previous_student_dir = runtime.STUDENT_DIR
    try:
        runtime.student_cfg = runtime.nx_student_cfg
        runtime.STUDENT_DIR = str(runtime.NX_ROOT / 'students_factorial')
        runtime.Path(runtime.STUDENT_DIR).mkdir(parents=True, exist_ok=True)
        for runtime.arm in runtime.FACTORIAL_RUN:
            for runtime.seed in runtime.NX['seeds']:
                assert not hasattr(runtime.model, 'peft_config')
                runtime.output = runtime.train_arm(runtime.arm, runtime.seed)
                runtime.run = f'{runtime.arm}_s{runtime.seed}'
                runtime.MARG[runtime.run] = runtime.output['marg']
                runtime.rows = runtime.result_rows(runtime.run, runtime.output['marg'])
                runtime.factorial_rows.extend(runtime.rows)
                runtime.display(runtime.pd.DataFrame(runtime.rows))
                runtime.pd.DataFrame(runtime.factorial_rows).to_csv(runtime.NX_ROOT / 'factorial_results.csv', index=False)
                runtime.predictions = runtime.pd.DataFrame([{'id': runtime.key(it), 'video': it['video'], 'question': it['question'], 'answer': it['answer'], 'base_margin': runtime.MARG['base'][runtime.key(it)], 'student_margin': runtime.output['marg'][runtime.key(it)]} for it in runtime.eval_set()])
                runtime.predictions.to_csv(runtime.NX_ROOT / f'{runtime.run}_predictions.csv', index=False)
                runtime.gc.collect()
                runtime.torch.cuda.empty_cache()
    finally:
        runtime.student_cfg = runtime.previous_student_cfg
        runtime.STUDENT_DIR = runtime.previous_student_dir
    runtime.factorial_report = runtime.pd.DataFrame(runtime.factorial_rows)
    runtime.base_reference = runtime.factorial_report[runtime.factorial_report['run'] == 'base'].set_index('split')
    runtime.seed_reference = runtime.factorial_report[runtime.factorial_report['run'] == 'seed_base_s0'].set_index('split')
    for runtime.metric in ['q_acc', 'pair_acc']:
        runtime.factorial_report[f'{runtime.metric}_delta_base_pp'] = 100 * (runtime.factorial_report[runtime.metric] - runtime.factorial_report['split'].map(runtime.base_reference[runtime.metric]))
        runtime.factorial_report[f'{runtime.metric}_delta_seed_pp'] = 100 * (runtime.factorial_report[runtime.metric] - runtime.factorial_report['split'].map(runtime.seed_reference[runtime.metric]))
    runtime.factorial_report['net_corrections'] = runtime.factorial_report['wrong_to_correct'] - runtime.factorial_report['correct_to_wrong']
    runtime.factorial_report.to_csv(runtime.NX_ROOT / 'factorial_results.csv', index=False)
    runtime.main_splits = ['VH_VAL', 'VH_TEST', 'EH_ALL']
    runtime.main_report = runtime.factorial_report[runtime.factorial_report['split'].isin(runtime.main_splits)][['run', 'split', 'n', 'q_acc', 'pair_acc', 'q_acc_delta_base_pp', 'pair_acc_delta_base_pp', 'q_acc_delta_seed_pp', 'pair_acc_delta_seed_pp', 'wrong_to_correct', 'correct_to_wrong', 'net_corrections', 'yes_rate']]
    runtime.display(runtime.main_report)
    runtime.new_arm_names = {f'{arm}_s{seed}' for arm in runtime.FACTORIAL_RUN for seed in runtime.NX['seeds']}
    runtime.new_arm_report = runtime.factorial_report[runtime.factorial_report['run'].isin(runtime.new_arm_names) & runtime.factorial_report['split'].isin(runtime.main_splits)].copy()
    runtime.display(runtime.new_arm_report[['run', 'split', 'q_acc', 'pair_acc', 'q_acc_delta_base_pp', 'pair_acc_delta_base_pp', 'wrong_to_correct', 'correct_to_wrong', 'net_corrections']])
    runtime.new_arm_report.to_csv(runtime.NX_ROOT / 'factorial_new_arms_summary.csv', index=False)
    print('Hoàn thành:', runtime.NX_ROOT / 'factorial_results.csv')
