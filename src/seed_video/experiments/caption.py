"""experiments: caption stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType

def nx_spec(runtime, **kwargs):
    spec = dict(visual=False, pair=False, caption=False, adaptive_pos=False, adaptive_neg=False, agreement=False, keep_repair=False, null=False, reverse=False)
    spec.update(kwargs)
    return spec

def nx_digest(runtime, obj):
    return runtime.hashlib.sha256(runtime.json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()

def nx_cache(runtime, kind, metadata, compute):
    path = runtime.NX_ROOT / f'{kind}_{runtime.nx_digest(metadata)}.pt'
    if path.exists():
        obj = runtime.torch.load(path, map_location='cpu', weights_only=False)
        assert obj['metadata'] == metadata
        return obj['value']
    value = compute()
    temp = str(path) + '.tmp'
    runtime.torch.save({'metadata': metadata, 'value': value}, temp)
    runtime.os.replace(temp, path)
    return value

def nx_raw_forward(runtime, it, caption=None, n_frames=None, special=None, grad=False):
    if n_frames is None or n_frames == runtime.CFG['n_frames']:
        frames, fps, _ = runtime.view_frames(it, 'orig')
    else:
        frames, duration = runtime._decode(it['video'], n_frames, runtime.CFG['frame_size'])
        frames = list(frames)
        fps = len(frames) / duration
    if special == 'null':
        frames = [runtime.Image.new('RGB', frame.size, (0, 0, 0)) for frame in frames]
    elif special == 'reverse':
        frames = list(reversed(frames))
    question = it['question']
    if caption is not None:
        question = runtime.CAPTION_PREFIX + caption + '\n</description>\nQuestion: ' + question
    inputs = runtime.build_inputs(frames, question, fps, yn=True).to(runtime.DEV)
    previous_hook = runtime.hook.active
    runtime.hook.active = False
    runtime._reset_rope(runtime.model)
    try:
        context = runtime.torch.enable_grad() if grad else runtime.torch.no_grad()
        with context, runtime.keep_last(1):
            z = runtime.model(**inputs).logits[0, -1].float()
        return z if grad else z.detach().cpu()
    finally:
        runtime.hook.active = previous_hook
        runtime._reset_rope(runtime.model)

def nx_caption(runtime, it):
    metadata = dict(version=runtime.NX['version'], source=runtime.SOURCE_HASH, video_hash=runtime.item_sig(it)[0], model=runtime.CFG['model_id'], frames=runtime.NX['caption_frames'], size=runtime.CFG['frame_size'], prompt=runtime.CAPTION_PROMPT, max_tokens=runtime.NX['caption_max_tokens'], decoding='greedy')

    def compute():
        frames, duration = runtime._decode(it['video'], runtime.NX['caption_frames'], runtime.CFG['frame_size'])
        inputs = runtime.build_inputs(list(frames), runtime.CAPTION_PROMPT, len(frames) / duration, yn=False).to(runtime.DEV)
        old_training = runtime.model.training
        old_hook = runtime.hook.active
        runtime.model.eval()
        runtime.hook.active = False
        runtime._reset_rope(runtime.model)
        try:
            with runtime.adapter_off(), runtime.torch.no_grad(), runtime.keep_last(1):
                generated = runtime.model.generate(**inputs, do_sample=False, max_new_tokens=runtime.NX['caption_max_tokens'])
            caption = runtime.processor.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            if not caption:
                raise RuntimeError(f"Caption rỗng: {it['video']}")
            return caption
        finally:
            runtime._reset_rope(runtime.model)
            runtime.hook.active = old_hook
            runtime.model.train(old_training)
    return runtime.nx_cache('caption32', metadata, compute)

def nx_teacher_logits(runtime, it, caption=None, n_frames=32):
    metadata = dict(version=runtime.NX['version'], source=runtime.SOURCE_HASH, item=runtime.item_sig(it), model=runtime.CFG['model_id'], frames=n_frames, size=runtime.CFG['frame_size'], caption=caption, caption_prefix=runtime.CAPTION_PREFIX, prompt_tag=runtime.PROMPT_TAG, enhancement=False)

    def compute():
        old_training = runtime.model.training
        runtime.model.eval()
        try:
            with runtime.adapter_off():
                z = runtime.nx_raw_forward(it, caption=caption, n_frames=n_frames, grad=False)
            assert runtime.torch.isfinite(z).all()
            return z
        finally:
            runtime.model.train(old_training)
    return runtime.nx_cache('teacher_logits', metadata, compute)

def nx_target(runtime, orig, positive, negative, caption_logits, beta, spec, pair_factor, answer):
    orig = orig.float()
    positive = positive.float()
    negative = negative.float()
    gt = runtime.YES if answer == 'yes' else runtime.NO
    other = runtime.NO if gt == runtime.YES else runtime.YES
    dp = positive - orig
    dn = orig - negative
    ap, an = (runtime.AP0, runtime.AN0)
    if spec['adaptive_pos']:
        gain = float(dp[gt] - dp[other])
        ap *= min(1.0, max(0.0, gain / runtime.ABL['gain_tau']))
    if spec['adaptive_neg']:
        gain = runtime.AN0 * float(dn[gt] - dn[other])
        an *= min(1.0, max(0.0, gain / runtime.ABL['gain_tau']))
    if spec['agreement']:
        product = float(dp[runtime.YES] - dp[runtime.NO]) * float(dn[runtime.YES] - dn[runtime.NO])
        an *= max(product, 0.0) / (abs(product) + 1e-08)
    if spec['pair']:
        an *= pair_factor
    residual = ap * dp + an * dn
    if spec['caption']:
        residual = residual + runtime.NX['caption_weight'] * (caption_logits.float() - positive)
    dm = beta * (residual[runtime.YES] - residual[runtime.NO])
    target = orig.clone()
    target[runtime.YES] += dm / 2
    target[runtime.NO] -= dm / 2
    weight, route = (1.0, 'normal')
    if spec['keep_repair']:
        base_ok = (float(orig[runtime.YES] - orig[runtime.NO]) > 0) == (answer == 'yes')
        target_ok = (float(target[runtime.YES] - target[runtime.NO]) > 0) == (answer == 'yes')
        if base_ok:
            target = orig.clone()
            route = 'keep'
        elif target_ok:
            route = 'repair'
        else:
            weight = 0.0
            route = 'ce_only'
    return (target, weight, route, ap, an)

def nx_step(runtime, arm):
    spec = runtime.NX_SPECS[arm]

    def step(it, context):
        k = runtime.key(it)
        student = runtime.student_last(it)
        teacher = runtime.NX_TARGETS[arm][k].to(runtime.DEV)
        kd = runtime.rkl(teacher[None], student[None]) * runtime.NX_KD_WEIGHT[arm, k]
        sample_seed = runtime.seed_of(k) + context['g']
        draw = runtime.random.Random(sample_seed).random()
        if runtime.NX_KD_WEIGHT[arm, k] > 0 and draw < runtime.CFG['seed_gamma']:
            label = runtime.torch.multinomial(teacher.softmax(0), 1, generator=runtime.torch.Generator(device=runtime.DEV).manual_seed(sample_seed))
        else:
            label = runtime.torch.tensor([runtime.gt_tok(it)], device=runtime.DEV)
        ce = runtime.F.cross_entropy(student[None].float(), label.view(1))
        loss = kd + ce
        parts = dict(rkl=float(kd.detach()), ce=float(ce.detach()), null=0.0, reverse=0.0, equiv=0.0)
        if spec['null'] and runtime.random.Random(sample_seed + 7919).random() < runtime.NX['null_probability']:
            null_z = runtime.nx_raw_forward(it, special='null', grad=True)
            lp = null_z[[runtime.YES, runtime.NO]].log_softmax(0)
            null_loss = (lp.exp() * (lp + runtime.math.log(2))).sum()
            loss = loss + runtime.NX['null_weight'] * null_loss
            parts['null'] = float(null_loss.detach())
        if spec['reverse'] and k in runtime.NX_REVERSE:
            reverse_z = runtime.nx_raw_forward(it, special='reverse', grad=True)
            reverse_answer = runtime.NX_REVERSE[k]
            reverse_label = runtime.torch.tensor([runtime.YES if reverse_answer == 'yes' else runtime.NO], device=runtime.DEV)
            reverse_ce = runtime.F.cross_entropy(reverse_z[None], reverse_label)
            pf = student[[runtime.YES, runtime.NO]].softmax(0)
            pr = reverse_z[[runtime.YES, runtime.NO]].softmax(0)
            if reverse_answer != it['answer']:
                pr = pr.flip(0)
            midpoint = (pf + pr) / 2
            lm = midpoint.clamp_min(1e-08).log()
            eq_loss = 0.5 * ((pf * (pf.clamp_min(1e-08).log() - lm)).sum() + (pr * (pr.clamp_min(1e-08).log() - lm)).sum())
            loss = loss + runtime.NX['reverse_weight'] * reverse_ce + runtime.NX['equivariance_weight'] * eq_loss
            parts['reverse'] = float(reverse_ce.detach())
            parts['equiv'] = float(eq_loss.detach())
        return (loss, parts)
    return step

def nx_student_cfg(runtime, arm, seed):
    target_hash = runtime.hashlib.sha256(b''.join((runtime.NX_TARGETS[arm][runtime.key(it)].numpy().tobytes() for it in runtime.TRAIN))).hexdigest()
    return dict(version=runtime.NX['version'], source=runtime.SOURCE_HASH, config=runtime.NX, arm=arm, spec=runtime.NX_SPECS[arm], seed=seed, teacher=runtime.BASE_TEACHER, abl=runtime.ABL, lora_targets=runtime.LORA_TARGETS, hp={name: runtime.CFG[name] for name in runtime.STUDENT_HP}, train=[runtime.item_sig(it) for it in runtime.TRAIN], evaluation=[runtime.item_sig(it) for it in runtime.eval_set()], reverse_labels=runtime.NX_REVERSE, target_hash=target_hash, kd_weights=[runtime.NX_KD_WEIGHT[arm, runtime.key(it)] for it in runtime.TRAIN])

def run(runtime):
    from pathlib import Path
    runtime.Path = Path
    from PIL import Image
    runtime.Image = Image
    from tqdm.auto import tqdm
    runtime.tqdm = tqdm
    import hashlib
    runtime.hashlib = hashlib
    import json
    runtime.json = json
    import math
    runtime.math = math
    import os
    runtime.os = os
    import random
    runtime.random = random
    import gc
    runtime.gc = gc
    import numpy as np
    runtime.np = np
    import pandas as pd
    runtime.pd = pd
    import torch
    runtime.torch = torch
    import torch.nn.functional as F
    runtime.F = F
    runtime.required = ['SHARED', 'BASE_TEACHER', 'METHODS', 'train_arm', 'student_cfg', 'STUDENT_DIR', 'STUDENT_HP', 'result_rows', 'MARG', 'ABL', 'item_sig', 'seed_beta', 'build_inputs', 'view_frames', 'hook', '_decode', '_reset_rope', 'adapter_off', 'keep_last']
    runtime.missing = [name for name in runtime.required if name not in vars(runtime)]
    assert not runtime.missing, f'Thiếu runtime notebook cũ: {runtime.missing}'
    assert not hasattr(runtime.model, 'peft_config'), 'Unload adapter trước.'
    assert runtime.CFG['n_frames'] == 8, 'Giữ student ở 8 frame.'
    assert 'base' in runtime.MARG and runtime.TRAIN and runtime.VAL
    runtime.NX = dict(version='seed_teacher32_components_v2', seeds=[0], caption_frames=32, teacher_frames=32, caption_max_tokens=160, caption_weight=0.5, null_probability=0.3, null_weight=0.1, reverse_weight=0.2, equivariance_weight=0.05, teacher_val=True, mismatch_control=True)
    runtime.NX_ROOT = runtime.Path(runtime.CACHE) / runtime.NX['version']
    runtime.NX_ROOT.mkdir(parents=True, exist_ok=True)
    runtime.CAPTION_PROMPT = 'Describe only clearly visible events in chronological order. Use Beginning, Middle, End, Observed changes, and Uncertain or not visible. Mention objects and actions. Do not infer intentions or unseen events. Keep the description short. Do not answer any question.'
    runtime.CAPTION_PREFIX = 'The following automatic description may omit events or contain errors. Check it against the video. An omitted event is not evidence of absence.\n<description>\n'
    runtime.nx_spec = MethodType(nx_spec, runtime)
    runtime.NX_SPECS = {'nx_answer_control': runtime.nx_spec(), 'nx_answer_visual_pair': runtime.nx_spec(visual=True, pair=True), 'nx_answer_caption32': runtime.nx_spec(caption=True), 'nx_answer_null': runtime.nx_spec(null=True), 'nx_answer_reverse': runtime.nx_spec(reverse=True), 'nx_visual_pair_caption32': runtime.nx_spec(visual=True, pair=True, caption=True), 'nx_vpc_adaptive_pos': runtime.nx_spec(visual=True, pair=True, caption=True, adaptive_pos=True), 'nx_vpc_adaptive_neg': runtime.nx_spec(visual=True, pair=True, caption=True, adaptive_neg=True), 'nx_vpc_agreement': runtime.nx_spec(visual=True, pair=True, caption=True, agreement=True), 'nx_vpc_keep_repair': runtime.nx_spec(visual=True, pair=True, caption=True, keep_repair=True), 'nx_vpc_null': runtime.nx_spec(visual=True, pair=True, caption=True, null=True), 'nx_vpc_reverse': runtime.nx_spec(visual=True, pair=True, caption=True, reverse=True), 'nx_all': runtime.nx_spec(visual=True, pair=True, caption=True, adaptive_pos=True, adaptive_neg=True, keep_repair=True, null=True, reverse=True)}
    runtime.NX_RUN = list(runtime.NX_SPECS)
    runtime.NX_REVERSE = {runtime.key(it): it['reverse_answer'] for it in runtime.TRAIN if it.get('reverse_verified') is True and it.get('reverse_answer') in ('yes', 'no')}
    if not runtime.NX_REVERSE:
        runtime.skipped = [name for name in runtime.NX_RUN if runtime.NX_SPECS[name]['reverse']]
        runtime.NX_RUN = [name for name in runtime.NX_RUN if not runtime.NX_SPECS[name]['reverse']]
        print('SKIP — chưa có nhãn reverse xác minh:', runtime.skipped)
    runtime.AP0 = float(runtime.BASE_TEACHER['a_pos'])
    runtime.AN0 = float(runtime.BASE_TEACHER['negs']['tcd'])
    print('Student:', runtime.CFG['n_frames'], 'frames')
    print('Caption generation:', runtime.NX['caption_frames'], 'frames')
    print('Caption teacher:', runtime.NX['teacher_frames'], 'frames')
    print('Positive cũ:', runtime.CFG['n_frames'], '| TCD:', runtime.CFG['tcd_frames'])
    print('a_pos:', runtime.AP0, '| a_neg:', runtime.AN0, '| epochs:', runtime.CFG['epochs'])
    print('New arms:', runtime.NX_RUN)
    print('Output:', runtime.NX_ROOT)
    runtime.nx_digest = MethodType(nx_digest, runtime)
    runtime.nx_cache = MethodType(nx_cache, runtime)
    runtime.nx_raw_forward = MethodType(nx_raw_forward, runtime)
    runtime.nx_caption = MethodType(nx_caption, runtime)
    runtime.nx_teacher_logits = MethodType(nx_teacher_logits, runtime)
    runtime.NX_CAPTIONS = {}
    runtime.NX_CAPTION_LOGITS = {}
    runtime.use_caption = any((runtime.NX_SPECS[name]['caption'] for name in runtime.NX_RUN))
    if runtime.use_caption:
        for runtime.it in runtime.tqdm(runtime.TRAIN, desc='Caption32 + teacher32 TRAIN'):
            runtime.k = runtime.key(runtime.it)
            runtime.caption = runtime.nx_caption(runtime.it)
            runtime.NX_CAPTIONS[runtime.k] = runtime.caption
            runtime.NX_CAPTION_LOGITS[runtime.k] = runtime.nx_teacher_logits(runtime.it, caption=runtime.caption, n_frames=runtime.NX['teacher_frames'])
        runtime.pd.DataFrame([dict(id=runtime.key(it), video=it['video'], caption=runtime.NX_CAPTIONS[runtime.key(it)]) for it in runtime.TRAIN]).to_csv(runtime.NX_ROOT / 'captions_train.csv', index=False)
    if runtime.NX['teacher_val'] and runtime.use_caption:
        runtime.val_captions = {runtime.key(it): runtime.nx_caption(it) for it in runtime.tqdm(runtime.VAL, desc='Caption32 VAL')}
        runtime.unique_videos = {}
        for runtime.it in runtime.VAL:
            runtime.unique_videos.setdefault(runtime.item_sig(runtime.it)[0], runtime.it)
        runtime.unique_items = list(runtime.unique_videos.values())
        runtime.teacher_rows = []
        runtime.teacher_margins = {}
        for runtime.it in runtime.tqdm(runtime.VAL, desc='Teacher comparisons VAL'):
            runtime.k = runtime.key(runtime.it)
            runtime.caption = runtime.val_captions[runtime.k]
            runtime.views = {'base8': runtime.nx_teacher_logits(runtime.it, n_frames=runtime.CFG['n_frames']), 'dense32': runtime.nx_teacher_logits(runtime.it, n_frames=runtime.NX['teacher_frames']), 'base8_caption32': runtime.nx_teacher_logits(runtime.it, caption=runtime.caption, n_frames=runtime.CFG['n_frames']), 'dense32_caption32': runtime.nx_teacher_logits(runtime.it, caption=runtime.caption, n_frames=runtime.NX['teacher_frames'])}
            if runtime.NX['mismatch_control'] and len(runtime.unique_items) > 1:
                runtime.other = next((candidate for candidate in runtime.unique_items if runtime.item_sig(candidate)[0] != runtime.item_sig(runtime.it)[0]))
                runtime.views['dense32_wrong_caption'] = runtime.nx_teacher_logits(runtime.it, caption=runtime.val_captions[runtime.key(runtime.other)], n_frames=runtime.NX['teacher_frames'])
            runtime.base_ok = (runtime.margin(runtime.views['base8']) > 0) == (runtime.it['answer'] == 'yes')
            for runtime.name, runtime.z in runtime.views.items():
                runtime.m = runtime.margin(runtime.z)
                runtime.teacher_margins.setdefault(runtime.name, {})[runtime.k] = runtime.m
                runtime.teacher_rows.append(dict(id=runtime.k, teacher=runtime.name, margin=runtime.m, correct=(runtime.m > 0) == (runtime.it['answer'] == 'yes'), base_correct=runtime.base_ok))
        runtime.td = runtime.pd.DataFrame(runtime.teacher_rows)
        runtime.td.to_csv(runtime.NX_ROOT / 'teacher_val_predictions.csv', index=False)
        runtime.teacher_summary = []
        for runtime.name, runtime.group in runtime.td.groupby('teacher'):
            runtime.teacher_summary.append(dict(teacher=runtime.name, n=len(runtime.group), q_acc=float(runtime.group['correct'].mean()), pair_acc=float(runtime.pair_acc(runtime.teacher_margins[runtime.name], runtime.VAL)), wrong_to_correct=int((~runtime.group['base_correct'] & runtime.group['correct']).sum()), correct_to_wrong=int((runtime.group['base_correct'] & ~runtime.group['correct']).sum())))
        runtime.teacher_summary = runtime.pd.DataFrame(runtime.teacher_summary)
        runtime.display(runtime.teacher_summary)
        runtime.teacher_summary.to_csv(runtime.NX_ROOT / 'teacher_val_summary.csv', index=False)
    runtime.nx_target = MethodType(nx_target, runtime)
    runtime.NX_TARGETS = {}
    runtime.NX_KD_WEIGHT = {}
    runtime.NX_AUDIT = []
    for runtime.arm in runtime.NX_RUN:
        runtime.spec = runtime.NX_SPECS[runtime.arm]
        runtime.NX_TARGETS[runtime.arm] = {}
        for runtime.it in runtime.TRAIN:
            runtime.k = runtime.key(runtime.it)
            runtime.record = runtime.SHARED[runtime.k]
            runtime.positive = runtime.record['selected'] if runtime.spec['visual'] else runtime.record['orig+fg']
            runtime.pair_factor = float(runtime.record['pair'][bool(runtime.spec['visual'])][0]) if runtime.record.get('pair_eligible', False) else 1.0
            assert runtime.np.isfinite(runtime.pair_factor) and 0 <= runtime.pair_factor <= 1
            runtime.caption_logits = runtime.NX_CAPTION_LOGITS[runtime.k] if runtime.spec['caption'] else None
            runtime.target, runtime.weight, runtime.route, runtime.ap, runtime.an = runtime.nx_target(runtime.record['orig'], runtime.positive, runtime.record['tcd'], runtime.caption_logits, runtime.seed_beta(runtime.k), runtime.spec, runtime.pair_factor, runtime.it['answer'])
            assert runtime.torch.isfinite(runtime.target).all()
            runtime.NX_TARGETS[runtime.arm][runtime.k] = runtime.target
            runtime.NX_KD_WEIGHT[runtime.arm, runtime.k] = runtime.weight
            runtime.NX_AUDIT.append(dict(arm=runtime.arm, id=runtime.k, route=runtime.route, kd_weight=runtime.weight, a_pos=runtime.ap, a_neg=runtime.an, pair_gate=runtime.pair_factor, base_margin=runtime.margin(runtime.record['orig']), target_margin=runtime.margin(runtime.target)))
    runtime.audit = runtime.pd.DataFrame(runtime.NX_AUDIT)
    runtime.audit.to_csv(runtime.NX_ROOT / 'target_diagnostics_train.csv', index=False)
    runtime.display(runtime.audit.groupby(['arm', 'route']).size().rename('n'))
    for runtime.it in runtime.TRAIN[:5]:
        runtime.k = runtime.key(runtime.it)
        runtime.r = runtime.SHARED[runtime.k]
        runtime.projected, *runtime._ = runtime.nx_target(runtime.r['orig'], runtime.r['orig+fg'], runtime.r['tcd'], None, runtime.seed_beta(runtime.k), runtime.nx_spec(), 1.0, runtime.it['answer'])
        runtime.expected = runtime.r['orig'] + runtime.seed_beta(runtime.k) * (runtime.AP0 * (runtime.r['orig+fg'] - runtime.r['orig']) + runtime.AN0 * (runtime.r['orig'] - runtime.r['tcd']))
        assert runtime.torch.allclose(runtime.projected[runtime.YES] - runtime.projected[runtime.NO], runtime.expected[runtime.YES] - runtime.expected[runtime.NO], atol=0.0001, rtol=1e-05)
        runtime.outside = runtime.torch.ones_like(runtime.projected, dtype=runtime.torch.bool)
        runtime.outside[runtime.YES] = False
        runtime.outside[runtime.NO] = False
        assert runtime.torch.equal(runtime.projected[runtime.outside], runtime.r['orig'].float()[runtime.outside])
    print('Target algebra checks: OK')
    runtime.nx_step = MethodType(nx_step, runtime)
    runtime.nx_student_cfg = MethodType(nx_student_cfg, runtime)
    runtime.nx_rows = []
    for runtime.tag in ['base', 'seed_base_s0', 'token_pair_s0', 'answer_token_only_s0']:
        if runtime.tag in runtime.MARG:
            runtime.nx_rows.extend(runtime.result_rows(runtime.tag, runtime.MARG[runtime.tag]))
        else:
            print('Reference không có trong RAM, bỏ qua:', runtime.tag)
    runtime.previous_student_cfg = runtime.student_cfg
    runtime.previous_student_dir = runtime.STUDENT_DIR
    try:
        runtime.student_cfg = runtime.nx_student_cfg
        runtime.STUDENT_DIR = str(runtime.NX_ROOT / 'students')
        runtime.Path(runtime.STUDENT_DIR).mkdir(parents=True, exist_ok=True)
        for runtime.arm in runtime.NX_RUN:
            runtime.METHODS[runtime.arm] = (lambda arm=runtime.arm: runtime.nx_step(arm), None, None, True)
            for runtime.seed in runtime.NX['seeds']:
                assert not hasattr(runtime.model, 'peft_config')
                runtime.output = runtime.train_arm(runtime.arm, runtime.seed)
                runtime.tag = f'{runtime.arm}_s{runtime.seed}'
                runtime.MARG[runtime.tag] = runtime.output['marg']
                runtime.rows = runtime.result_rows(runtime.tag, runtime.output['marg'])
                runtime.nx_rows.extend(runtime.rows)
                runtime.display(runtime.pd.DataFrame(runtime.rows))
                runtime.pd.DataFrame(runtime.nx_rows).to_csv(runtime.NX_ROOT / 'results.csv', index=False)
                runtime.pd.DataFrame([dict(id=runtime.key(it), video=it['video'], question=it['question'], answer=it['answer'], base_margin=runtime.MARG['base'][runtime.key(it)], student_margin=runtime.output['marg'][runtime.key(it)]) for it in runtime.eval_set()]).to_csv(runtime.NX_ROOT / f'{runtime.tag}_predictions.csv', index=False)
                runtime.gc.collect()
                runtime.torch.cuda.empty_cache()
    finally:
        runtime.student_cfg = runtime.previous_student_cfg
        runtime.STUDENT_DIR = runtime.previous_student_dir
    runtime.nx_report = runtime.pd.DataFrame(runtime.nx_rows)
    runtime.seed_reference = runtime.nx_report[runtime.nx_report['run'] == 'seed_base_s0']
    if len(runtime.seed_reference):
        runtime.reference = runtime.seed_reference.set_index('split')
        for runtime.metric in ['q_acc', 'pair_acc']:
            runtime.nx_report[f'{runtime.metric}_delta_vs_seed_pp'] = 100 * (runtime.nx_report[runtime.metric] - runtime.nx_report['split'].map(runtime.reference[runtime.metric]))
    runtime.nx_report.to_csv(runtime.NX_ROOT / 'results.csv', index=False)
    runtime.display(runtime.nx_report[runtime.nx_report['split'].isin(['VH_VAL', 'VH_TEST', 'EH_ALL'])])
    print('Hoàn thành. Kết quả:', runtime.NX_ROOT)
