"""experiments: components stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType
from seed_video.evaluation.reporting import result_rows
from seed_video.components.adaptive import adaptive_weight
from seed_video.components.gates import norm_jsd, pair_gate, correct_margin

def file_sha(runtime, path):
    h = runtime.hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def item_sig(runtime, it):
    return [runtime.VIDEO_SHA[it['video']], it['question'], it.get('answer'), it.get('pi'), it.get('pi_valid')]

def feature_enhance(runtime, V, fg, alpha, alpha_s, beta):
    enhanced = runtime._original_feature(V, fg, alpha, alpha_s, beta)
    if not runtime.TOKEN_SELECTED:
        return enhanced
    diff = runtime.torch.zeros_like(V)
    diff[1:] = V[1:] - V[:-1]
    mag = diff.norm(dim=-1)
    eligible = fg > 0
    mask = runtime.torch.zeros_like(fg, dtype=runtime.torch.bool)
    for t in range(V.shape[0]):
        ids = runtime.torch.where(eligible[t].flatten())[0]
        if not len(ids):
            continue
        count = max(1, runtime.math.ceil(len(ids) * runtime.ABL['top_fraction']))
        scores = (mag[t] * fg[t]).flatten()[ids]
        chosen = ids[runtime.torch.argsort(scores, descending=True, stable=True)[:count]]
        mask[t].view(-1)[chosen] = True
    runtime.TOKEN_STATS.append(float(mask.float().mean()))
    return runtime.torch.where(mask[..., None], enhanced, V)

def extra_logits(runtime, it, selected=False, reverse=False, probe=None):
    fr, fps, _ = runtime.view_frames(it, 'orig', runtime.CFG['teacher_frames'])
    if reverse:
        fr = list(reversed(fr))
    inp = runtime.build_inputs(fr, probe or it['question'], fps, yn=True).to(runtime.DEV)
    T, gh, gw = inp['video_grid_thw'][0].tolist()
    fg = runtime.fg_tokens(it['video'], T, gh // 2, gw // 2)
    if reverse:
        fg = fg.flip(0)
    old = runtime.TOKEN_SELECTED
    runtime.TOKEN_SELECTED = selected
    runtime.hook.grid, runtime.hook.fg, runtime.hook.active = ((T, gh, gw), fg, True)
    try:
        runtime.model.eval()
        with runtime.adapter_off(), runtime.torch.no_grad(), runtime.keep_last(1):
            z = runtime.model(**inp).logits[0, -1].detach().float().cpu()
        return z
    finally:
        runtime.hook.active = False
        runtime.TOKEN_SELECTED = old

def prepare_shared(runtime):
    shared = {}
    for it in runtime.tqdm(runtime.TRAIN, desc='Shared teacher views TRAIN'):
        k = runtime.key(it)
        meta = dict(abl=runtime.ABL, source=runtime.SOURCE_HASH, item=runtime.item_sig(it), branches={b: runtime.branch_cfg(b) for b in ['orig', 'orig+fg', 'tcd']})
        path = runtime.OUT / ('views_' + runtime.cfg_hash(meta) + '.pt')
        if path.exists():
            obj = runtime.torch.load(path, map_location='cpu', weights_only=False)
            assert obj['meta'] == meta
            shared[k] = obj['data']
            continue
        rec = {}
        for b in ['orig', 'orig+fg', 'tcd']:
            runtime.model.eval()
            with runtime.adapter_off(), runtime.torch.no_grad():
                rec[b] = runtime.fwd_view(it, b, nf=runtime.CFG['teacher_frames'])[0].logits[0, -1].detach().float().cpu()
        rec['selected'] = runtime.extra_logits(it, selected=True)
        rec['pair'] = {}
        eligible = it.get('pi_valid', False)
        if runtime.ABL['pair_mode'] == 'annotated':
            probe = None
            pi = it.get('pi', [1, 0])
        else:
            eligible = it.get('subset') == 'temporal'
            pi = [1, 0]
            probe = 'Does the main moving subject move predominantly from left to right in the image?'
        for selected in [False, True]:
            if eligible:
                a = runtime.extra_logits(it, selected=selected, probe=probe)
                b = runtime.extra_logits(it, selected=selected, reverse=True, probe=probe)
                rec['pair'][selected] = runtime.pair_gate(a, b, pi)
            else:
                rec['pair'][selected] = (1.0, float('nan'), float('nan'))
        rec['pair_eligible'] = eligible
        runtime.torch.save(dict(meta=meta, data=rec), str(path) + '.tmp')
        runtime.os.replace(str(path) + '.tmp', path)
        shared[k] = rec
    return shared

def component_step(runtime, name):

    def step(it, ctx):
        k = runtime.key(it)
        t = runtime.TARGETS_ABL[name][k].to(runtime.DEV)
        s = runtime.student_last(it)
        kd = runtime.rkl(t[None], s[None])
        draw = runtime.random.Random(runtime.seed_of(k) + ctx['g']).random()
        if draw < runtime.CFG['seed_gamma']:
            y = runtime.torch.multinomial(t.softmax(0), 1, generator=runtime.torch.Generator(device=runtime.DEV).manual_seed(runtime.seed_of(k) + ctx['g']))
        else:
            y = runtime.torch.tensor([runtime.gt_tok(it)], device=runtime.DEV)
        ce = runtime.F.cross_entropy(s[None].float(), y.view(1))
        return (kd + ce, dict(rkl=float(kd.detach()), ce=float(ce.detach())))
    return step

def save_results(runtime):
    """Merge ROWS into RES/results.csv: runs from other arms sharing this (a_pos, a_neg) config are kept."""
    path = runtime.RES / 'results.csv'
    new = runtime.pd.DataFrame(runtime.ROWS)
    if path.exists():
        old = runtime.pd.read_csv(path)
        new = runtime.pd.concat([old[~old['run'].isin(set(new['run']))], new], ignore_index=True)
    temp = path.with_name(path.name + f'.tmp.{runtime.os.getpid()}')
    new.to_csv(temp, index=False)
    runtime.os.replace(temp, path)

def student_cfg(runtime, name, seed):
    d = runtime._old_student_cfg(name, seed)
    d.update(source=runtime.SOURCE_HASH, abl=runtime.ABL, flags=runtime.ARMS[name], teacher=runtime.BASE_TEACHER, train_content=[runtime.item_sig(i) for i in runtime.TRAIN], eval_content=[runtime.item_sig(i) for i in runtime.eval_set()], confidence_cuts=runtime._qt.tolist(), target_digest=runtime.hashlib.sha256(b''.join((runtime.TARGETS_ABL[name][runtime.key(i)].numpy().tobytes() for i in runtime.TRAIN))).hexdigest())
    return d

def run(runtime):
    """Append after SEED v5 definitions. Teacher cache built once; labels TRAIN only."""
    from pathlib import Path
    runtime.Path = Path
    from tqdm.auto import tqdm
    runtime.tqdm = tqdm
    import copy
    runtime.copy = copy
    runtime.ABL = dict(version='seed_components_v1', top_fraction=0.25, gain_tau=1.0, pair_mode='direction_proxy', seeds=[0])
    runtime.ARMS = {'seed_base': (0, 0, 0, 0), 'token_only': (1, 0, 0, 0), 'pair_only': (0, 1, 0, 0), 'adaptive_pos_only': (0, 0, 1, 0), 'adaptive_neg_only': (0, 0, 0, 1), 'token_pair': (1, 1, 0, 0), 'adaptive_pos_neg': (0, 0, 1, 1), 'all_components': (1, 1, 1, 1)}
    assert 'T_pos_tcd' in runtime.TEACHERS
    assert runtime.TEACHERS['T_pos_tcd']['pos'] == 'orig+fg'
    assert set(runtime.TEACHERS['T_pos_tcd']['negs']) == {'tcd'}
    assert not hasattr(runtime.model, 'peft_config'), 'Unload adapters first.'
    assert 0 < runtime.ABL['top_fraction'] <= 1 and runtime.ABL['gain_tau'] > 0
    runtime.OUT = runtime.Path(runtime.CACHE) / 'seed_component_ablation'
    runtime.OUT.mkdir(exist_ok=True)
    runtime.BASE_TEACHER = runtime.copy.deepcopy(runtime.TEACHERS['T_pos_tcd'])
    runtime.RUN_TAG = f"ap{runtime.BASE_TEACHER['a_pos']:g}_an{runtime.BASE_TEACHER['negs']['tcd']:g}_tf{runtime.CFG['teacher_frames']}"
    runtime.RES = runtime.OUT / 'runs' / runtime.RUN_TAG
    runtime.RES.mkdir(parents=True, exist_ok=True)
    runtime.ARM_RUN = list(getattr(runtime.args, 'arms', None) or runtime.ARMS)
    print('Resolved original teacher:', runtime.BASE_TEACHER)
    print('Run tag:', runtime.RUN_TAG, '| arms to train:', runtime.ARM_RUN, '| results:', runtime.RES)
    print('Original LoRA targets:', runtime.LORA_TARGETS)
    print('Original SEED CE probability:', runtime.CFG['seed_gamma'])
    runtime.file_sha = MethodType(file_sha, runtime)
    runtime.VIDEO_SHA = {p: runtime.file_sha(p) for p in sorted({i['video'] for i in runtime.TRAIN + runtime.VAL + runtime.TEST + runtime.EH})}
    runtime.item_sig = MethodType(item_sig, runtime)
    runtime._original_feature = runtime.feature_enhance
    runtime.TOKEN_SELECTED = False
    runtime.TOKEN_STATS = []
    runtime.feature_enhance = MethodType(feature_enhance, runtime)
    runtime.extra_logits = MethodType(extra_logits, runtime)
    runtime.norm_jsd = MethodType(norm_jsd, runtime)
    runtime.pair_gate = MethodType(pair_gate, runtime)
    runtime.correct_margin = MethodType(correct_margin, runtime)
    runtime.prepare_shared = MethodType(prepare_shared, runtime)
    runtime.SHARED = runtime.prepare_shared()
    runtime.coverage = sum((r['pair_eligible'] for r in runtime.SHARED.values()))
    print('Pair gate eligible:', runtime.coverage, '/', len(runtime.TRAIN), 'mode:', runtime.ABL['pair_mode'])
    assert runtime.coverage, 'No eligible pair samples. Add verified pi metadata or use direction_proxy.'
    runtime._conf = {k: runtime.torch.log_softmax(r['orig'], 0).max().item() for k, r in runtime.SHARED.items()}
    runtime._qt = runtime.np.quantile(list(runtime._conf.values()), runtime.np.linspace(0, 1, runtime.CFG['seed_groups'] + 1)[1:-1])
    runtime.TARGETS_ABL = {}
    runtime.AUDIT = []
    for runtime.name, runtime.flags in runtime.ARMS.items():
        runtime.selected, runtime.pair, runtime.adapt_pos, runtime.adapt_neg = runtime.flags
        runtime.TARGETS_ABL[runtime.name] = {}
        for runtime.it in runtime.TRAIN:
            runtime.k = runtime.key(runtime.it)
            runtime.r = runtime.SHARED[runtime.k]
            runtime.o = runtime.r['orig']
            runtime.p = runtime.r['selected'] if runtime.selected else runtime.r['orig+fg']
            runtime.n = runtime.r['tcd']
            runtime.ap = float(runtime.BASE_TEACHER['a_pos'])
            runtime.an = float(runtime.BASE_TEACHER['negs']['tcd'])
            runtime.gt = runtime.gt_tok(runtime.it)
            runtime.gp = runtime.correct_margin(runtime.p, runtime.gt) - runtime.correct_margin(runtime.o, runtime.gt)
            runtime.ref = runtime.o + runtime.ap * (runtime.p - runtime.o)
            runtime.gn = runtime.correct_margin(runtime.ref + runtime.an * (runtime.o - runtime.n), runtime.gt) - runtime.correct_margin(runtime.ref, runtime.gt)
            if runtime.adapt_pos:
                runtime.ap *= min(1.0, max(0.0, runtime.gp / runtime.ABL['gain_tau']))
            if runtime.adapt_neg:
                runtime.an *= min(1.0, max(0.0, runtime.gn / runtime.ABL['gain_tau']))
            runtime.gate, runtime.d, runtime.e = runtime.r['pair'][bool(runtime.selected)]
            if runtime.pair:
                runtime.an *= runtime.gate
            runtime.z = runtime.o + runtime.seed_beta(runtime.k) * (runtime.ap * (runtime.p - runtime.o) + runtime.an * (runtime.o - runtime.n))
            assert runtime.torch.isfinite(runtime.z).all()
            runtime.TARGETS_ABL[runtime.name][runtime.k] = runtime.z
            runtime.AUDIT.append(dict(arm=runtime.name, id=runtime.k, ap=runtime.ap, an=runtime.an, gain_pos=runtime.gp, gain_neg=runtime.gn, gate=runtime.gate, sensitivity=runtime.d, equivariance_error=runtime.e, pair_eligible=runtime.r['pair_eligible'], target_margin=runtime.margin(runtime.z), base_margin=runtime.margin(runtime.o), gt=runtime.it['answer']))
    runtime.pd.DataFrame(runtime.AUDIT).to_csv(runtime.RES / 'target_diagnostics_train.csv', index=False)
    for runtime.it in runtime.TRAIN[:5]:
        runtime.k = runtime.key(runtime.it)
        runtime.r = runtime.SHARED[runtime.k]
        runtime.expected = runtime.r['orig'] + runtime.seed_beta(runtime.k) * (runtime.BASE_TEACHER['a_pos'] * (runtime.r['orig+fg'] - runtime.r['orig']) + runtime.BASE_TEACHER['negs']['tcd'] * (runtime.r['orig'] - runtime.r['tcd']))
        assert runtime.torch.equal(runtime.TARGETS_ABL['seed_base'][runtime.k], runtime.expected)
    runtime.component_step = MethodType(component_step, runtime)
    runtime._old_student_cfg = runtime.student_cfg
    runtime.student_cfg = MethodType(student_cfg, runtime)
    for runtime.name in runtime.ARMS:
        runtime.METHODS[runtime.name] = (lambda name=runtime.name: runtime.component_step(name), None, None, True)
    runtime.STUDENT_DIR = str(runtime.OUT / 'students')
    runtime.os.makedirs(runtime.STUDENT_DIR, exist_ok=True)
    runtime.base_meta = dict(source=runtime.SOURCE_HASH, branches=runtime.branch_cfg('orig'), items=[runtime.item_sig(i) for i in runtime.eval_set()])
    runtime.base_path = runtime.OUT / ('base_' + runtime.cfg_hash(runtime.base_meta) + '.pt')
    if runtime.base_path.exists():
        runtime.MARG['base'] = runtime.torch.load(runtime.base_path, weights_only=False)
    else:
        runtime.MARG['base'] = runtime.eval_items(runtime.eval_set())
        runtime.torch.save(runtime.MARG['base'], runtime.base_path)
    runtime.SPLITS = {'VH_VAL': runtime.VAL, 'VH_TEST': runtime.TEST, 'EH_ALL': runtime.EH}
    for runtime.sub in sorted({i['subset'] for i in runtime.EH}):
        runtime.SPLITS[runtime.sub] = [i for i in runtime.EH if i['subset'] == runtime.sub]
    runtime.result_rows = MethodType(result_rows, runtime)
    runtime.save_results = MethodType(save_results, runtime)
    runtime.ROWS = runtime.result_rows('base', runtime.MARG['base'])
    for runtime.name in runtime.ARM_RUN:
        for runtime.seed in runtime.ABL['seeds']:
            runtime.out = runtime.train_arm(runtime.name, runtime.seed)
            runtime.tag = f'{runtime.name}_s{runtime.seed}'
            runtime.MARG[runtime.tag] = runtime.out['marg']
            runtime.rows = runtime.result_rows(runtime.tag, runtime.out['marg'])
            runtime.ROWS.extend(runtime.rows)
            runtime.display(runtime.pd.DataFrame(runtime.rows))
            runtime.save_results()
            runtime.pd.DataFrame([dict(id=runtime.key(i), video=i['video'], question=i['question'], answer=i['answer'], base_margin=runtime.MARG['base'][runtime.key(i)], student_margin=runtime.out['marg'][runtime.key(i)]) for i in runtime.eval_set()]).to_csv(runtime.RES / f'{runtime.tag}_predictions.csv', index=False)
            runtime.gc.collect()
            runtime.torch.cuda.empty_cache()
    runtime.display(runtime.pd.DataFrame(runtime.ROWS))
    print('Saved per-arm results and adapters:', runtime.OUT)
