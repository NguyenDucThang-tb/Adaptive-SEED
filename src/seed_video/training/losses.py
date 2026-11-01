"""training: losses stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType

def kl(runtime, t, s, T=1.0):
    lt, ls = (runtime.F.log_softmax(t.float() / T, -1), runtime.F.log_softmax(s.float() / T, -1))
    return (lt.exp() * (lt - ls)).sum(-1).mean()

def rkl(runtime, t, s):
    lt, ls = (runtime.F.log_softmax(t.float(), -1), runtime.F.log_softmax(s.float(), -1))
    return (ls.exp() * (ls - lt)).sum(-1).mean()

def jsd(runtime, t, s, b):
    p, q = (runtime.F.softmax(t.float(), -1), runtime.F.softmax(s.float(), -1))
    m = (b * p + (1 - b) * q).clamp_min(1e-12).log()
    return (b * (p * (p.clamp_min(1e-12).log() - m)).sum(-1) + (1 - b) * (q * (q.clamp_min(1e-12).log() - m)).sum(-1)).mean()

def student_last(runtime, it, grad=True):
    return runtime.fwd_view(it, 'orig', grad=grad)[0].logits[0, -1]

def gt_tok(runtime, it):
    return runtime.YES if it['answer'] == 'yes' else runtime.NO

def step_kd_basic(runtime, T, perm=False):

    def f(it, ctx):
        t = runtime.tlog_perm(runtime.key(it), T) if perm else runtime.tlog(runtime.key(it), T)
        l = runtime.kl(t[None], runtime.student_last(it)[None], runtime.CFG['hinton_T']) * runtime.CFG['hinton_T'] ** 2
        return (l, dict(kd=l.item()))
    return f

def step_sft(runtime):

    def f(it, ctx):
        l = runtime.F.cross_entropy(runtime.student_last(it)[None].float(), runtime.torch.tensor([runtime.gt_tok(it)], device=runtime.DEV))
        return (l, dict(ce=l.item()))
    return f

def _reset_rope(runtime, m):
    for mod in m.modules():
        if hasattr(mod, 'rope_deltas'):
            mod.rope_deltas = None

def sample_response(runtime, it, ctx):
    open_q = runtime.random.Random(runtime.seed_of(runtime.key(it)) + ctx['g']).random() < runtime.CFG['p_open']
    q = runtime.CFG['open_prompt'] if open_q else None
    fr, fps, _ = runtime.view_frames(it, 'orig')
    inp = runtime.build_inputs(fr, q or it['question'], fps, yn=q is None).to(runtime.DEV)
    runtime.model.eval()
    with runtime.torch.no_grad(), runtime.keep_last(1):
        gen = runtime.model.generate(**inp, max_new_tokens=runtime.CFG['open_max_new'] if open_q else runtime.CFG['yn_max_new'], do_sample=True, temperature=runtime.CFG['gen_temp'], top_p=1.0, top_k=0, repetition_penalty=1.0)
    runtime._reset_rope(runtime.model)
    return (q, open_q, gen[0, inp['input_ids'].shape[1]:])

def views_of(runtime, spec):
    return ['orig'] + ([spec['pos']] if spec['pos'] else []) + list(spec['negs'])

def seq_logits(runtime, it, views, resp, q, adapter_on):
    n = resp.numel()
    ctx = runtime.contextlib.nullcontext() if adapter_on else runtime.adapter_off()
    with runtime.torch.no_grad(), ctx:
        out = {v: runtime.fwd_view(it, v, resp=resp, question=q)[0].logits[0, -n - 1:-1].float() for v in views}
    runtime._reset_rope(runtime.model)
    return out

def compose_seq(runtime, seq, spec):
    lo = seq['orig']
    lt = runtime.compose(lambda v: seq[v], spec)
    if not runtime.CFG['seq_plaus']:
        return lt
    po = runtime.F.softmax(lo, -1)
    keep = po >= runtime.CFG['plaus_beta'] * po.max(-1, keepdim=True).values
    return runtime.torch.where(keep, lt, lo)

def yn_margins(runtime, s, t, open_q):
    if open_q:
        return {}
    return dict(m_S=(s[0, runtime.YES] - s[0, runtime.NO]).item(), m_T=(t[0, runtime.YES] - t[0, runtime.NO]).item())

def step_onpolicy(runtime, T, div='jsd'):
    spec = runtime.TEACHERS[T]

    def f(it, ctx):
        q, open_q, resp = runtime.sample_response(it, ctx)
        n = resp.numel()
        if n == 0:
            runtime.model.train()
            return (None, {})
        t = runtime.compose_seq(runtime.seq_logits(it, runtime.views_of(spec), resp, q, adapter_on=False), spec)
        runtime.model.train()
        s = runtime.fwd_view(it, 'orig', grad=True, resp=resp, question=q)[0].logits[0, -n - 1:-1]
        l = runtime.rkl(t, s) if div == 'rkl' else runtime.jsd(t, s, runtime.CFG['jsd_beta'])
        return (l, dict(div=l.item(), resp_len=n, open=int(open_q), **runtime.yn_margins(s.detach(), t, open_q)))
    return f

def step_opsd(runtime, T):
    spec = runtime.TEACHERS[T]
    lam = runtime.CFG['opsd_anchor']

    def f(it, ctx):
        q, open_q, resp = runtime.sample_response(it, ctx)
        n = resp.numel()
        if n == 0:
            runtime.model.train()
            return (None, {})
        t = runtime.compose_seq(runtime.seq_logits(it, runtime.views_of(spec), resp, q, adapter_on=True), spec)
        b = runtime.seq_logits(it, ['orig'], resp, q, adapter_on=False)['orig'] if lam > 0 else None
        runtime.model.train()
        s = runtime.fwd_view(it, 'orig', grad=True, resp=resp, question=q)[0].logits[0, -n - 1:-1]
        l_t = runtime.jsd(t, s, runtime.CFG['jsd_beta'])
        l_a = runtime.rkl(b, s) if lam > 0 else runtime.torch.zeros((), device=s.device)
        return (l_t + lam * l_a, dict(div=l_t.item(), anchor=l_a.item(), resp_len=n, open=int(open_q), **runtime.yn_margins(s.detach(), t, open_q)))
    return f

def tvdf_select(runtime, T):
    tau = float(runtime.np.median([abs(runtime.MARG[f'{T}_soft'][runtime.key(i)]) for i in runtime.TRAIN]))

    def sel(items, ep):
        if ep == 0:
            return items
        runtime.model.eval()
        rows = []
        with runtime.torch.no_grad():
            for it in items:
                t = runtime.tlog(runtime.key(it), T)
                s = runtime.student_last(it, grad=False)
                mT = runtime.margin(t)
                ok = (mT > 0) == (it['answer'] == 'yes') if runtime.CFG['tvdf_use_gt'] else abs(mT) >= tau
                rows.append((ok, runtime.rkl(t[None], s[None]).item(), it))
        rel = sorted([r for r in rows if r[0]], key=lambda r: -r[1])
        keep = [r[2] for r in rel[:int(runtime.CFG['tvdf_keep'] * len(items))]]
        print(f'  TVDF epoch {ep}: {len(rel)} mẫu đáng tin → giữ {len(keep)}')
        return keep or items
    return sel

def seed_beta(runtime, k):
    return runtime.CFG['seed_betas'][int(runtime.np.searchsorted(runtime._qt, runtime._conf[k]))]

def seed_logits(runtime, k, T):
    lo = runtime.L['orig'][k].float().to(runtime.DEV)
    return lo + runtime.seed_beta(k) * (runtime.tlog(k, T) - lo)

def step_seed(runtime, T, use_gt=True):

    def f(it, ctx):
        k = runtime.key(it)
        lp = runtime.seed_logits(k, T)
        s = runtime.student_last(it)
        l_d = runtime.rkl(lp[None], s[None])
        r = runtime.random.Random(runtime.seed_of(k) + ctx['g']).random()
        if not use_gt or r < runtime.CFG['seed_gamma']:
            y = runtime.torch.multinomial(runtime.F.softmax(lp, -1), 1, generator=runtime.torch.Generator(device=runtime.DEV).manual_seed(runtime.seed_of(k) + ctx['g']))
        else:
            y = runtime.torch.tensor([runtime.gt_tok(it)], device=runtime.DEV)
        l_s = runtime.F.cross_entropy(s[None].float(), y.view(1))
        return (l_d + l_s, dict(rkl=l_d.item(), ce=l_s.item(), beta=runtime.seed_beta(k)))
    return f

def register(runtime, m, T):
    mk, sel, gt = runtime.METHOD_DEF[m]
    runtime.METHODS[runtime.arm_name(m, T)] = (lambda mk=mk, T=T: mk(T), T, (lambda sel=sel, T=T: sel(T)) if sel else None, gt)

def run(runtime):
    from peft import LoraConfig, get_peft_model
    runtime.LoraConfig = LoraConfig
    runtime.get_peft_model = get_peft_model
    from transformers import get_cosine_schedule_with_warmup
    runtime.get_cosine_schedule_with_warmup = get_cosine_schedule_with_warmup
    if runtime.CFG['model_id'].startswith('OpenGVLab/InternVL3_5-'):
        runtime.LORA_TARGETS = r'(.*language_model\.model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*mlp1\.(1|3))'
    else:
        runtime.LORA_TARGETS = '(.*\\.layers\\.\\d+\\.self_attn\\.(q|k|v|o)_proj)|(.*visual\\.merger\\.(mlp\\.(0|2)|linear_fc[12]))'
    runtime.kl = MethodType(kl, runtime)
    runtime.rkl = MethodType(rkl, runtime)
    runtime.jsd = MethodType(jsd, runtime)
    runtime.student_last = MethodType(student_last, runtime)
    runtime.gt_tok = MethodType(gt_tok, runtime)
    runtime.step_kd_basic = MethodType(step_kd_basic, runtime)
    runtime.step_sft = MethodType(step_sft, runtime)
    runtime._reset_rope = MethodType(_reset_rope, runtime)
    runtime.sample_response = MethodType(sample_response, runtime)
    runtime.views_of = MethodType(views_of, runtime)
    runtime.seq_logits = MethodType(seq_logits, runtime)
    runtime.compose_seq = MethodType(compose_seq, runtime)
    runtime.yn_margins = MethodType(yn_margins, runtime)
    runtime.step_onpolicy = MethodType(step_onpolicy, runtime)
    runtime.step_opsd = MethodType(step_opsd, runtime)
    runtime.tvdf_select = MethodType(tvdf_select, runtime)
    runtime._conf = {}
    runtime._qt = []
    runtime.seed_beta = MethodType(seed_beta, runtime)
    runtime.seed_logits = MethodType(seed_logits, runtime)
    runtime.step_seed = MethodType(step_seed, runtime)
    runtime.METHOD_DEF = {'kd_basic': (lambda T: runtime.step_kd_basic(T), None, False), 'kd_perm': (lambda T: runtime.step_kd_basic(T, perm=True), None, False), 'seed': (lambda T: runtime.step_seed(T, True), None, True), 'seed_nolabel': (lambda T: runtime.step_seed(T, False), None, False), 'onpolicy_gkd': (lambda T: runtime.step_onpolicy(T, 'jsd'), None, False), 'video_opd': (lambda T: runtime.step_onpolicy(T, 'rkl'), lambda T: runtime.tvdf_select(T), runtime.CFG['tvdf_use_gt']), 'opsd': (lambda T: runtime.step_opsd(T), None, False)}
    runtime.register = MethodType(register, runtime)
    runtime.METHODS = {'sft_gt': (lambda: runtime.step_sft(), None, None, True)}
    for runtime.m in runtime.CFG['methods']:
        for runtime.T in runtime.CFG['teachers']:
            runtime.register(runtime.m, runtime.T)
    runtime.missing = [a for a in runtime.CFG['run'] if a not in runtime.METHODS]
    assert not runtime.missing, f'arm thiếu teacher/không tồn tại: {runtime.missing}'
