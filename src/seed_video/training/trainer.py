"""training: trainer stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType
from seed_video.evaluation.inference import eval_items, eval_set

def _wrap(runtime):
    runtime.model = runtime.get_peft_model(runtime.model, runtime.LoraConfig(r=runtime.CFG['lora_r'], lora_alpha=runtime.CFG['lora_alpha'], lora_dropout=runtime.CFG['lora_dropout'], target_modules=runtime.LORA_TARGETS, bias='none'))

def _unwrap(runtime):
    runtime.model = runtime.model.unload()
    runtime.model.config.use_cache = True
    if hasattr(runtime.model, 'peft_config'):
        del runtime.model.peft_config

def student_cfg(runtime, name, seed):
    T = runtime.METHODS[name][1]
    d = dict(code=runtime.CODE_VERSION, name=name, seed=seed, lora_targets=runtime.LORA_TARGETS, hp={k: runtime.CFG[k] for k in runtime.STUDENT_HP}, train=[runtime.key(i) for i in runtime.TRAIN])
    if T is not None:
        spec = runtime.TEACHERS[T]
        d['teacher'] = dict(name=T, spec=spec, branches={v: runtime.branch_cfg(v) for v in runtime.views_of(spec)})
    return d

def arm_paths(runtime, name, seed):
    stem = f"{runtime.STUDENT_DIR}/{name.replace('@', '__')}_s{seed}_{runtime.cfg_hash(runtime.student_cfg(name, seed))}"
    return (stem + '.pt', stem + '_adapter.pt', stem + '.json')

def train_arm(runtime, name, seed):
    tag = f'{name}_s{seed}'
    path, apath, jpath = runtime.arm_paths(name, seed)
    ev = runtime.eval_set()
    if runtime.os.path.exists(path):
        out = runtime.torch.load(path, weights_only=False)
        miss = [i for i in ev if runtime.key(i) not in out['marg']]
        if not miss:
            print(f'[{tag}] dùng cache {runtime.os.path.basename(path)}')
            return out
        if runtime.os.path.exists(apath):
            runtime._wrap()
            runtime.set_peft_model_state_dict(runtime.model, runtime.torch.load(apath, weights_only=False))
            try:
                out['marg'].update(runtime.eval_items(miss))
            finally:
                runtime._unwrap()
            runtime.torch.save(out, path)
            print(f'[{tag}] dùng cache + chấm thêm {len(miss)} câu')
            return out
        print(f'[{tag}] cache không có adapter và thiếu {len(miss)} câu → train lại')
    runtime.set_seed(seed)
    runtime._wrap()
    runtime.model.config.use_cache = False
    if runtime.CFG['grad_ckpt']:
        runtime.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        runtime.model.enable_input_require_grads()
    step = runtime.METHODS[name][0]()
    select = runtime.METHODS[name][2]() if runtime.METHODS[name][2] else lambda items, ep: items
    params = [p for p in runtime.model.parameters() if p.requires_grad]
    opt = runtime.torch.optim.AdamW(params, lr=runtime.CFG['lr'], weight_decay=0.0)
    total = runtime.CFG['epochs'] * len(runtime.TRAIN)
    sched = runtime.get_cosine_schedule_with_warmup(opt, int(0.05 * total / runtime.CFG['grad_accum']), runtime.math.ceil(total / runtime.CFG['grad_accum']))
    rng, log, g, t0 = (runtime.random.Random(seed), [], 0, runtime.time.time())
    runtime.model.train()
    try:
        for ep in range(runtime.CFG['epochs']):
            order = select(runtime.TRAIN[:], ep)
            rng.shuffle(order)
            runtime.model.train()
            for i, it in enumerate(order):
                loss, parts = step(it, dict(ep=ep, g=g, progress=g / total))
                g += 1
                if loss is not None:
                    (loss / runtime.CFG['grad_accum']).backward()
                    log.append(dict(ep=ep, loss=loss.item(), **parts))
                if (i + 1) % runtime.CFG['grad_accum'] == 0 or i == len(order) - 1:
                    runtime.torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
            d = runtime.pd.DataFrame(log)
            d = d[d.ep == ep].drop(columns='ep').mean().round(5).to_dict()
            print(f'[{tag}] epoch {ep}: {d} ({(runtime.time.time() - t0) / 60:.1f} phút)')
        t1 = runtime.time.time()
        marg = runtime.eval_items(ev)
        out = dict(marg=marg, log=log, train_min=(t1 - t0) / 60, sec_per_q=(runtime.time.time() - t1) / len(ev))
        runtime.torch.save({k: v.detach().cpu() for k, v in runtime.get_peft_model_state_dict(runtime.model).items()}, apath)
        runtime.torch.save(out, path)
        runtime.json.dump(runtime.student_cfg(name, seed), open(jpath, 'w'), indent=1, default=str)
    finally:
        if runtime.CFG['grad_ckpt']:
            runtime.model.gradient_checkpointing_disable()
            runtime.model.disable_input_require_grads()
        runtime._unwrap()
    d = abs(runtime.margin(runtime.student_last(runtime.TEST[0], grad=False)) - runtime.MARG['base'][runtime.key(runtime.TEST[0])])
    assert d < 0.1, f'unload không trả về base (|Δmargin|={d})'
    return out

def run(runtime):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    runtime.get_peft_model_state_dict = get_peft_model_state_dict
    runtime.set_peft_model_state_dict = set_peft_model_state_dict
    runtime._wrap = MethodType(_wrap, runtime)
    runtime._unwrap = MethodType(_unwrap, runtime)
    runtime.eval_items = MethodType(eval_items, runtime)
    runtime.STUDENT_HP = ['model_id', 'n_frames', 'frame_size', 'lora_r', 'lora_alpha', 'lora_dropout', 'lr', 'epochs', 'grad_accum', 'hinton_T', 'p_open', 'open_prompt', 'open_max_new', 'yn_max_new', 'gen_temp', 'jsd_beta', 'tvdf_keep', 'tvdf_use_gt', 'seed_groups', 'seed_betas', 'seed_gamma', 'plaus_beta', 'seq_plaus', 'opsd_anchor']
    runtime.student_cfg = MethodType(student_cfg, runtime)
    runtime.STUDENT_DIR = f'{runtime.CACHE}/students'
    runtime.os.makedirs(runtime.STUDENT_DIR, exist_ok=True)
    runtime.arm_paths = MethodType(arm_paths, runtime)
    runtime.eval_set = MethodType(eval_set, runtime)
    runtime.train_arm = MethodType(train_arm, runtime)
