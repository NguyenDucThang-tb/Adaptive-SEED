"""training: trainer stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType
from seed_video.evaluation.inference import eval_items, eval_set


def _atomic_torch_save(runtime, value, path):
    temp = f"{path}.tmp.{runtime.os.getpid()}"
    try:
        runtime.torch.save(value, temp)
        runtime.os.replace(temp, path)
    finally:
        if runtime.os.path.exists(temp):
            runtime.os.unlink(temp)


def _training_rng_state(runtime, rng):
    return dict(
        python=runtime.random.getstate(),
        numpy=runtime.np.random.get_state(),
        torch=runtime.torch.get_rng_state(),
        cuda=runtime.torch.cuda.get_rng_state_all(),
        local=rng.getstate(),
    )


def _restore_training_rng_state(runtime, rng, state):
    runtime.random.setstate(state['python'])
    runtime.np.random.set_state(state['numpy'])
    runtime.torch.set_rng_state(state['torch'])
    runtime.torch.cuda.set_rng_state_all(state['cuda'])
    rng.setstate(state['local'])


def _checkpoint_interval(name, default):
    import os

    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f'{name} must be >= 1')
    return value


def _valid_order_keys(order_keys, item_by_key):
    return (
        len(order_keys) <= len(item_by_key)
        and len(set(order_keys)) == len(order_keys)
        and all(key in item_by_key for key in order_keys)
    )

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
    cfg = runtime.student_cfg(name, seed)
    train_checkpoint_path = path + '.train_checkpoint.pt'
    eval_checkpoint_path = path + '.eval_checkpoint.pt'
    save_every_steps = _checkpoint_interval('SEED_CHECKPOINT_EVERY_STEPS', 5)
    save_every_eval = _checkpoint_interval('SEED_EVAL_CHECKPOINT_EVERY', 25)

    if runtime.os.path.exists(path):
        out = runtime.torch.load(path, weights_only=False)
        miss = [i for i in ev if runtime.key(i) not in out['marg']]
        if not miss:
            print(f'[{tag}] dùng cache {runtime.os.path.basename(path)}')
            return out
        print(f'[{tag}] đánh giá tiếp {len(miss)} câu còn thiếu')

    out = runtime.torch.load(path, weights_only=False) if runtime.os.path.exists(path) else None
    train_min = float(out.get('train_min', 0.0)) if out else 0.0
    log = out.get('log', []) if out else []
    marg = dict(out.get('marg', {})) if out else {}

    if runtime.os.path.exists(eval_checkpoint_path):
        eval_state = runtime.torch.load(eval_checkpoint_path, map_location='cpu', weights_only=False)
        if eval_state.get('cfg') != cfg or eval_state.get('eval_keys') != [runtime.key(i) for i in ev]:
            raise RuntimeError(f'Checkpoint đánh giá không khớp cấu hình: {eval_checkpoint_path}')
        marg.update(eval_state.get('marg', {}))

    runtime.set_seed(seed)
    runtime._wrap()
    runtime.model.config.use_cache = False
    if runtime.CFG['grad_ckpt']:
        runtime.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        runtime.model.enable_input_require_grads()

    try:
        if not runtime.os.path.exists(apath):
            step = runtime.METHODS[name][0]()
            select = runtime.METHODS[name][2]() if runtime.METHODS[name][2] else lambda items, ep: items
            params = [p for p in runtime.model.parameters() if p.requires_grad]
            opt = runtime.torch.optim.AdamW(params, lr=runtime.CFG['lr'], weight_decay=0.0)
            total = runtime.CFG['epochs'] * len(runtime.TRAIN)
            sched = runtime.get_cosine_schedule_with_warmup(
                opt,
                int(0.05 * total / runtime.CFG['grad_accum']),
                runtime.math.ceil(total / runtime.CFG['grad_accum']),
            )
            rng = runtime.random.Random(seed)
            epoch, next_index, order_keys, g, opt_steps = 0, 0, None, 0, 0
            checkpoint = None
            if runtime.os.path.exists(train_checkpoint_path):
                checkpoint = runtime.torch.load(
                    train_checkpoint_path, map_location='cpu', weights_only=False
                )
                if checkpoint.get('cfg') != cfg:
                    raise RuntimeError(f'Checkpoint train không khớp cấu hình: {train_checkpoint_path}')
                runtime.set_peft_model_state_dict(runtime.model, checkpoint['adapter'])
                opt.load_state_dict(checkpoint['optimizer'])
                sched.load_state_dict(checkpoint['scheduler'])
                epoch = checkpoint['epoch']
                next_index = checkpoint['next_index']
                order_keys = checkpoint['order_keys']
                g = checkpoint['g']
                opt_steps = checkpoint['opt_steps']
                log = checkpoint['log']
                train_min = float(checkpoint.get('train_elapsed_sec', 0.0)) / 60
                if hasattr(runtime, 'restore_method_state'):
                    runtime.restore_method_state(checkpoint.get('method_state'))
                _restore_training_rng_state(runtime, rng, checkpoint['rng'])
                print(
                    f'[{tag}] resume checkpoint: epoch={epoch}, '
                    f'next_sample={next_index}, optimizer_steps={opt_steps}'
                )

            item_by_key = {runtime.key(item): item for item in runtime.TRAIN}
            started = runtime.time.time()
            while epoch < runtime.CFG['epochs']:
                if order_keys is None:
                    order = select(runtime.TRAIN[:], epoch)
                    rng.shuffle(order)
                    order_keys = [runtime.key(item) for item in order]
                    next_index = 0
                if not _valid_order_keys(order_keys, item_by_key):
                    raise RuntimeError(f'Checkpoint train order không khớp dữ liệu: {train_checkpoint_path}')
                order = [item_by_key[k] for k in order_keys]
                runtime.model.train()
                for i in range(next_index, len(order)):
                    it = order[i]
                    loss, parts = step(it, dict(ep=epoch, g=g, progress=g / total))
                    g += 1
                    if loss is not None:
                        (loss / runtime.CFG['grad_accum']).backward()
                        log.append(dict(ep=epoch, loss=loss.item(), **parts))
                    if (i + 1) % runtime.CFG['grad_accum'] == 0 or i == len(order) - 1:
                        runtime.torch.nn.utils.clip_grad_norm_(params, 1.0)
                        opt.step()
                        sched.step()
                        opt.zero_grad(set_to_none=True)
                        opt_steps += 1
                        next_index = i + 1

                        if opt_steps % save_every_steps == 0:
                            _atomic_torch_save(runtime, dict(
                                cfg=cfg,
                                adapter={k: v.detach().cpu() for k, v in runtime.get_peft_model_state_dict(runtime.model).items()},
                                optimizer=opt.state_dict(),
                                scheduler=sched.state_dict(),
                                epoch=epoch,
                                next_index=next_index,
                                order_keys=order_keys,
                                g=g,
                                opt_steps=opt_steps,
                                log=log,
                                train_elapsed_sec=train_min * 60 + runtime.time.time() - started,
                                rng=_training_rng_state(runtime, rng),
                                method_state=runtime.capture_method_state() if hasattr(runtime, 'capture_method_state') else None,
                            ), train_checkpoint_path)

                d = runtime.pd.DataFrame(log)
                d = d[d.ep == epoch].drop(columns='ep').mean().round(5).to_dict()
                print(f'[{tag}] epoch {epoch}: {d} ({train_min + (runtime.time.time() - started) / 60:.1f} phút)')
                epoch += 1
                next_index, order_keys = 0, None
                _atomic_torch_save(runtime, dict(
                    cfg=cfg,
                    adapter={k: v.detach().cpu() for k, v in runtime.get_peft_model_state_dict(runtime.model).items()},
                    optimizer=opt.state_dict(),
                    scheduler=sched.state_dict(),
                    epoch=epoch,
                    next_index=next_index,
                    order_keys=order_keys,
                    g=g,
                    opt_steps=opt_steps,
                    log=log,
                    train_elapsed_sec=train_min * 60 + runtime.time.time() - started,
                    rng=_training_rng_state(runtime, rng),
                    method_state=runtime.capture_method_state() if hasattr(runtime, 'capture_method_state') else None,
                ), train_checkpoint_path)

            train_min += (runtime.time.time() - started) / 60
            _atomic_torch_save(runtime, {
                k: v.detach().cpu()
                for k, v in runtime.get_peft_model_state_dict(runtime.model).items()
            }, apath)
            temp_json = f'{jpath}.tmp.{runtime.os.getpid()}'
            try:
                with open(temp_json, 'w', encoding='utf-8') as handle:
                    runtime.json.dump(cfg, handle, indent=1, default=str)
                runtime.os.replace(temp_json, jpath)
            finally:
                if runtime.os.path.exists(temp_json):
                    runtime.os.unlink(temp_json)
            if runtime.os.path.exists(train_checkpoint_path):
                runtime.os.unlink(train_checkpoint_path)
        else:
            runtime.set_peft_model_state_dict(
                runtime.model, runtime.torch.load(apath, map_location='cpu', weights_only=False)
            )
            print(f'[{tag}] dùng adapter đã train; tiếp tục đánh giá')

        eval_keys = [runtime.key(i) for i in ev]
        miss = [i for i in ev if runtime.key(i) not in marg]
        eval_started = runtime.time.time()
        runtime.model.eval()
        with runtime.torch.no_grad():
            for start in range(0, len(miss), save_every_eval):
                batch = miss[start:start + save_every_eval]
                marg.update(runtime.eval_items(batch))
                _atomic_torch_save(runtime, dict(cfg=cfg, eval_keys=eval_keys, marg=marg), eval_checkpoint_path)

        out = dict(
            marg=marg,
            log=log,
            train_min=train_min,
            sec_per_q=(runtime.time.time() - eval_started) / max(1, len(miss)),
        )
        _atomic_torch_save(runtime, out, path)
        if runtime.os.path.exists(eval_checkpoint_path):
            runtime.os.unlink(eval_checkpoint_path)
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
