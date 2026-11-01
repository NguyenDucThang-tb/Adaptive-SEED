"""config stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType

def seed_of(runtime, s):
    return int(runtime.hashlib.md5(s.encode()).hexdigest()[:8], 16)

def set_seed(runtime, s):
    runtime.random.seed(s)
    runtime.np.random.seed(s)
    runtime.torch.manual_seed(s)
    runtime.torch.cuda.manual_seed_all(s)

def cfg_hash(runtime, d):
    return runtime.hashlib.md5(runtime.json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]

def arm_name(runtime, method, teacher):
    return f'{method}@{teacher}'

def run(runtime):
    from pathlib import Path
    runtime.Path = Path
    import os, json, math, random, time, gc, hashlib, glob, contextlib
    runtime.os = os
    runtime.json = json
    runtime.math = math
    runtime.random = random
    runtime.time = time
    runtime.gc = gc
    runtime.hashlib = hashlib
    runtime.glob = glob
    runtime.contextlib = contextlib
    from functools import lru_cache
    runtime.lru_cache = lru_cache
    import numpy as np, pandas as pd, torch, torch.nn.functional as F
    runtime.np = np
    runtime.pd = pd
    runtime.torch = torch
    runtime.F = F
    from PIL import Image
    runtime.Image = Image
    runtime.USE_DRIVE = False
    runtime.FAST = True
    runtime.DEV = 'cuda'
    runtime.torch.backends.cuda.matmul.allow_tf32 = True
    runtime.torch.backends.cudnn.allow_tf32 = True
    runtime.CFG = dict(model_id='Qwen/Qwen3-VL-8B-Instruct', birefnet_id='ZhengPeng7/BiRefNet', hf_repo='bigai-nlco/VideoHallucer', hf_revision='c93475bf1be466c1af90d005749ccb4e6b3ad066', data_root='/content/VideoHallucer', subsets=['temporal', 'object_relation'], download_workers=8, hf_token=None, n_train_pairs=150, n_val_pairs=50, n_test_pairs=300, split_seed=0, n_frames=16, frame_size=448, alpha=0.8, alpha_s=0.8, beta=0.4, fg_thr=0.15, fg_kernel=5, fg_avg_weight=0.7, fg_soft=False, tcd_frames=4, seed_noise_alpha=0.3, a_pos=1, a_tcd=0.25, plaus_beta=0.1, lora_r=16, lora_alpha=32, lora_dropout=0.05, lr=0.0001, epochs=2, grad_accum=8, seeds=[0], hinton_T=2.0, p_open=0.5, open_prompt='Describe what happens in the video in temporal order, mentioning the main objects and actions.', open_max_new=48, yn_max_new=4, gen_temp=1.0, jsd_beta=0.5, tvdf_keep=0.5, tvdf_use_gt=False, seed_groups=4, seed_betas=(1.1, 0.8, 0.6, 0.4), seed_gamma=0.1, grad_ckpt=True, eval_train=True, teachers=['T_pos', 'T_tcd', 'T_pos_tcd'], methods=['kd_basic', 'seed', 'seed_nolabel'], controls=[], seq_plaus=True, opsd_anchor=1.0, control_teachers=['T_base', 'T_posrand_tcd', 'T_noise'], a_noise=0.5, best_method='auto', best_pool='nolabel', eh=dict(enable=True, gdrive_id='1IPmx6Y80UrXwVPmZJh6zjCPHtlsw4p9n', q_commit='aa544c21c7cd93b4685423cb94f77ab441f754bc', splits=['entire', 'mix', 'misleading'], allow_missing=False))
    runtime.CFG['model_id'] = os.environ.get('SEED_MODEL_ID', runtime.CFG['model_id'])
    if runtime.FAST:
        runtime.CFG.update(n_frames=8, tcd_frames=4, epochs=1, grad_ckpt=False, eval_train=True, open_max_new=32)
    smoke_pairs = getattr(runtime.args, 'smoke_pairs', None)
    if smoke_pairs is not None:
        if smoke_pairs < 1:
            raise ValueError('--smoke-pairs must be >= 1')
        runtime.CFG.update(subsets=['temporal'], n_train_pairs=smoke_pairs, n_val_pairs=1, n_test_pairs=1)
        print(f'SMOKE mode: temporal only; train={smoke_pairs} pair(s), val=1 pair, test=1 pair; EventHallusion disabled')
    teacher_frames = getattr(runtime.args, 'teacher_frames', None) or runtime.CFG['n_frames']
    assert teacher_frames % 2 == 0 and teacher_frames >= runtime.CFG['tcd_frames'], 'teacher_frames phải chẵn và >= tcd_frames'
    runtime.CFG['teacher_frames'] = teacher_frames
    print('student n_frames =', runtime.CFG['n_frames'], '| teacher_frames =', teacher_frames)
    runtime.WORK = runtime.args.work_dir.expanduser().resolve()
    runtime.WORK.mkdir(parents=True, exist_ok=True)
    runtime.CFG['data_root'] = str(runtime.WORK / 'VideoHallucer')
    from seed_video.models import model_spec
    _, runtime.CFG['spatial_stride'] = model_spec(runtime.CFG['model_id'])
    runtime.CFG['feature_temporal_stride'] = 1 if runtime.CFG['model_id'].startswith('OpenGVLab/InternVL3_5-') else 2
    runtime.CFG['eh']['enable'] = not runtime.args.no_eh and smoke_pairs is None
    runtime.CFG['eh']['root'] = str(runtime.WORK / 'EventHallusion')
    runtime.CACHE = str(runtime.WORK / f"cdt_distill_cache_f{runtime.CFG['n_frames']}_{runtime.CFG['frame_size']}")
    runtime.CFG['cache_dir'] = runtime.CACHE
    if not runtime.torch.cuda.is_available() or not runtime.torch.cuda.is_bf16_supported():
        raise RuntimeError('This experiment requires a CUDA GPU supporting BF16.')
    runtime.os.makedirs(f'{runtime.CACHE}/masks', exist_ok=True)
    print('FAST =', runtime.FAST, '| cache:', runtime.CACHE)
    runtime.seed_of = MethodType(seed_of, runtime)
    runtime.set_seed = MethodType(set_seed, runtime)
    runtime.CODE_VERSION = 'v5'
    runtime.cfg_hash = MethodType(cfg_hash, runtime)
    runtime.set_seed(0)
    runtime.arm_name = MethodType(arm_name, runtime)
    runtime.CFG['run'] = list(runtime.CFG['controls']) + [runtime.arm_name(m, t) for m in runtime.CFG['methods'] for t in runtime.CFG['teachers']]
    print(len(runtime.CFG['run']), 'arm:', runtime.CFG['run'])
    runtime.CFG['teachers'] = ['T_pos_tcd']
    runtime.CFG['methods'] = ['seed']
    runtime.CFG['controls'] = []
    runtime.CFG['control_teachers'] = []
    runtime.CFG['run'] = []
    runtime.CFG['a_pos'] = 0.9
    a_pos, a_neg = getattr(runtime.args, 'a_pos', None), getattr(runtime.args, 'a_neg', None)
    if a_pos is not None:
        assert 0 < a_pos <= 1, 'a_pos phải trong (0, 1]'
        runtime.CFG['a_pos'] = a_pos
    if a_neg is not None:
        assert 0 < a_neg <= 1, 'a_neg phải trong (0, 1]'
        runtime.CFG['a_tcd'] = a_neg
    print('a_pos =', runtime.CFG['a_pos'], '| a_neg (tcd) =', runtime.CFG['a_tcd'])
