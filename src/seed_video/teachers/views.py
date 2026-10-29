"""teachers: views stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType

def branch_cfg(runtime, b):
    base, _, fe = b.partition('+')
    d = dict(model=runtime.CFG['model_id'], n_frames=runtime.CFG['n_frames'], frame_size=runtime.CFG['frame_size'], prompt=runtime.PROMPT_TAG, view=base)
    if base == 'tcd':
        d['tcd_frames'] = runtime.CFG['tcd_frames']
    if base == 'shuf':
        d['shuf'] = 'pair_perm_v1'
    if base == 'seednoise':
        d['noise_alpha'] = runtime.CFG['seed_noise_alpha']
    if fe:
        d.update(fe=fe, alpha=runtime.CFG['alpha'], alpha_s=runtime.CFG['alpha_s'], beta=runtime.CFG['beta'], fg=runtime.MASK_TAG, fg_thr=runtime.CFG['fg_thr'], fg_soft=runtime.CFG['fg_soft'], rand='cell_perm_v1' if fe == 'rand' else None)
    return d

def bpath(runtime, b):
    return f"{runtime.CACHE}/branch_{b.replace('+', '_')}_{runtime.cfg_hash(runtime.branch_cfg(b))}_1.pt"

def _load_branch(runtime, b):
    if runtime.os.path.exists(runtime.bpath(b)):
        return runtime.torch.load(runtime.bpath(b), weights_only=False)
    legacy = f"{runtime.CACHE}/branch_{b.replace('+', '_')}.pt"
    if '+' not in b and runtime.LEGACY_OK and runtime.os.path.exists(legacy):
        print(f'  {b}: dùng lại cache cũ {runtime.os.path.basename(legacy)}')
        return runtime.torch.load(legacy, weights_only=False)
    return {}

def run(runtime):
    runtime.TEACHER_BRANCHES = {'T_pos': ['orig', 'orig+fg'], 'T_tcd': ['orig', 'tcd'], 'T_pos_tcd': ['orig', 'orig+fg', 'tcd'], 'T_base': ['orig'], 'T_posrand_tcd': ['orig', 'orig+rand', 'tcd'], 'T_noise': ['orig', 'seednoise']}
    runtime.NEED_T = list(runtime.CFG['teachers']) + [t for t in runtime.CFG['control_teachers'] if t not in runtime.CFG['teachers']]
    runtime.BRANCHES = list(dict.fromkeys((b for t in runtime.NEED_T for b in runtime.TEACHER_BRANCHES[t])))
    print('teacher:', runtime.NEED_T, '| nhánh cần cache:', runtime.BRANCHES)
    runtime.PROMPT_TAG = 'yn_suffix:Answer with yes or no.|fps'
    runtime.branch_cfg = MethodType(branch_cfg, runtime)
    runtime.bpath = MethodType(bpath, runtime)
    runtime.LEGACY_OK = runtime.CFG['model_id'] == 'Qwen/Qwen2.5-VL-7B-Instruct' and runtime.CFG['tcd_frames'] == 4 and (runtime.CFG['seed_noise_alpha'] == 0.3)
    runtime._load_branch = MethodType(_load_branch, runtime)
