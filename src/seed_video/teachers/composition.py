"""teachers: composition stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType
from seed_video.evaluation.metrics import pair_vec, pair_acc

def coefs(runtime, spec):
    c = {'orig': 1.0}
    if spec['pos'] and spec['a_pos']:
        c['orig'] -= spec['a_pos']
        c[spec['pos']] = spec['a_pos']
    for v, a in spec['negs'].items():
        c['orig'] += a
        c[v] = c.get(v, 0.0) - a
    return c

def formula(runtime, spec):
    return ' '.join((f"{('+' if w >= 0 else '−')} {abs(w):g}·l_{v}" for v, w in runtime.coefs(spec).items())).lstrip('+ ')

def compose(runtime, get, spec):
    lo = get('orig')
    lt = lo.clone()
    if spec['pos'] and spec['a_pos']:
        lt = lt + spec['a_pos'] * (get(spec['pos']) - lo)
    for v, a in spec['negs'].items():
        lt = lt - a * (get(v) - lo)
    return lt

def tlog(runtime, k, name):
    return runtime.compose(lambda v: runtime.L[v][k].float().to(runtime.DEV), runtime.TEACHERS[name])

def run(runtime):
    from scipy.stats import binomtest
    runtime.binomtest = binomtest
    runtime.A_P, runtime.A_T = (runtime.CFG['a_pos'], runtime.CFG['a_tcd'])
    runtime.TEACHERS = {'T_pos': dict(pos='orig+fg', a_pos=runtime.A_P, negs={}), 'T_tcd': dict(pos=None, a_pos=0.0, negs={'tcd': runtime.A_T}), 'T_pos_tcd': dict(pos='orig+fg', a_pos=runtime.A_P, negs={'tcd': runtime.A_T}), 'T_base': dict(pos=None, a_pos=0.0, negs={}), 'T_posrand_tcd': dict(pos='orig+rand', a_pos=runtime.A_P, negs={'tcd': runtime.A_T}), 'T_noise': dict(pos=None, a_pos=0.0, negs={'seednoise': runtime.CFG['a_noise']})}
    runtime.TEACHERS = {k: v for k, v in runtime.TEACHERS.items() if k in runtime.NEED_T}
    runtime.coefs = MethodType(coefs, runtime)
    runtime.formula = MethodType(formula, runtime)
    print('Teacher (dạng tường minh):')
    for runtime._n, runtime._s in runtime.TEACHERS.items():
        runtime._c = runtime.coefs(runtime._s)['orig']
        print(f'  {runtime._n:10s} = {runtime.formula(runtime._s)}')
        assert abs(runtime._c) > 1e-06, f"{runtime._n}: hệ số của l_orig = 0 → teacher không còn nhánh gốc; đổi CFG['a_pos'] / CFG['a_tcd']"
    runtime.compose = MethodType(compose, runtime)
    runtime.tlog = MethodType(tlog, runtime)
    runtime.MARG = {}
    runtime.pair_vec = MethodType(pair_vec, runtime)
    runtime.pair_acc = MethodType(pair_acc, runtime)
