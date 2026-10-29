"""Reusable gates functions with explicit runtime dependencies."""

def norm_jsd(runtime, p, r):
    p = p.double().clamp_min(1e-12)
    r = r.double().clamp_min(1e-12)
    m = (p + r) / 2
    return float((0.5 * ((p * (p / m).log()).sum() + (r * (r / m).log()).sum()) / runtime.math.log(2)).clamp(0, 1))

def pair_gate(runtime, fwd, rev, pi):
    p = fwd[[runtime.YES, runtime.NO]].double().softmax(0)
    r = rev[[runtime.YES, runtime.NO]].double().softmax(0)
    assert sorted(pi) == [0, 1]
    d = runtime.norm_jsd(p, r)
    e = runtime.norm_jsd(p, r[pi])
    return (d * (1 - e), d, e)

def correct_margin(runtime, z, gt):
    other = runtime.NO if gt == runtime.YES else runtime.YES
    return float(z[gt] - z[other])
