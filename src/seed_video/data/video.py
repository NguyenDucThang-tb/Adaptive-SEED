"""data: video stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType

def view_frames(runtime, it, view, nf=None):
    fr, dur = runtime._decode(it['video'], nf or runtime.CFG['n_frames'], runtime.CFG['frame_size'])
    fr = list(fr)
    n = len(fr)
    fidx = list(range(n))
    if view == 'tcd':
        fidx = runtime.np.linspace(0, n - 1, runtime.CFG['tcd_frames']).round().astype(int).tolist()
        return ([fr[i] for i in fidx], len(fidx) / dur, fidx)
    if view == 'shuf':
        perm = list(range(n // 2))
        r = runtime.random.Random(runtime.seed_of(it['video']))
        while perm == sorted(perm):
            r.shuffle(perm)
        fidx = [2 * p + j for p in perm for j in (0, 1)]
        return ([fr[i] for i in fidx], n / dur, fidx)
    if view == 'seednoise':
        g = runtime.torch.Generator().manual_seed(runtime.seed_of(it['video']))
        x = runtime.torch.from_numpy(runtime.np.stack([runtime.np.asarray(f) for f in fr])).float() / 127.5 - 1
        a = runtime.CFG['seed_noise_alpha']
        x = runtime.math.sqrt(a) * x + runtime.math.sqrt(1 - a) * runtime.torch.randn(x.shape, generator=g)
        x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
        return ([runtime.Image.fromarray(v) for v in x], n / dur, fidx)
    assert view == 'orig', view
    return (fr, n / dur, fidx)

def run(runtime):
    from decord import VideoReader, cpu
    runtime.VideoReader = VideoReader
    runtime.cpu = cpu

    @runtime.lru_cache(maxsize=16)
    def _decode(path, n, size):
        vr = runtime.VideoReader(path, ctx=runtime.cpu(0))
        N, fps = (len(vr), vr.get_avg_fps() or 30.0)
        idx = runtime.np.linspace(0, N - 1, n).round().astype(int)
        arr = vr.get_batch(idx).asnumpy()
        frames = tuple((runtime.Image.fromarray(a).convert('RGB').resize((size, size), runtime.Image.BICUBIC) for a in arr))
        return (frames, max(N / fps, 0.001))
    runtime._decode = _decode
    runtime.view_frames = MethodType(view_frames, runtime)
