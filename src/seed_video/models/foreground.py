"""models: foreground stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType

def mask_path(runtime, v):
    return f'{runtime.CACHE}/fg_{runtime.MASK_TAG}/{runtime.hashlib.md5(v.encode()).hexdigest()}.pt'

def fg_soft_grid(runtime, preds, grid, kernel, avg_weight):
    """preds [N,H,W]: xác suất BiRefNet theo frame (N = 2T hoặc T). Trả g [T,Ht,Wt] ∈ [0,1], đúng các bước của fg_birefnet."""
    T, Ht, Wt = grid
    N = preds.shape[0]
    out = []
    for t in range(T):
        if N == 2 * T:
            pred = runtime.torch.maximum(preds[min(2 * t, N - 1)], preds[min(2 * t + 1, N - 1)])
        elif N == T:
            pred = preds[t]
        else:
            pred = preds[min(round(t * (N - 1) / max(T - 1, 1)), N - 1)]
        if kernel is not None and kernel > 1:
            pred = runtime.torch.from_numpy(runtime.cv2.morphologyEx(pred.numpy(), runtime.cv2.MORPH_CLOSE, runtime.np.ones((kernel, kernel), dtype=runtime.np.uint8))).float()
        g_avg = runtime.F.adaptive_avg_pool2d(pred[None, None], (Ht, Wt))[0, 0]
        g_max = runtime.F.adaptive_max_pool2d(pred[None, None], (Ht, Wt))[0, 0]
        out.append((avg_weight * g_avg + (1 - avg_weight) * g_max).clamp(0, 1))
    return runtime.torch.stack(out)

def fg_tokens(runtime, video, T, hm, wm):
    g = runtime.FG_SOFT[video].float()
    assert tuple(g.shape) == (T, hm, wm), f'lưới fg {tuple(g.shape)} ≠ lưới token {(T, hm, wm)}'
    return g if runtime.CFG['fg_soft'] else (g > runtime.CFG['fg_thr']).float()

def run(runtime):
    import cv2
    runtime.cv2 = cv2
    from transformers import AutoModelForImageSegmentation
    runtime.AutoModelForImageSegmentation = AutoModelForImageSegmentation
    from torchvision import transforms
    runtime.transforms = transforms
    runtime.FG_GRID = (runtime.CFG['teacher_frames'] // 2, runtime.CFG['frame_size'] // 28, runtime.CFG['frame_size'] // 28)
    runtime.MASK_CFG = dict(algo='fg_birefnet_v1', birefnet=runtime.CFG['birefnet_id'], bir_res=1024, n_frames=runtime.CFG['teacher_frames'], frame_size=runtime.CFG['frame_size'], grid=runtime.FG_GRID, kernel=runtime.CFG['fg_kernel'], avg_weight=runtime.CFG['fg_avg_weight'])
    runtime.MASK_TAG = runtime.cfg_hash(runtime.MASK_CFG)
    runtime.os.makedirs(f'{runtime.CACHE}/fg_{runtime.MASK_TAG}', exist_ok=True)
    runtime.mask_path = MethodType(mask_path, runtime)
    runtime.fg_soft_grid = MethodType(fg_soft_grid, runtime)
    runtime.todo_m = [v for v in runtime.ALL_VIDEOS if not runtime.os.path.exists(runtime.mask_path(v))]
    if runtime.todo_m:
        runtime.birefnet = runtime.AutoModelForImageSegmentation.from_pretrained(runtime.CFG['birefnet_id'], trust_remote_code=True).to(runtime.DEV).eval().half()
        runtime.tf_bir = runtime.transforms.Compose([runtime.transforms.Resize((1024, 1024)), runtime.transforms.ToTensor(), runtime.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        runtime.t0 = runtime.time.time()
        with runtime.torch.inference_mode():
            for runtime.j, runtime.v in enumerate(runtime.todo_m):
                runtime.fr = list(runtime._decode(runtime.v, runtime.CFG['teacher_frames'], runtime.CFG['frame_size'])[0])
                runtime.preds = []
                for runtime.k in range(0, len(runtime.fr), 4):
                    runtime.p = runtime.birefnet(runtime.torch.stack([runtime.tf_bir(f) for f in runtime.fr[runtime.k:runtime.k + 4]]).to(runtime.DEV).half())
                    runtime.p = runtime.p[-1] if isinstance(runtime.p, (list, tuple)) else runtime.p
                    runtime.preds.append(runtime.p.sigmoid().float()[:, 0].cpu())
                runtime.g = runtime.fg_soft_grid(runtime.torch.cat(runtime.preds), runtime.FG_GRID, runtime.CFG['fg_kernel'], runtime.CFG['fg_avg_weight'])
                runtime.torch.save(runtime.g.half(), runtime.mask_path(runtime.v))
                if (runtime.j + 1) % 50 == 0:
                    print(f'  fg {runtime.j + 1}/{len(runtime.todo_m)} ({(runtime.time.time() - runtime.t0) / 60:.1f} phút)')
        del runtime.birefnet
        runtime.gc.collect()
        runtime.torch.cuda.empty_cache()
    runtime.FG_SOFT = {v: runtime.torch.load(runtime.mask_path(v), weights_only=False) for v in runtime.ALL_VIDEOS}
    runtime.fg_tokens = MethodType(fg_tokens, runtime)
    runtime.fr_tok = runtime.np.mean([(g.float() > runtime.CFG['fg_thr']).float().mean().item() for g in runtime.FG_SOFT.values()])
    print(f"fg cache: {runtime.MASK_TAG} | lưới {runtime.FG_GRID} | tỉ lệ token foreground (thr={runtime.CFG['fg_thr']}): {runtime.fr_tok:.3f}")
