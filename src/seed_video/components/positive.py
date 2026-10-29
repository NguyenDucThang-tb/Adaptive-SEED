"""Reusable positive functions with explicit runtime dependencies."""

def random_like_mask(runtime, fg, seed):
    T, h, w = fg.shape
    perm = runtime.torch.randperm(h * w, generator=runtime.torch.Generator().manual_seed(seed))
    return fg.reshape(T, -1)[:, perm].reshape(T, h, w)

def feature_enhance(runtime, V, fg, alpha, alpha_s, beta):
    persist = fg.mean(0, keepdim=True).expand_as(fg)
    Vs = V * (1 + alpha * fg[..., None] + alpha_s * (persist * fg)[..., None])
    gate = runtime.torch.zeros_like(fg)
    gate[1:] = runtime.torch.maximum(fg[1:], fg[:-1])
    diff = runtime.torch.zeros_like(V)
    diff[1:] = V[1:] - V[:-1]
    e = diff * gate[..., None]
    e_hat = e / (e.norm(dim=-1, keepdim=True) + 1e-06) * V.norm(dim=-1, keepdim=True)
    return Vs + beta * gate[..., None] * e_hat
