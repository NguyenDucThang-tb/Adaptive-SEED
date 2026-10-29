"""models: qwen stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType
from seed_video.components.positive import feature_enhance, random_like_mask

def build_inputs(runtime, frames, question, fps, yn=True):
    msgs = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': question.strip() + ('\nAnswer with yes or no.' if yn else '')}]}]
    text = runtime.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    vid = runtime.torch.from_numpy(runtime.np.stack([runtime.np.asarray(f) for f in frames])).permute(0, 3, 1, 2)
    try:
        return runtime.processor(text=[text], videos=[vid], fps=fps, return_tensors='pt')
    except TypeError:
        return runtime.processor(text=[text], videos=[vid], return_tensors='pt')

def with_response(runtime, inp, resp):
    d = dict(inp)
    d['input_ids'] = runtime.torch.cat([inp['input_ids'], resp[None].to(inp['input_ids'].device)], 1)
    d['attention_mask'] = runtime.torch.cat([inp['attention_mask'], runtime.torch.ones_like(resp)[None].to(inp['attention_mask'].device)], 1)
    return d

def fwd_view(runtime, it, view, grad=False, resp=None, need_hs=False, question=None, nf=None):
    base, _, fe = view.partition('+')
    fr, fps, fidx = runtime.view_frames(it, base, nf)
    inp = runtime.build_inputs(fr, question or it['question'], fps, yn=question is None).to(runtime.DEV)
    if resp is not None:
        inp = runtime.with_response(inp, resp)
    T, gh, gw = inp['video_grid_thw'][0].tolist()
    if fe:
        assert base == 'orig', 'positive chỉ hỗ trợ trên view orig (fg cache theo frame gốc)'
        fg = runtime.fg_tokens(it['video'], T, gh // 2, gw // 2)
        if fe == 'rand':
            fg = runtime.random_like_mask(fg, runtime.seed_of(it['video']))
        runtime.hook.grid, runtime.hook.fg = ((T, gh, gw), fg)
    runtime.hook.active = bool(fe)
    keep = 1 if resp is None else resp.numel() + 1
    try:
        with runtime.torch.enable_grad() if grad else runtime.torch.no_grad(), runtime.keep_last(keep):
            out = runtime.model(**inp, output_hidden_states=need_hs)
    finally:
        runtime.hook.active = False
    return (out, inp)

def adapter_off(runtime):
    return runtime.model.disable_adapter() if hasattr(runtime.model, 'disable_adapter') else runtime.contextlib.nullcontext()

def margin(runtime, l):
    return (l[runtime.YES].float() - l[runtime.NO].float()).item()

def run(runtime):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    runtime.AutoProcessor = AutoProcessor
    runtime.Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration
    runtime.processor = runtime.AutoProcessor.from_pretrained(runtime.CFG['model_id'])
    runtime.model = runtime.Qwen2_5_VLForConditionalGeneration.from_pretrained(runtime.CFG['model_id'], torch_dtype=runtime.torch.bfloat16, attn_implementation='sdpa', device_map='cuda').eval()
    runtime.model.requires_grad_(False)

    class SliceHead(runtime.torch.nn.Module):

        def __init__(self, head):
            super().__init__()
            self.head = head
            self.keep = None

        def forward(self, h):
            return self.head(h if self.keep is None else h[:, -self.keep:])
    runtime.SliceHead = SliceHead
    if not isinstance(runtime.model.lm_head, runtime.SliceHead):
        runtime.model.lm_head = runtime.SliceHead(runtime.model.lm_head)
    runtime.HEAD = runtime.model.lm_head

    @runtime.contextlib.contextmanager
    def keep_last(k):
        old = runtime.HEAD.keep
        runtime.HEAD.keep = k
        try:
            yield
        finally:
            runtime.HEAD.keep = old
    runtime.keep_last = keep_last
    runtime.tok = runtime.processor.tokenizer
    runtime.YES = runtime.tok.encode('Yes', add_special_tokens=False)[0]
    runtime.NO = runtime.tok.encode('No', add_special_tokens=False)[0]
    runtime.VID_TOK = getattr(runtime.model.config, 'video_token_id', None) or runtime.tok.convert_tokens_to_ids('<|video_pad|>')
    runtime.VISUAL = runtime.model.visual if hasattr(runtime.model, 'visual') else runtime.model.model.visual
    runtime.random_like_mask = MethodType(random_like_mask, runtime)
    runtime.feature_enhance = MethodType(feature_enhance, runtime)

    class FEHook:

        def __init__(self):
            self.active, self.grid, self.fg = (False, None, None)
            self.kw = dict(alpha=runtime.CFG['alpha'], alpha_s=runtime.CFG['alpha_s'], beta=runtime.CFG['beta'])

        def __call__(self, module, args, out):
            if not self.active:
                return None
            X = out[0] if isinstance(out, (tuple, list)) else out
            T, gh, gw = self.grid
            hm, wm = (gh // 2, gw // 2)
            assert X.shape[0] == T * hm * wm, (tuple(X.shape), self.grid)
            V2 = runtime.feature_enhance(X.float().view(T, hm, wm, -1), self.fg.to(X.device), **self.kw)
            Y = V2.to(X.dtype).view_as(X)
            return (Y,) + tuple(out[1:]) if isinstance(out, tuple) else Y
    runtime.FEHook = FEHook
    runtime.hook = runtime.FEHook()
    if '_h' in vars(runtime):
        runtime._h.remove()
    runtime._h = runtime.VISUAL.register_forward_hook(runtime.hook)
    runtime.build_inputs = MethodType(build_inputs, runtime)
    runtime.with_response = MethodType(with_response, runtime)
    runtime.fwd_view = MethodType(fwd_view, runtime)
    runtime.adapter_off = MethodType(adapter_off, runtime)
    runtime.margin = MethodType(margin, runtime)
    runtime.it0 = runtime.TEST[0]
    for runtime.v in ['orig', 'tcd', 'shuf']:
        runtime._, runtime.inp = runtime.fwd_view(runtime.it0, runtime.v, nf=runtime.CFG['teacher_frames'])
        print(f"{runtime.v:5s} grid={runtime.inp['video_grid_thw'][0].tolist()} second_per_grid_ts={runtime.inp.get('second_per_grid_ts')}")
    runtime.saved = dict(runtime.hook.kw)
    runtime.hook.kw = dict(alpha=0.0, alpha_s=0.0, beta=0.0)
    runtime.d = (runtime.fwd_view(runtime.it0, 'orig', nf=runtime.CFG['teacher_frames'])[0].logits[0, -1].float() - runtime.fwd_view(runtime.it0, 'orig+fg', nf=runtime.CFG['teacher_frames'])[0].logits[0, -1].float()).abs().max().item()
    runtime.hook.kw = runtime.saved
    print(f'null positive vs orig: max|Δ| = {runtime.d:.2e}')
    assert runtime.d == 0.0, 'harness không sạch'
