"""InternVL3.5 backend for controlled, fixed-resolution video frames."""
import copy
from types import MethodType

from seed_video.components.positive import feature_enhance, random_like_mask


MODEL_ID = "OpenGVLab/InternVL3_5-8B-Instruct"


def model_spec(model_id):
    if model_id != MODEL_ID:
        raise ValueError(f"Unsupported InternVL model: {model_id}")
    return ("InternVLChatModel", 28)


def build_inputs(runtime, frames, question, fps, yn=True):
    del fps
    prompt_question = question.strip()
    if yn:
        prompt_question += "\nAnswer with yes or no."
    frame_prompt = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(frames)))
    template = copy.deepcopy(runtime.model.conv_template)
    template.system_message = runtime.model.system_message
    template.append_message(template.roles[0], frame_prompt + prompt_question)
    template.append_message(template.roles[1], None)
    prompt = template.get_prompt()

    image_context = "<img>" + "<IMG_CONTEXT>" * runtime.model.num_image_token + "</img>"
    for _ in frames:
        prompt = prompt.replace("<image>", image_context, 1)

    encoded = runtime.tok(prompt, return_tensors="pt")
    pixel_values = runtime.torch.stack([runtime.image_transform(frame) for frame in frames])
    runtime.model.img_context_token_id = runtime.tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    return runtime.BatchFeature(data={
        "pixel_values": pixel_values.to(dtype=runtime.torch.bfloat16),
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "image_flags": runtime.torch.ones((len(frames), 1), dtype=runtime.torch.long),
    })


def with_response(runtime, inp, resp):
    out = dict(inp)
    out["input_ids"] = runtime.torch.cat(
        [inp["input_ids"], resp[None].to(inp["input_ids"].device)], dim=1
    )
    out["attention_mask"] = runtime.torch.cat(
        [inp["attention_mask"], runtime.torch.ones_like(resp)[None].to(inp["attention_mask"].device)],
        dim=1,
    )
    return runtime.BatchFeature(data=out)


def get_feature_grid(runtime, inp):
    frame_count = int(inp["image_flags"].shape[0])
    token_grid = runtime.CFG["frame_size"] // runtime.CFG["spatial_stride"]
    return (frame_count, token_grid * 2, token_grid * 2)


def fwd_view(runtime, it, view, grad=False, resp=None, need_hs=False, question=None, nf=None):
    base, _, fe = view.partition("+")
    frames, fps, _ = runtime.view_frames(it, base, nf)
    inp = runtime.build_inputs(frames, question or it["question"], fps, yn=question is None).to(runtime.DEV)
    if resp is not None:
        inp = runtime.with_response(inp, resp)
    T, gh, gw = runtime.get_feature_grid(inp)
    if fe:
        assert base == "orig", "positive chỉ hỗ trợ trên view orig (foreground cache theo frame gốc)"
        fg = runtime.fg_tokens(it["video"], T, gh // 2, gw // 2)
        if fe == "rand":
            fg = runtime.random_like_mask(fg, runtime.seed_of(it["video"]))
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
    return runtime.model.disable_adapter() if hasattr(runtime.model, "disable_adapter") else runtime.contextlib.nullcontext()


def margin(runtime, logits):
    return (logits[runtime.YES].float() - logits[runtime.NO].float()).item()


def run(runtime):
    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from transformers import AutoModel, AutoTokenizer, BatchFeature

    model_spec(runtime.CFG["model_id"])
    runtime.BatchFeature = BatchFeature
    runtime.image_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
        transforms.Resize((runtime.CFG["frame_size"], runtime.CFG["frame_size"]), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    runtime.tok = AutoTokenizer.from_pretrained(runtime.CFG["model_id"], trust_remote_code=True, use_fast=False)
    runtime.model = AutoModel.from_pretrained(
        runtime.CFG["model_id"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()
    runtime.model.requires_grad_(False)

    class SliceHead(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head
            self.keep = None

        def forward(self, hidden):
            return self.head(hidden if self.keep is None else hidden[:, -self.keep:])

    runtime.SliceHead = SliceHead
    language_model = runtime.model.language_model
    if not isinstance(language_model.lm_head, SliceHead):
        language_model.lm_head = SliceHead(language_model.lm_head)
    runtime.HEAD = language_model.lm_head

    @runtime.contextlib.contextmanager
    def keep_last(count):
        previous = runtime.HEAD.keep
        runtime.HEAD.keep = count
        try:
            yield
        finally:
            runtime.HEAD.keep = previous

    runtime.keep_last = keep_last
    runtime.YES = runtime.tok.encode("Yes", add_special_tokens=False)[0]
    runtime.NO = runtime.tok.encode("No", add_special_tokens=False)[0]
    runtime.VISUAL = runtime.model.mlp1
    runtime.TRANSPOSE_FOREGROUND = runtime.model.ps_version == "v1"

    class FeatureHook:
        def __init__(self):
            self.active, self.grid, self.fg = False, None, None
            self.kw = dict(alpha=runtime.CFG["alpha"], alpha_s=runtime.CFG["alpha_s"], beta=runtime.CFG["beta"])

        def __call__(self, module, args, output):
            if not self.active:
                return None
            T, gh, gw = self.grid
            hm, wm = gh // 2, gw // 2
            if output.ndim != 3 or output.shape[:2] != (T, hm * wm):
                raise RuntimeError(f"Unexpected InternVL mlp1 output {tuple(output.shape)} for grid {self.grid}")
            visual = output.float().reshape(T, hm, wm, -1)
            foreground = self.fg.to(output.device)
            if runtime.TRANSPOSE_FOREGROUND:
                foreground = foreground.transpose(1, 2)
            enhanced = runtime.feature_enhance(visual, foreground, **self.kw)
            return enhanced.to(output.dtype).reshape_as(output)

    runtime.FeatureHook = FeatureHook
    runtime.hook = FeatureHook()
    if hasattr(runtime, "_h"):
        runtime._h.remove()
    runtime._h = runtime.model.mlp1.register_forward_hook(runtime.hook)
    runtime.random_like_mask = MethodType(random_like_mask, runtime)
    runtime.feature_enhance = MethodType(feature_enhance, runtime)
    runtime.build_inputs = MethodType(build_inputs, runtime)
    runtime.with_response = MethodType(with_response, runtime)
    runtime.get_feature_grid = MethodType(get_feature_grid, runtime)
    runtime.fwd_view = MethodType(fwd_view, runtime)
    runtime.adapter_off = MethodType(adapter_off, runtime)
    runtime.margin = MethodType(margin, runtime)

    runtime.it0 = runtime.TEST[0]
    for runtime.view in ("orig", "tcd", "shuf"):
        runtime.out, runtime.inp = runtime.fwd_view(runtime.it0, runtime.view, nf=runtime.CFG["teacher_frames"])
        print(f"{runtime.view:5s} InternVL frames={runtime.inp['image_flags'].shape[0]} pixel_values={tuple(runtime.inp['pixel_values'].shape)}")
    runtime.saved = dict(runtime.hook.kw)
    runtime.hook.kw = dict(alpha=0.0, alpha_s=0.0, beta=0.0)
    original = runtime.fwd_view(runtime.it0, "orig", nf=runtime.CFG["teacher_frames"])[0].logits[0, -1].float()
    no_op = runtime.fwd_view(runtime.it0, "orig+fg", nf=runtime.CFG["teacher_frames"])[0].logits[0, -1].float()
    runtime.hook.kw = runtime.saved
    runtime.delta = (original - no_op).abs().max().item()
    print(f"null positive vs orig: max|delta| = {runtime.delta:.2e}")
    assert runtime.delta == 0.0, "positive hook is not a no-op at zero coefficients"
