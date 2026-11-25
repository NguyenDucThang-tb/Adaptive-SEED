#!/usr/bin/env python3
"""Experimental CounterVid 16/8-frame token-gain LoRA distillation."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-answer-tokens', type=int, default=128)
    parser.add_argument('--size', type=int, default=448)
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_answer_tokens < 1 or args.size < 1:
        parser.error('step, token, and size limits must be positive')

    import cv2
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from seed_video.data.countervid import load_freeform
    from seed_video.training.countervid_loss import adaptive_sequence_loss

    if not torch.cuda.is_available():
        raise RuntimeError('This training entry point requires a CUDA GPU')
    items = load_freeform(args.data_root, 'train')
    if not items:
        raise RuntimeError('No training examples')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.shuffle(items)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True).to('cuda')
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none'))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    def inputs(item, count, tokens):
        cap = cv2.VideoCapture(item['video_path'])
        try:
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if not cap.isOpened() or length < 1:
                raise RuntimeError(f"Cannot decode {item['video_path']}")
            frames = []
            for index in np.rint(np.linspace(0, length - 1, count)).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode frame {index}: {item['video_path']}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(frame, (args.size, args.size)))
        finally:
            cap.release()
        prompt = processor.apply_chat_template([{'role': 'user', 'content': [
            {'type': 'video'}, {'type': 'text', 'text': item['question']}]}],
            tokenize=False, add_generation_prompt=True)
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        batch = processor(text=[prompt], videos=[video], fps=count / max(length / fps, 0.001),
                          return_tensors='pt').to('cuda')
        prompt_len = batch['input_ids'].shape[1]
        batch['input_ids'] = torch.cat((batch['input_ids'], tokens), dim=1)
        batch['attention_mask'] = torch.cat((batch['attention_mask'], torch.ones_like(tokens)), dim=1)
        return batch, prompt_len

    for step in range(args.max_steps):
        item = items[step % len(items)]
        ids = processor.tokenizer(item['answer'], add_special_tokens=False,
                                  return_tensors='pt')['input_ids'][:, :args.max_answer_tokens].to('cuda')
        if ids.shape[1] == 0:
            raise RuntimeError(f"Empty answer tokens: {item['id']}")
        input8, p8 = inputs(item, 8, ids)
        input16, p16 = inputs(item, 16, ids)
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            teacher = model(**input16).logits[:, p16 - 1:p16 - 1 + ids.shape[1]].float()
            base = model(**input8).logits[:, p8 - 1:p8 - 1 + ids.shape[1]].float()
        model.train()
        student = model(**input8).logits[:, p8 - 1:p8 - 1 + ids.shape[1]]
        loss, info = adaptive_sequence_loss(student, teacher, base, ids, torch.ones_like(ids))
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        print(json.dumps({'step': step + 1, 'id': item['id'], 'loss': loss.item(),
                          **{k: v.item() for k, v in info.items()}}), flush=True)
    model.save_pretrained(output / 'adapter')
    processor.save_pretrained(output / 'processor')
    print(f'Completed {args.max_steps} optimizer steps; adapter: {output / "adapter"}', flush=True)


if __name__ == '__main__':
    main()
