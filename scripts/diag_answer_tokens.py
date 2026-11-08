"""Show which tokens the model actually predicts at the yes/no answer position.

Runs the normal setup stages (config, data, foreground, model) on a smoke-sized
EventHallusion split, then prints, for each item, the probability of the
"Yes"/"No" tokens used by the margin and CE loss, the lowercase variants, and
the top-5 next tokens. Lives outside src/ so it does not change SOURCE_HASH.

    PYTHONPATH=src python scripts/diag_answer_tokens.py --work-dir $SMOKE --videos 4
"""
import argparse
from argparse import Namespace
from importlib import import_module
from pathlib import Path

from seed_video.pipeline import SETUP, source_hash
from seed_video.runtime import Runtime

STAGES = SETUP[:SETUP.index("training.losses") + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=4, help="Train videos in the smoke split")
    parser.add_argument("--teacher-frames", type=int, default=32)
    cli = parser.parse_args()

    args = Namespace(
        work_dir=cli.work_dir, stage="core", no_eh=False, train_dataset="eventhallusion",
        eh_train_fraction=0.7, eh_val_fraction=0.1, eh_split_seed=42,
        a_pos=None, a_neg=None, distill_methods=None, base_only=False,
        teacher_frames=cli.teacher_frames, smoke_pairs=cli.videos, arms=None, dry_run=False,
    )
    runtime = Runtime(args)
    for name in STAGES:
        print(f"[stage] {name}", flush=True)
        module_name = f"seed_video.{name}"
        if name == "models.qwen" and runtime.CFG.get("model_id", "").startswith("OpenGVLab/"):
            module_name = "seed_video.models.internvl"
        import_module(module_name).run(runtime)
        if name == "provenance":
            runtime.SOURCE_HASH = source_hash()

    torch, tok = runtime.torch, runtime.tok
    ids = {w: tok.encode(w, add_special_tokens=False) for w in ("Yes", "No", "yes", "no")}
    print("model:", runtime.CFG["model_id"])
    print("token ids:", ids, "| runtime.YES/NO =", runtime.YES, runtime.NO)

    items = runtime.TRAIN + runtime.VAL + runtime.TEST
    runtime.model.eval()
    for it in items:
        with torch.no_grad():
            logits = runtime.student_last(it, grad=False).float()
        probs = logits.softmax(-1)
        p = {w: float(probs[v[0]]) for w, v in ids.items()}
        top = probs.topk(5)
        top_txt = ", ".join(f"{tok.decode([int(i)])!r}:{float(q):.3f}" for q, i in zip(top.values, top.indices))
        print(
            f"{it['subset']:<14} gt={it['answer']:<3} margin={runtime.margin(logits):+.2f} "
            + " ".join(f"P({w})={v:.4f}" for w, v in p.items())
            + f" | top5: {top_txt}",
            flush=True,
        )


if __name__ == "__main__":
    main()
