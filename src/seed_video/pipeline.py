"""Ordered stage orchestration and source fingerprinting."""
import hashlib
from importlib import import_module
from pathlib import Path
from .runtime import Runtime

SETUP = (
    "provenance", "config", "data.videohallucer", "data.eventhallusion",
    "data.video", "models.foreground", "models.qwen", "teachers.views",
    "teachers.composition", "training.losses", "training.trainer",
    "experiments.components",
)
EXTRAS = {
    "core": (),
    "no-caption": ("experiments.answer_tokens", "experiments.keep_repair"),
    "full": ("experiments.caption", "experiments.factorial"),
    "all": ("experiments.seednoise", "experiments.answer_tokens", "experiments.caption",
            "experiments.keep_repair", "experiments.factorial"),
}


def plan(stage):
    return SETUP + EXTRAS[stage]


def source_hash():
    digest = hashlib.sha256()
    base = Path(__file__).parent
    for path in sorted(base.rglob("*.py")):
        digest.update(str(path.relative_to(base)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(args):
    runtime = Runtime(args)
    for name in plan(args.stage):
        print(f"[stage] {name}", flush=True)
        module_name = f"seed_video.{name}"
        if name == "models.qwen" and runtime.CFG.get("model_id", "").startswith("OpenGVLab/"):
            module_name = "seed_video.models.internvl"
        import_module(module_name).run(runtime)
        if name == "provenance":
            runtime.SOURCE_HASH = source_hash()
    return runtime
