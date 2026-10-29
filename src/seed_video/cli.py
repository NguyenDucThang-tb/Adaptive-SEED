"""Command-line interface; --help and --dry-run work without ML dependencies."""
import argparse
from pathlib import Path
from .pipeline import plan, run

ABLATION_ARMS = ["seed_base", "token_only", "pair_only", "adaptive_pos_only",
                 "adaptive_neg_only", "token_pair", "adaptive_pos_neg", "all_components"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="SEED video distillation experiments")
    parser.add_argument("--work-dir", type=Path, default=Path("runs/seed_full"))
    parser.add_argument("--stage", choices=["core", "no-caption", "full", "all"], default="all")
    parser.add_argument("--no-eh", action="store_true")
    parser.add_argument("--a-pos", type=float, default=None, help="Max positive coefficient a_p in (0, 1]; default 0.9")
    parser.add_argument("--a-neg", type=float, default=None, help="Max negative (TCD) coefficient a_n in (0, 1]; default 0.25")
    parser.add_argument("--arms", nargs="+", choices=ABLATION_ARMS, default=None,
                        help="Only train these ablation arms (stage components); default all 8")
    parser.add_argument("--dry-run", action="store_true", help="Print stages without data downloads or GPU access")
    args = parser.parse_args(argv)
    if args.dry_run:
        print("\n".join(plan(args.stage)))
        return
    run(args)
