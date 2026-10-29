"""Command-line interface; --help and --dry-run work without ML dependencies."""
import argparse
from pathlib import Path
from .pipeline import plan, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="SEED video distillation experiments")
    parser.add_argument("--work-dir", type=Path, default=Path("runs/seed_full"))
    parser.add_argument("--stage", choices=["core", "no-caption", "full", "all"], default="all")
    parser.add_argument("--no-eh", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print stages without data downloads or GPU access")
    args = parser.parse_args(argv)
    if args.dry_run:
        print("\n".join(plan(args.stage)))
        return
    run(args)
