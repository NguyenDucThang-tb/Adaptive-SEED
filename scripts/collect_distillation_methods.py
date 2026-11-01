"""Print the base and per-method metrics from a distillation comparison run."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--a-pos", type=float, default=0.9)
    parser.add_argument("--a-neg", type=float, default=0.25)
    parser.add_argument("--teacher-frames", type=int, default=32)
    args = parser.parse_args()

    tag = f"ap{args.a_pos:g}_an{args.a_neg:g}_tf{args.teacher_frames}"
    path = args.work / "cdt_distill_cache_f8_448" / "distillation_methods" / "runs" / tag / "results.csv"
    if not path.exists():
        raise SystemExit(f"Missing results: {path}")

    rows = pd.read_csv(path)
    print(f"Results: {path}")
    print(f"{'Run':34} {'Split':12} {'N':>5} {'Q acc':>8} {'Pair acc':>10} {'W2C':>5} {'C2W':>5}")
    print("-" * 88)
    for _, row in rows.iterrows():
        pair = row.get("pair_acc")
        pair_text = "N/A" if pd.isna(pair) else f"{pair * 100:.2f}%"
        print(
            f"{row['run']:34} {row['split']:12} {int(row['n']):5d} "
            f"{row['q_acc'] * 100:7.2f}% {pair_text:>10} "
            f"{int(row['wrong_to_correct']):5d} {int(row['correct_to_wrong']):5d}"
        )


if __name__ == "__main__":
    main()
