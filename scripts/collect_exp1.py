"""Gộp kết quả Thực nghiệm 1a thành bảng: Arm | VH pair acc | VH Q acc | EventHallusion."""
import argparse
import os
from pathlib import Path

import pandas as pd

ARMS = {
    "Only Pos": ("adaptive_pos_only", lambda v, ap, an: (v, an)),
    "Only Neg": ("adaptive_neg_only", lambda v, ap, an: (ap, v)),
    "Both": ("adaptive_pos_neg", lambda v, ap, an: (v, v)),
}


def tag(a_pos, a_neg, teacher_frames):
    return f"ap{a_pos:g}_an{a_neg:g}_tf{teacher_frames}"


def summarize(frame, run, vh_split):
    """Một dòng: (VH pair acc, VH Q acc, EH acc) và số câu đổi đúng/sai trên tập đánh giá."""
    rows = frame[frame["run"] == run].set_index("split")
    if rows.empty:
        return None

    def get(split, col):
        return rows.loc[split, col] if split in rows.index else float("nan")

    return {
        "VH pair acc": get(vh_split, "pair_acc"),
        "VH Q acc": get(vh_split, "q_acc"),
        "EH acc": get("EH_ALL", "q_acc"),
        "VH w2c": get(vh_split, "wrong_to_correct"),
        "VH c2w": get(vh_split, "correct_to_wrong"),
        "EH w2c": get("EH_ALL", "wrong_to_correct"),
        "EH c2w": get("EH_ALL", "correct_to_wrong"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path(f"/scratch/hn98/{os.environ.get('USER', '')}/runs/exp1"))
    parser.add_argument("--values", nargs="+", type=float, default=[0.25, 0.5, 0.75, 0.9, 1.0])
    parser.add_argument(
        "--both-pairs",
        nargs="+",
        default=None,
        metavar="A_POS:A_NEG",
        help="Report explicit 2D Both-grid pairs instead of the legacy diagonal.",
    )
    parser.add_argument("--fixed-ap", type=float, default=0.9)
    parser.add_argument("--fixed-an", type=float, default=0.25)
    parser.add_argument("--teacher-frames", type=int, default=32)
    parser.add_argument("--vh-split", default="VH_TEST", choices=["VH_TEST", "VH_VAL"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.work / "cdt_distill_cache_f8_448" / "seed_component_ablation" / "runs"
    out_rows = []
    for label, (arm, coeffs) in ARMS.items():
        if label == "Both" and args.both_pairs is not None:
            points = []
            for pair in args.both_pairs:
                try:
                    ap_text, an_text = pair.split(":", maxsplit=1)
                    ap, an = float(ap_text), float(an_text)
                except (ValueError, TypeError):
                    parser.error(f"Invalid --both-pairs value {pair!r}; expected A_POS:A_NEG")
                if not (0 <= ap <= 1 and 0 <= an <= 1):
                    parser.error(f"Both-grid values must be in [0, 1]: {pair!r}")
                points.append((None, ap, an))
        else:
            points = [(v, *coeffs(v, args.fixed_ap, args.fixed_an)) for v in args.values]

        for v, ap, an in points:
            path = root / tag(ap, an, args.teacher_frames) / "results.csv"
            if not path.exists():
                out_rows.append({"Arm": label, "max a": v, "a_pos": ap, "a_neg": an, "Method": "(missing results)"})
                continue
            frame = pd.read_csv(path)
            for method, run in (("Base_fix", "seed_base_s0"), ("Adaptive", f"{arm}_s0")):
                summary = summarize(frame, run, args.vh_split)
                if summary is None:
                    out_rows.append({"Arm": label, "max a": v, "a_pos": ap, "a_neg": an, "Method": f"{method} (missing)"})
                else:
                    out_rows.append({"Arm": label, "max a": v, "a_pos": ap, "a_neg": an, "Method": method, **summary})

    table = pd.DataFrame(out_rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(table.round(4).to_string(index=False))
    out = args.out or args.work / "exp1_table.csv"
    table.to_csv(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()
