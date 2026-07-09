"""Step 2: Sample 20 few-shot examples from train split (10 malignant + 10 benign).

Random seed = 42. Saves ./few_shot_samples.json with structure:
{
  "seed": 42,
  "all_20": [{"AnnotationID": ..., "label": 0|1, "PatientID": ...}, ...],
  "malignant_10": [...],
  "benign_10": [...],
  "five_shot": {"malignant_3": [...], "benign_2": [...]}
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_utils import ID_COL, LABEL_COL, PID_COL, PROJECT_ROOT, get_split_rows, load_label_df, load_split

DEFAULT_OUT = PROJECT_ROOT / "benchmark" / "few_shot_samples.json"


def sample_few_shot(seed: int = 42, n_per_class: int = 10) -> dict:
    df = load_label_df()
    split = load_split()
    train_rows = get_split_rows(df, split, "train")

    rng = np.random.default_rng(seed)

    def _pick(label_value: int, k: int):
        sub = train_rows[train_rows[LABEL_COL] == label_value]
        if len(sub) < k:
            raise ValueError(f"Not enough rows with label={label_value}: have {len(sub)}, need {k}")
        idx = rng.choice(len(sub), size=k, replace=False)
        sub = sub.iloc[sorted(idx.tolist())]
        return [
            {ID_COL: r[ID_COL], LABEL_COL: int(r[LABEL_COL]), PID_COL: int(r[PID_COL])}
            for _, r in sub.iterrows()
        ]

    malignant_10 = _pick(1, n_per_class)
    benign_10 = _pick(0, n_per_class)

    # 5-shot subset: 3 malignant + 2 benign (deterministic prefix)
    five_shot = {
        "malignant_3": malignant_10[:3],
        "benign_2": benign_10[:2],
    }

    all_20 = malignant_10 + benign_10
    return {
        "seed": seed,
        "all_20": all_20,
        "malignant_10": malignant_10,
        "benign_10": benign_10,
        "five_shot": five_shot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    data = sample_few_shot(seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {args.out}")
    print(f"  malignant: {[s[ID_COL] for s in data['malignant_10']]}")
    print(f"  benign:    {[s[ID_COL] for s in data['benign_10']]}")
    print(f"  5-shot:    {[s[ID_COL] for s in data['five_shot']['malignant_3'] + data['five_shot']['benign_2']]}")


if __name__ == "__main__":
    main()
