#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from benchmark.paths import VLM_RESULTS_DIR


BENCH = Path(__file__).parent
RESULTS_DIR = VLM_RESULTS_DIR
RNG_SEED = 42
N_BOOT = 1000

MAIN_FILES = [
    "luna25_model_comparison/f3_gemini25flash.jsonl",
    "luna25_model_comparison/f3_gemini25pro.jsonl",
    "luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl",
    "luna25_model_comparison/f3_gemini31pro.jsonl",
    "luna25_model_comparison/f3_gemma4_31b.jsonl",
    "luna25_model_comparison/f3_gemma4_26b_a4b.jsonl",
    "luna25_model_comparison/f3_medgemma15_4b.jsonl",
]

CURATED_LUNA25_DIRS = [
    "luna25_ablation",
    "luna25_replicates",
    "luna25_sensitivity",
    "luna25_model_comparison",
]


def should_include(path: Path) -> bool:
    if path.suffix != ".jsonl":
        return False
    try:
        top_level = path.relative_to(RESULTS_DIR).parts[0]
    except ValueError:
        return False
    return top_level in CURATED_LUNA25_DIRS


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        record["_line"] = line_no
        rows.append(record)
    return pd.DataFrame(rows)


def normalize_model_name(raw: Any, stem: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return stem
    lower = text.lower()
    if "medgemma-1.5-4b-it" in lower:
        return "google/medgemma-1.5-4b-it"
    return text


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    auc = float(roc_auc_score(y_true, y_score))
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    aucs = np.empty(n_boot, dtype=float)
    rng = np.random.default_rng(RNG_SEED)
    for i in range(n_boot):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        q = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, q])
        aucs[i] = roc_auc_score(y_true[idx], y_score[idx])
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return auc, float(lo), float(hi)


def _cm(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tn, fp, fn, tp


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_total": int(len(df)),
        "n_valid": 0,
        "n_invalid": 0,
        "n_duplicates": 0,
        "auc": np.nan,
        "ci_lo": np.nan,
        "ci_hi": np.nan,
        "sens_at_0p5": np.nan,
        "spec_at_0p5": np.nan,
        "opt_thresh": np.nan,
        "sens_at_opt": np.nan,
        "spec_at_opt": np.nan,
    }
    if df.empty:
        return out

    work = df.copy()
    work["confidence"] = pd.to_numeric(work["confidence"], errors="coerce")
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    valid = work["confidence"].between(0.0, 1.0, inclusive="both") & work["label"].notna()
    out["n_invalid"] = int((~valid).sum())
    work = work[valid].copy()
    before = len(work)
    if "aid" in work.columns:
        work = work.sort_values("_line").drop_duplicates("aid", keep="last")
    out["n_duplicates"] = int(before - len(work))
    out["n_valid"] = int(len(work))
    if work.empty:
        return out

    y_true = work["label"].astype(int).values
    y_score = work["confidence"].astype(float).values
    auc, ci_lo, ci_hi = bootstrap_ci(y_true, y_score)

    pred_05 = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = _cm(y_true, pred_05)
    sens_05 = tp / (tp + fn) if (tp + fn) else np.nan
    spec_05 = tn / (tn + fp) if (tn + fp) else np.nan

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    opt_thresh = float(thresholds[best_idx])
    pred_opt = (y_score >= opt_thresh).astype(int)
    tn2, fp2, fn2, tp2 = _cm(y_true, pred_opt)
    sens_opt = tp2 / (tp2 + fn2) if (tp2 + fn2) else np.nan
    spec_opt = tn2 / (tn2 + fp2) if (tn2 + fp2) else np.nan

    out.update(
        {
            "auc": auc,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "sens_at_0p5": float(sens_05),
            "spec_at_0p5": float(spec_05),
            "opt_thresh": opt_thresh,
            "sens_at_opt": float(sens_opt),
            "spec_at_opt": float(spec_opt),
        }
    )
    return out


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for directory in CURATED_LUNA25_DIRS:
        for path in sorted((RESULTS_DIR / directory).glob("*.jsonl")):
            if not should_include(path):
                continue
            df = load_jsonl(path)
            metrics = compute_metrics(df)
            rel_path = path.relative_to(RESULTS_DIR).as_posix()
            if not df.empty:
                first = df.iloc[0].to_dict()
                mode = first.get("mode")
                prompt = first.get("prompt")
                model = normalize_model_name(first.get("model"), path.stem)
                sample_fps = first.get("sample_fps")
                frame_anchor = first.get("frame_anchor")
            else:
                mode = prompt = sample_fps = frame_anchor = None
                model = path.stem
            rows.append(
                {
                    "file": rel_path,
                    "family": "medgemma" if "medgemma" in rel_path else "vlm_sync",
                    "model": model,
                    "mode": mode,
                    "prompt": prompt,
                    "sample_fps": sample_fps,
                    "frame_anchor": frame_anchor,
                    **metrics,
                }
            )
    return rows

def round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["auc", "ci_lo", "ci_hi", "sens_at_0p5", "spec_at_0p5", "opt_thresh", "sens_at_opt", "spec_at_opt"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: round(float(x), 4) if pd.notna(x) else x)
    return out


def main() -> None:
    rows = collect_rows()
    all_df = pd.DataFrame(rows)
    all_df = all_df.sort_values(["auc", "n_valid", "file"], ascending=[False, False, True]).reset_index(drop=True)
    all_df = round_numeric_columns(all_df)

    main_df = all_df[all_df["file"].isin(MAIN_FILES)].copy()
    main_df = main_df.set_index("file").loc[[f for f in MAIN_FILES if f in set(main_df["file"])]].reset_index()

    all_path = RESULTS_DIR / "luna25_model_metrics_all.csv"
    main_path = RESULTS_DIR / "luna25_model_metrics_main.csv"
    all_df.to_csv(all_path, index=False)
    main_df.to_csv(main_path, index=False)

    print(f"Wrote {all_path}")
    print(f"Wrote {main_path}")
    print("\nMain models:")
    if main_df.empty:
        print("  (none)")
    else:
        for _, row in main_df.iterrows():
            print(
                f"  {row['file']}: AUC={row['auc']:.4f} "
                f"[{row['ci_lo']:.4f}-{row['ci_hi']:.4f}] "
                f"Sens@0.5={row['sens_at_0p5']:.4f} Spec@0.5={row['spec_at_0p5']:.4f}"
            )


if __name__ == "__main__":
    main()
