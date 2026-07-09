#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

from benchmark.paths import LNDB_DL_PRED_DIR, LNDB_EVAL_CSV, PROJECT_ROOT, VLM_RESULTS_DIR

BENCH = Path(__file__).parent
ROOT = PROJECT_ROOT
RESULTS = VLM_RESULTS_DIR
LNDB_DL_DIR = LNDB_DL_PRED_DIR
RNG_SEED = 42
N_BOOT = 1000

VLM_RUNS = {
    "Gemini 3 Flash Preview zero-shot rich": RESULTS / "lndb_external/z2_gemini3flash_zeroshot_rich.jsonl",
    "Gemini 3 Flash Preview 20-shot rich": RESULTS / "lndb_external/f2_gemini3flash_20shot_rich.jsonl",
}

DL_PER_SAMPLE = {
    "EfficientNet-B0": "efficientnet_b0_baseline_lndb_preds.csv",
    "ResNet-18": "resnet18_baseline_lndb_preds.csv",
    "DenseNet-121": "densenet121_baseline_lndb_preds.csv",
    "ResNet-50": "resnet50_baseline_lndb_preds.csv",
    "ViT-Base": "vit_baseline_lndb_preds.csv",
    "STU-Net": "stunet_base_warmup_lndb_preds.csv",
    "Swin-UNETR": "swin_unetr_final_gpu_lndb_preds.csv",
}

INTERNAL_AUC = {
    "Gemini 3 Flash Preview zero-shot rich": 0.6823,
    "Gemini 3 Flash Preview 20-shot rich": 0.6995,
    "EfficientNet-B0": 0.8632,
    "ResNet-18": 0.8564,
    "DenseNet-121": 0.8327,
    "STU-Net": 0.8722,
    "ResNet-50": 0.8167,
    "ViT-Base": 0.6933,
    "Swin-UNETR": 0.7999,
}

def load_dl_predictions(name: str, fname: str) -> pd.DataFrame:
    path = LNDB_DL_DIR / fname
    df = pd.read_csv(path, dtype={"FindingID": str})
    if df.groupby("FindingID")["pred_prob"].nunique().max() > 1:
        raise RuntimeError(f"{name}: pred_prob varies within a FindingID")
    df = df.drop_duplicates("FindingID", keep="first")[["FindingID", "label", "pred_prob"]]
    df = df.rename(columns={"label": "label_dl", "pred_prob": "score"})
    return df


def load_vlm_predictions(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            conf = float(record.get("confidence"))
        except (TypeError, ValueError):
            continue
        if not 0.0 <= conf <= 1.0:
            continue
        rows.append(
            {
                "FindingID": str(record["aid"]),
                "label_jsonl": int(record["label"]),
                "score": conf,
                "_line": line_no,
            }
        )
    if not rows:
        raise RuntimeError(f"No valid VLM predictions in {path}")
    df = pd.DataFrame(rows).sort_values("_line")
    return df.drop_duplicates("FindingID", keep="last").drop(columns=["_line"])


def cm_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tn, fp, fn, tp


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    auc = float(roc_auc_score(y_true, y_score))
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    aucs = np.empty(N_BOOT, dtype=float)
    rng = np.random.default_rng(RNG_SEED)
    for i in range(N_BOOT):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        q = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, q])
        aucs[i] = roc_auc_score(y_true[idx], y_score[idx])
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return auc, float(lo), float(hi)


def midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_var(scores: list[np.ndarray], y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = y_true == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())
    k = len(scores)
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    aucs = np.empty(k)
    for r, score in enumerate(scores):
        sx = score[pos]
        sy = score[neg]
        tx[r] = midrank(sx)
        ty[r] = midrank(sy)
        tz[r] = midrank(np.concatenate([sx, sy]))
        aucs[r] = (tz[r, :m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx_cov = np.cov(v01, ddof=1)
    sy_cov = np.cov(v10, ddof=1)
    if k == 1:
        sx_cov = np.array([[sx_cov]])
        sy_cov = np.array([[sy_cov]])
    cov = sx_cov / m + sy_cov / n
    return aucs, cov


def delong_p(score_a: np.ndarray, score_b: np.ndarray, y_true: np.ndarray) -> tuple[float, float, float]:
    aucs, cov = delong_var([score_a, score_b], y_true)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = diff / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(p)


def compute_metrics(name: str, mtype: str, source: str, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    auc, lo, hi = bootstrap_ci(y_true, y_score)
    pred_05 = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = cm_counts(y_true, pred_05)
    sens_05 = tp / (tp + fn) if (tp + fn) else np.nan
    spec_05 = tn / (tn + fp) if (tn + fp) else np.nan

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_idx = int(np.argmax(tpr - fpr))
    opt_thresh = float(thresholds[best_idx])
    pred_opt = (y_score >= opt_thresh).astype(int)
    tn2, fp2, fn2, tp2 = cm_counts(y_true, pred_opt)
    sens_opt = tp2 / (tp2 + fn2) if (tp2 + fn2) else np.nan
    spec_opt = tn2 / (tn2 + fp2) if (tn2 + fp2) else np.nan

    return {
        "model": name,
        "type": mtype,
        "n_rows": int(len(y_true)),
        "n_malignant": int(y_true.sum()),
        "n_benign": int((y_true == 0).sum()),
        "external_auc": auc,
        "internal_auc": INTERNAL_AUC.get(name, np.nan),
        "delta_external_minus_internal": auc - INTERNAL_AUC.get(name, np.nan),
        "ci_lo": lo,
        "ci_hi": hi,
        "sens_at_0p5": float(sens_05),
        "spec_at_0p5": float(spec_05),
        "opt_thresh": opt_thresh,
        "sens_at_opt": float(sens_opt),
        "spec_at_opt": float(spec_opt),
        "source": source,
    }


def main() -> None:
    eval_df = pd.read_csv(LNDB_EVAL_CSV, dtype={"FindingID": str})
    pred_sheet = eval_df.reset_index(names="eval_row").copy()
    n_unique = int(eval_df["FindingID"].nunique())
    y_eval = eval_df["label"].astype(int).values
    metric_rows: list[dict[str, Any]] = []
    all_scores: dict[str, np.ndarray] = {}
    score_kind: dict[str, str] = {}

    for name, path in VLM_RUNS.items():
        pred = load_vlm_predictions(path)
        merged = pred_sheet[["eval_row", "FindingID", "label"]].merge(
            pred, on="FindingID", how="left", validate="many_to_one"
        )
        if merged["score"].isna().any():
            missing = merged.loc[merged["score"].isna(), "FindingID"].drop_duplicates().head(20).tolist()
            raise RuntimeError(f"{name} missing predictions for {len(missing)} finding IDs, examples={missing}")
        if not (merged["label"].astype(int).values == merged["label_jsonl"].astype(int).values).all():
            raise RuntimeError(f"Label mismatch for {name}")

        score_col = name.lower().replace(" ", "_").replace("-", "_") + "_score"
        pred_sheet[score_col] = merged["score"].values
        y_score = merged["score"].astype(float).values
        all_scores[name] = y_score
        score_kind[name] = "VLM"
        row = compute_metrics(name, "VLM", str(path.relative_to(ROOT)), y_eval, y_score)
        row["n_unique_findings"] = n_unique
        metric_rows.append(row)

    for name, fname in DL_PER_SAMPLE.items():
        dl = load_dl_predictions(name, fname)
        merged = pred_sheet[["eval_row", "FindingID", "label"]].merge(
            dl, on="FindingID", how="left", validate="many_to_one"
        )
        if merged["score"].isna().any():
            missing = merged.loc[merged["score"].isna(), "FindingID"].drop_duplicates().head(20).tolist()
            raise RuntimeError(f"{name} missing predictions for {len(missing)} finding IDs, examples={missing}")
        if not (merged["label"].astype(int).values == merged["label_dl"].astype(int).values).all():
            raise RuntimeError(f"Label mismatch for {name}")

        score_col = name.lower().replace(" ", "_").replace("-", "_") + "_score"
        pred_sheet[score_col] = merged["score"].values
        y_score = merged["score"].astype(float).values
        all_scores[name] = y_score
        score_kind[name] = "DL"
        row = compute_metrics(name, "DL", str((LNDB_DL_DIR / fname).relative_to(ROOT)), y_eval, y_score)
        row["n_unique_findings"] = n_unique
        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    metrics = metrics.sort_values(["external_auc", "type", "model"], ascending=[False, True, True])

    comp_rows: list[dict[str, Any]] = []
    zero = all_scores["Gemini 3 Flash Preview zero-shot rich"]
    shot = all_scores["Gemini 3 Flash Preview 20-shot rich"]
    auc_zero, auc_20, p = delong_p(zero, shot, y_eval)
    comp_rows.append(
        {
            "comparison": "Gemini 3 Flash Preview 20-shot rich vs zero-shot rich",
            "n_rows": int(len(eval_df)),
            "n_unique_findings": n_unique,
            "auc_a": auc_zero,
            "auc_b": auc_20,
            "delta_b_minus_a": auc_20 - auc_zero,
            "delong_p": p,
        }
    )

    for vlm_name in VLM_RUNS:
        for dl_name in DL_PER_SAMPLE:
            auc_v, auc_d, pp = delong_p(all_scores[vlm_name], all_scores[dl_name], y_eval)
            comp_rows.append(
                {
                    "comparison": f"{vlm_name} vs {dl_name}",
                    "n_rows": int(len(eval_df)),
                    "n_unique_findings": n_unique,
                    "auc_a": auc_v,
                    "auc_b": auc_d,
                    "delta_b_minus_a": auc_d - auc_v,
                    "delong_p": pp,
                }
            )

    comparisons = pd.DataFrame(comp_rows)

    pred_path = RESULTS / "lndb_external_predictions_814.csv"
    metrics_path = RESULTS / "lndb_external_metrics.csv"
    comp_path = RESULTS / "lndb_external_comparisons.csv"
    pred_sheet.to_csv(pred_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    comparisons.to_csv(comp_path, index=False)

    print(f"Wrote {pred_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {comp_path}")
    print("\nExternal metrics:")
    display = metrics[
        [
            "model",
            "type",
            "n_rows",
            "n_unique_findings",
            "internal_auc",
            "external_auc",
            "delta_external_minus_internal",
            "ci_lo",
            "ci_hi",
        ]
    ].copy()
    for col in ["internal_auc", "external_auc", "delta_external_minus_internal", "ci_lo", "ci_hi"]:
        display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    print(display.to_string(index=False))
    print("\nVLM-vs-DL paired DeLong (selected):")
    sel = comparisons[comparisons["comparison"].str.contains("20-shot rich vs ")].copy()
    for col in ["auc_a", "auc_b", "delta_b_minus_a", "delong_p"]:
        sel[col] = sel[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    print(sel.to_string(index=False))


if __name__ == "__main__":
    main()
