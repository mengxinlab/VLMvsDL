"""Compute PR-AUC, net-benefit, and cluster-bootstrap LNDb intervals.

Outputs:
  results/vlm/luna25_pr_auc_metrics.csv
  results/vlm/luna25_net_benefit_metrics.csv
  results/vlm/lndb_cluster_bootstrap_metrics.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
VLM = ROOT / "results" / "vlm"
DL_LUNA = ROOT / "data" / "predictions" / "luna25_dl" / "files"
DL_LNDB = ROOT / "data" / "predictions" / "lndb_dl"
LNDB_SHEET = ROOT / "data" / "metadata" / "lndb_10to1_eval.csv"

RNG = np.random.default_rng(20260512)
N_BOOT = 2000


# -----------------------------
# Helpers
# -----------------------------

def load_vlm_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(obj)
    return pd.DataFrame(rows)


def stratified_bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_boot: int = N_BOOT, alpha: float = 0.05):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.nan, np.nan
    values = np.empty(n_boot)
    for i in range(n_boot):
        s_pos = RNG.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = RNG.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([s_pos, s_neg])
        try:
            values[i] = metric_fn(y_true[idx], y_score[idx])
        except Exception:
            values[i] = np.nan
    lo, hi = np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def cluster_bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, cluster_id: np.ndarray, metric_fn, n_boot: int = N_BOOT, alpha: float = 0.05):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    cluster_id = np.asarray(cluster_id)
    clusters = pd.unique(cluster_id)
    cluster_to_idx: dict[object, np.ndarray] = {c: np.where(cluster_id == c)[0] for c in clusters}
    values = np.empty(n_boot)
    for i in range(n_boot):
        sampled = RNG.choice(clusters, size=len(clusters), replace=True)
        idx_parts = [cluster_to_idx[c] for c in sampled]
        idx = np.concatenate(idx_parts)
        try:
            values[i] = metric_fn(y_true[idx], y_score[idx])
        except Exception:
            values[i] = np.nan
    lo, hi = np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def net_benefit(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    """Vickers/Elkin net benefit at a given decision threshold."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    if n == 0:
        return np.nan
    w = threshold / (1.0 - threshold)
    return tp / n - fp / n * w


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "sens": sens,
        "spec": spec,
        "ppv": ppv,
        "npv": npv,
    }


# -----------------------------
# Build LUNA25 prediction table
# -----------------------------

def load_luna25_predictions() -> dict[str, pd.DataFrame]:
    """Return {model_name: DataFrame[AnnotationID, label, score]} for all benchmarked LUNA25 models."""
    out: dict[str, pd.DataFrame] = {}

    # VLM Gemini 3 Flash Preview F3 replicates. Run 00 is the primary
    # manuscript estimate; the five-run mean is retained separately as F3A.
    f3_runs = [
        "luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl",
        "luna25_replicates/f3_gemini3flash_20shot_rich_metadata_run01.jsonl",
        "luna25_replicates/f3_gemini3flash_20shot_rich_metadata_run02.jsonl",
        "luna25_replicates/f3_gemini3flash_20shot_rich_metadata_run03.jsonl",
        "luna25_replicates/f3_gemini3flash_20shot_rich_metadata_run04.jsonl",
    ]
    f3_frames = []
    for fn in f3_runs:
        df = load_vlm_jsonl(VLM / fn)[["aid", "label", "confidence"]].copy()
        df = df.rename(columns={"confidence": "score"})
        f3_frames.append(df)
    # Mean confidence per AnnotationID across the 5 runs (F3A sensitivity).
    merged = f3_frames[0][["aid", "label"]].copy()
    for i, df in enumerate(f3_frames):
        merged[f"score_run{i}"] = df.set_index("aid").loc[merged["aid"], "score"].values
    merged["score"] = merged[[c for c in merged.columns if c.startswith("score_run")]].mean(axis=1)
    out["Gemini 3 Flash Preview (F3A mean-of-5)"] = merged[["aid", "label", "score"]]
    out["Gemini 3 Flash Preview (F3 primary run)"] = f3_frames[0]

    # Other Gemini generations under F3 (single run).
    for label, fn in [
        ("Gemini 2.5 Flash (F3)", "luna25_model_comparison/f3_gemini25flash.jsonl"),
        ("Gemini 2.5 Pro (F3)", "luna25_model_comparison/f3_gemini25pro.jsonl"),
        ("Gemini 3.1 Pro Preview (F3)", "luna25_model_comparison/f3_gemini31pro.jsonl"),
    ]:
        df = load_vlm_jsonl(VLM / fn)[["aid", "label", "confidence"]].rename(columns={"confidence": "score"})
        out[label] = df

    # MedGemma 1.5-4B specialty.
    df = load_vlm_jsonl(VLM / "luna25_model_comparison/f3_medgemma15_4b.jsonl")
    df = df[["aid", "label", "confidence"]].rename(columns={"confidence": "score"})
    out["MedGemma 1.5-4B (F3)"] = df

    # Seven DL baselines.
    for label, fn in [
        ("STU-Net", "stunet_base_warmup_test_preds.csv"),
        ("EfficientNet-B0", "efficientnet_b0_baseline_test_preds.csv"),
        ("ResNet-18", "resnet18_baseline_test_preds.csv"),
        ("DenseNet-121", "densenet121_baseline_test_preds.csv"),
        ("ResNet-50", "resnet50_baseline_test_preds.csv"),
        ("Swin-UNETR", "swin_unetr_final_gpu_test_preds.csv"),
        ("ViT-Base", "vit_baseline_test_preds.csv"),
    ]:
        df = pd.read_csv(DL_LUNA / fn)
        df = df.rename(columns={"AnnotationID": "aid", "pred_prob": "score"})[["aid", "label", "score"]]
        out[label] = df

    return out


def load_lndb_predictions() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (lndb_sheet, {model_name: DataFrame[FindingID, label, score]})."""
    sheet = pd.read_csv(LNDB_SHEET)
    out: dict[str, pd.DataFrame] = {}

    # VLM external runs (20-shot rich and zero-shot rich).
    for label, fn in [
        ("Gemini 3 Flash (F2)", "lndb_external/f2_gemini3flash_20shot_rich.jsonl"),
        ("Gemini 3 Flash (Z2)", "lndb_external/z2_gemini3flash_zeroshot_rich.jsonl"),
    ]:
        df = load_vlm_jsonl(VLM / fn)
        # The VLM external uses aid for FindingID/filename alignment.
        # Inspect first row to figure out keys.
        if "aid" in df.columns:
            df = df[["aid", "label", "confidence"]].rename(columns={"aid": "FindingID", "confidence": "score"})
        elif "FindingID" in df.columns:
            df = df[["FindingID", "label", "confidence"]].rename(columns={"confidence": "score"})
        else:
            raise RuntimeError(f"Unknown LNDb VLM keys: {df.columns.tolist()}")
        out[label] = df

    # DL baselines.
    for label, fn in [
        ("EfficientNet-B0", "efficientnet_b0_baseline_lndb_preds.csv"),
        ("ResNet-18", "resnet18_baseline_lndb_preds.csv"),
        ("DenseNet-121", "densenet121_baseline_lndb_preds.csv"),
        ("STU-Net", "stunet_base_warmup_lndb_preds.csv"),
        ("ResNet-50", "resnet50_baseline_lndb_preds.csv"),
        ("ViT-Base", "vit_baseline_lndb_preds.csv"),
        ("Swin-UNETR", "swin_unetr_final_gpu_lndb_preds.csv"),
    ]:
        df = pd.read_csv(DL_LNDB / fn)
        if "pred_prob" in df.columns:
            df = df.rename(columns={"pred_prob": "score"})
        out[label] = df[["FindingID", "label", "score"]]

    return sheet, out


# -----------------------------
# Compute and write outputs
# -----------------------------

def compute_luna25_metrics():
    preds = load_luna25_predictions()
    out_pr_rows = []
    out_nb_rows = []

    for name, df in preds.items():
        df = df.dropna(subset=["score", "label"])
        y = df["label"].astype(int).values
        s = df["score"].astype(float).values
        n = len(y)
        if n == 0 or y.sum() == 0:
            continue

        auc = roc_auc_score(y, s)
        pr_auc = average_precision_score(y, s)
        pr_lo, pr_hi = stratified_bootstrap_ci(y, s, average_precision_score)

        out_pr_rows.append({
            "model": name,
            "n": n,
            "n_pos": int(y.sum()),
            "auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "pr_auc_ci_lo": round(pr_lo, 4),
            "pr_auc_ci_hi": round(pr_hi, 4),
        })

        for thr in (0.05, 0.10, 0.15, 0.20, 0.50):
            nb = net_benefit(y, s, thr)
            m = metrics_at_threshold(y, s, thr)
            out_nb_rows.append({
                "model": name,
                "threshold": thr,
                "net_benefit": round(nb, 4),
                "sens": round(m["sens"], 4),
                "spec": round(m["spec"], 4),
                "ppv": round(m["ppv"], 4),
                "npv": round(m["npv"], 4),
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
            })

    # Reference strategies: treat-all and treat-none.
    # treat-all at threshold p_t -> NB = prevalence - (1-prevalence) * p_t/(1-p_t)
    # using LUNA25 test prevalence 81/917
    prev = 81 / 917
    for thr in (0.05, 0.10, 0.15, 0.20, 0.50):
        nb_all = prev - (1 - prev) * thr / (1 - thr)
        out_nb_rows.append({
            "model": "Treat-all (reference)",
            "threshold": thr,
            "net_benefit": round(nb_all, 4),
            "sens": 1.0, "spec": 0.0, "ppv": round(prev, 4), "npv": float("nan"),
            "tp": 81, "fp": 836, "fn": 0, "tn": 0,
        })
        out_nb_rows.append({
            "model": "Treat-none (reference)",
            "threshold": thr,
            "net_benefit": 0.0, "sens": 0.0, "spec": 1.0, "ppv": float("nan"), "npv": round(1 - prev, 4),
            "tp": 0, "fp": 0, "fn": 81, "tn": 836,
        })

    pd.DataFrame(out_pr_rows).to_csv(VLM / "luna25_pr_auc_metrics.csv", index=False)
    pd.DataFrame(out_nb_rows).to_csv(VLM / "luna25_net_benefit_metrics.csv", index=False)
    print("Wrote PR-AUC and net-benefit CSVs.")
    return pd.DataFrame(out_pr_rows), pd.DataFrame(out_nb_rows)


def compute_lndb_cluster_bootstrap():
    sheet, preds = load_lndb_predictions()
    rows = []
    # The LNDb 814-row sheet and every DL prediction CSV are written in the same
    # row order with identical FindingID and label columns, so VLM and DL scores
    # align positionally rather than by merge-key.  We therefore reindex each
    # prediction frame onto the sheet by sequential row position when row counts
    # match, and only fall back to a merge if the lengths disagree (e.g. VLM
    # exports that index by AnnotationID).
    sheet_n = len(sheet)
    for name, df in preds.items():
        if len(df) == sheet_n and "FindingID" in df.columns and (df["FindingID"].values == sheet["FindingID"].values).all():
            y = sheet["label"].astype(int).values
            s = df["score"].astype(float).values
            finding = sheet["FindingID"].values
        else:
            # VLM external runs use one prediction per unique FindingID; broadcast onto the sheet's row order.
            mapping = df.set_index("FindingID")["score"].to_dict()
            s = sheet["FindingID"].map(mapping).astype(float).values
            y = sheet["label"].astype(int).values
            finding = sheet["FindingID"].values
        valid = ~np.isnan(s)
        y, s, finding = y[valid], s[valid], finding[valid]
        if len(y) == 0 or y.sum() == 0:
            continue
        auc = roc_auc_score(y, s)
        pr_auc = average_precision_score(y, s)
        row_lo, row_hi = stratified_bootstrap_ci(y, s, roc_auc_score)
        clu_lo, clu_hi = cluster_bootstrap_ci(y, s, finding, roc_auc_score)
        row_pr_lo, row_pr_hi = stratified_bootstrap_ci(y, s, average_precision_score)
        clu_pr_lo, clu_pr_hi = cluster_bootstrap_ci(y, s, finding, average_precision_score)
        rows.append({
            "model": name,
            "n_rows": len(y),
            "n_findings": int(pd.unique(finding).size),
            "n_pos_rows": int(y.sum()),
            "auc": round(auc, 4),
            "auc_row_lo": round(row_lo, 4),
            "auc_row_hi": round(row_hi, 4),
            "auc_cluster_lo": round(clu_lo, 4),
            "auc_cluster_hi": round(clu_hi, 4),
            "pr_auc": round(pr_auc, 4),
            "pr_auc_row_lo": round(row_pr_lo, 4),
            "pr_auc_row_hi": round(row_pr_hi, 4),
            "pr_auc_cluster_lo": round(clu_pr_lo, 4),
            "pr_auc_cluster_hi": round(clu_pr_hi, 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(VLM / "lndb_cluster_bootstrap_metrics.csv", index=False)
    print("Wrote LNDb cluster bootstrap CSV.")
    return df


if __name__ == "__main__":
    luna_pr, luna_nb = compute_luna25_metrics()
    print("\n== LUNA25 PR-AUC ==")
    print(luna_pr.to_string(index=False))
    print("\n== LUNA25 Net Benefit ==")
    print(luna_nb.to_string(index=False))
    lndb = compute_lndb_cluster_bootstrap()
    print("\n== LNDb Cluster Bootstrap ==")
    print(lndb.to_string(index=False))
