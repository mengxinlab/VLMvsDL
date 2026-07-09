#!/usr/bin/env python3
"""Analyze F0/F3 metadata-control outputs.

"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.paths import (  # noqa: E402
    LUNA25_CLINICAL_METADATA_CSV,
    VLM_RESULTS_DIR,
)

OUT_DIR = VLM_RESULTS_DIR / "luna25_controls"
PERMUTED_CSV = ROOT / "data" / "metadata" / "clinical_texts_permuted_seed42.csv"
RNG = np.random.default_rng(42)
N_BOOT = 1000

RUNS = {
    "F2 image-only": VLM_RESULTS_DIR / "luna25_ablation/f2_gemini3flash_20shot_rich.jsonl",
    "F3 image+text": VLM_RESULTS_DIR / "luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl",
    "F0 text-only": OUT_DIR / "f0_textonly_gemini3flash.jsonl",
    "F3-permuted-text": OUT_DIR / "f3_permuted_metadata_gemini3flash.jsonl",
}
OPTIONAL_RUNS = {"F3-permuted-text"}

COMPARISONS = [
    ("F3 image+text", "F0 text-only"),
    ("F3 image+text", "F2 image-only"),
    ("F3 image+text", "F3-permuted-text"),
    ("F3-permuted-text", "F2 image-only"),
]


def bootstrap_ci(y: np.ndarray, s: np.ndarray, n: int = N_BOOT) -> tuple[float, float, float]:
    aucs = np.empty(n)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    for i in range(n):
        p = RNG.choice(pos_idx, size=len(pos_idx), replace=True)
        q = RNG.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, q])
        aucs[i] = roc_auc_score(y[idx], s[idx])
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return float(roc_auc_score(y, s)), float(lo), float(hi)


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    ranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x))
    out[order] = ranks
    return out


def delong_var(scores_list: list[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = y == 1
    neg = ~pos
    m, n = pos.sum(), neg.sum()
    k = len(scores_list)
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    aucs = np.empty(k)
    for r, scores in enumerate(scores_list):
        sx = scores[pos]
        sy = scores[neg]
        tx[r] = _midrank(sx)
        ty[r] = _midrank(sy)
        tz[r] = _midrank(np.concatenate([sx, sy]))
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


def delong_p(s1: np.ndarray, s2: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    aucs, cov = delong_var([s1, s2], y)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = diff / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(p)


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df[df["confidence"].between(0.0, 1.0)]
    return df.drop_duplicates("aid", keep="last")[["aid", "label", "confidence"]].reset_index(drop=True)


def safe_spearman(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    mask = a.notna() & b.notna()
    if mask.sum() < 5:
        return np.nan, np.nan, int(mask.sum())
    rho, p = spearmanr(a[mask], b[mask])
    return float(rho), float(p), int(mask.sum())


def spic_mean(df: pd.DataFrame, score_col: str = "confidence") -> tuple[float, float, int, int]:
    margin = pd.to_numeric(df["sct_margins"], errors="coerce")
    spic = margin == 3
    other = margin.notna() & ~spic
    return (
        float(df.loc[spic, score_col].mean()),
        float(df.loc[other, score_col].mean()),
        int(spic.sum()),
        int(other.sum()),
    )


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(LUNA25_CLINICAL_METADATA_CSV, low_memory=False)
    meta = meta.rename(columns={"AnnotationID": "aid"})
    keep = ["aid", "sct_long_dia", "sct_margins", "sct_pre_att", "sct_epi_loc"]
    meta = meta[keep].drop_duplicates("aid")
    brock = pd.read_csv(VLM_RESULTS_DIR / "brock_pancan_predictions.csv").rename(
        columns={"AnnotationID": "aid"}
    )
    return meta.merge(brock[["aid", "brock_prob_correct"]], on="aid", how="left")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    available = {}
    for name, path in RUNS.items():
        if path.exists():
            available[name] = load_jsonl(path)
        else:
            prefix = "optional missing" if name in OPTIONAL_RUNS else "missing"
            print(f"{prefix}: {name} -> {path.relative_to(ROOT)}")

    metric_rows = []
    for name, df in available.items():
        y = df["label"].astype(int).to_numpy()
        s = df["confidence"].astype(float).to_numpy()
        auc, lo, hi = bootstrap_ci(y, s)
        metric_rows.append({
            "condition": name,
            "n": len(df),
            "auc": auc,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT_DIR / "metadata_control_metrics.csv", index=False)

    delong_rows = []
    for a, b in COMPARISONS:
        if a not in available or b not in available:
            continue
        merged = available[a].merge(available[b], on="aid", suffixes=("_a", "_b"))
        assert (merged["label_a"] == merged["label_b"]).all()
        y = merged["label_a"].astype(int).to_numpy()
        s1 = merged["confidence_a"].astype(float).to_numpy()
        s2 = merged["confidence_b"].astype(float).to_numpy()
        auc_a, auc_b, p = delong_p(s1, s2, y)
        delong_rows.append({
            "a": a,
            "b": b,
            "n": len(merged),
            "auc_a": auc_a,
            "auc_b": auc_b,
            "delta_a_minus_b": auc_a - auc_b,
            "p": p,
        })
    delongs = pd.DataFrame(delong_rows)
    delongs.to_csv(OUT_DIR / "metadata_control_delong.csv", index=False)

    meta = load_metadata()
    perm = None
    if PERMUTED_CSV.exists():
        perm = pd.read_csv(PERMUTED_CSV)[["target_plain_aid", "source_plain_aid"]]
        perm = perm.rename(columns={"target_plain_aid": "aid"})
        source_meta = meta.add_prefix("source_").rename(columns={"source_aid": "source_plain_aid"})
        perm = perm.merge(source_meta, on="source_plain_aid", how="left")

    audit_rows = []
    for name, pred in available.items():
        merged = pred.merge(meta, on="aid", how="left")
        rho_d, p_d, n_d = safe_spearman(merged["confidence"], pd.to_numeric(merged["sct_long_dia"], errors="coerce"))
        rho_b, p_b, n_b = safe_spearman(merged["confidence"], merged["brock_prob_correct"])
        spic, other, n_spic, n_other = spic_mean(merged)
        audit_rows.append({
            "condition": name,
            "feature_space": "true_metadata",
            "n_diameter": n_d,
            "spearman_diameter": rho_d,
            "spearman_diameter_p": p_d,
            "n_brock": n_b,
            "spearman_brock": rho_b,
            "spearman_brock_p": p_b,
            "mean_score_spiculated_or_irregular": spic,
            "mean_score_other_margin": other,
            "n_spiculated_or_irregular": n_spic,
            "n_other_margin": n_other,
        })
        if name == "F3-permuted-text" and perm is not None:
            assigned = pred.merge(perm, on="aid", how="left")
            rho_ad, p_ad, n_ad = safe_spearman(
                assigned["confidence"],
                pd.to_numeric(assigned["source_sct_long_dia"], errors="coerce"),
            )
            rho_ab, p_ab, n_ab = safe_spearman(
                assigned["confidence"],
                assigned["source_brock_prob_correct"],
            )
            assigned = assigned.rename(columns={"source_sct_margins": "sct_margins"})
            aspic, aother, an_spic, an_other = spic_mean(assigned)
            audit_rows.append({
                "condition": name,
                "feature_space": "permuted_assigned_metadata",
                "n_diameter": n_ad,
                "spearman_diameter": rho_ad,
                "spearman_diameter_p": p_ad,
                "n_brock": n_ab,
                "spearman_brock": rho_ab,
                "spearman_brock_p": p_ab,
                "mean_score_spiculated_or_irregular": aspic,
                "mean_score_other_margin": aother,
                "n_spiculated_or_irregular": an_spic,
                "n_other_margin": an_other,
            })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_DIR / "metadata_control_association_audit.csv", index=False)

    lines = [
        "# Metadata-control summary",
        "",
        "## AUC",
        "",
        metrics.to_markdown(index=False) if not metrics.empty else "No complete runs found.",
        "",
        "## Paired DeLong",
        "",
        delongs.to_markdown(index=False) if not delongs.empty else "No matched comparisons available yet.",
        "",
        "## Association audit",
        "",
        audit.to_markdown(index=False) if not audit.empty else "No association audit available yet.",
        "",
    ]
    if {"F3 image+text", "F2 image-only", "F0 text-only", "F3-permuted-text"}.issubset(available):
        lines.extend([
            "## Plain-language interpretation",
            "",
            "F0 estimates the discrimination available from the supplied structured text "
            "without CT videos. F3-permuted-text keeps the real target video but assigns another "
            "case's structured text. If F0 is close to F3, and F3-permuted-text loses "
            "the F3 advantage while correlating with the assigned metadata, the apparent F3 gain "
            "is best interpreted as metadata-conditioned rather than as independent volumetric CT "
            "image understanding.",
            "",
        ])
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines))

    print(metrics.to_string(index=False) if not metrics.empty else "No metrics.")
    if not delongs.empty:
        print("\nPaired DeLong:")
        print(delongs.to_string(index=False))
    print(f"\nWrote {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
