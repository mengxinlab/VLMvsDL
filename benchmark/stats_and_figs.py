"""Bootstrap CIs + DeLong P-values + Figures (ROC + bar chart) for VLMvsDL.

Outputs (in BENCH/figures/):
  fig2_roc.pdf       — ROC curves: 6 ablation conditions + 4 Gemini generations
  fig3_bar.pdf       — Adaptation spectrum, including 20-shot Med3D+TTT range
  stats.csv          — AUC + 95% bootstrap CI per (model, condition)
  delong.csv         — Pairwise DeLong P-values among VLM runs

Note: DL baselines only have point AUCs (per-sample probs unavailable),
so DeLong VLM-vs-DL is not computed here.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/vlmvsdl_mplconfig"))
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from benchmark.figure_style import (
    add_panel_label,
    apply_style,
    color_for,
    display_name,
    linestyle_for,
    marker_for,
    rounded_3,
)
from benchmark.paths import (
    FIGURES_DIR,
    LUNA25_DL_PRED_DIR,
    MANUSCRIPT_FIGURES_DIR,
    PROJECT_ROOT,
    VLM_RESULTS_DIR,
)

BENCH = Path(__file__).parent
RES   = VLM_RESULTS_DIR
CSV_DIR = FIGURES_DIR              # stats.csv / delong.csv side-products
FIGS = MANUSCRIPT_FIGURES_DIR      # manuscript PDF figures (single destination)
CSV_DIR.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
apply_style()

RNG = np.random.default_rng(42)
N_BOOT = 1000

# ── result files of interest ───────────────────────────────────────────────
# Six-condition Z/F ablation on Gemini 3 Flash Preview.
ABLATION = {
    "Z1 zero-shot minimal":   "luna25_ablation/z1_gemini3flash_zeroshot_minimal.jsonl",
    "Z2 zero-shot rich":      "luna25_ablation/z2_gemini3flash_zeroshot_rich.jsonl",
    "Z3 zero-shot rich+clinical": "luna25_ablation/z3_gemini3flash_zeroshot_rich_metadata.jsonl",
    "F1 20-shot minimal":     "luna25_ablation/f1_gemini3flash_20shot_minimal.jsonl",
    "F2 20-shot rich":        "luna25_ablation/f2_gemini3flash_20shot_rich.jsonl",
    # Primary single-run F3. Four additional F3 repeats are used;
    # the five-run average is reported separately as a sensitivity analysis.
    "F3 20-shot rich+clinical": "luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl",
}
# Fig 2(b) shows the four Gemini generations under F3 (caption-accurate);
# MedGemma is a specialty VLM shown in Fig 3 / the tables, not on this ROC.
GEMINI_GENERATIONS = {
    "Gemini 2.5 Flash":        "luna25_model_comparison/f3_gemini25flash.jsonl",
    "Gemini 2.5 Pro":          "luna25_model_comparison/f3_gemini25pro.jsonl",
    "Gemini 3 Flash Preview":  "luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl",
    "Gemini 3.1 Pro Preview":  "luna25_model_comparison/f3_gemini31pro.jsonl",
}
GENERATIONS = {**GEMINI_GENERATIONS,
    "MedGemma 1.5-4B":         "luna25_model_comparison/f3_medgemma15_4b.jsonl",
}
# DL baselines are read from per-sample test predictions (single source of
# truth, no hand-typed AUCs). Paired DeLong P vs the primary single-run F3 is
# computed on the fly from the stored per-sample predictions.
DL_FILES = {
    "STU-Net":         "stunet_base_warmup_test_preds.csv",
    "EfficientNet-B0": "efficientnet_b0_baseline_test_preds.csv",
    "ResNet-18":       "resnet18_baseline_test_preds.csv",
    "DenseNet-121":    "densenet121_baseline_test_preds.csv",
    "ResNet-50":       "resnet50_baseline_test_preds.csv",
    "Swin-UNETR":      "swin_unetr_final_gpu_test_preds.csv",
    "ViT-Base":        "vit_baseline_test_preds.csv",
}
TTT_SUMMARY = PROJECT_ROOT / "results" / "ttt" / "med3d_resnet18_adaptation_metrics.csv"


def load_dl_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["split"] == "test"].copy()
    return df.rename(columns={"AnnotationID": "aid", "pred_prob": "confidence"})[
        ["aid", "label", "confidence"]
    ]


def delong_p_vs_best(best_df: pd.DataFrame) -> dict[str, float]:
    y_best, s_best = ys(best_df)
    out = {}
    for name, fn in DL_FILES.items():
        ddf = load_dl_test(LUNA25_DL_PRED_DIR / fn)
        merged = best_df.merge(ddf, on="aid", suffixes=("_best", "_dl"))
        y = merged["label_best"].values.astype(int)
        s1 = merged["confidence_best"].values.astype(float)
        s2 = merged["confidence_dl"].values.astype(float)
        _, _, p = delong_p(s1, s2, y)
        out[name] = p
    return out


def sig_marker(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def load_20shot_ttt_range() -> tuple[float, float] | None:
    if not TTT_SUMMARY.exists():
        return None
    df = pd.read_csv(TTT_SUMMARY)
    sub = df[(df["split"] == "luna25") & (df["config"].str.contains("20shot", na=False))]
    if sub.empty:
        return None
    return float(sub["auc"].min()), float(sub["auc"].max())

# ── helpers ────────────────────────────────────────────────────────────────
# Single project-wide round-half-up convention (benchmark.figure_style).
fmt3 = rounded_3


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        df = df.rename(columns={"AnnotationID": "aid"})
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        df = df[df["confidence"].between(0.0, 1.0, inclusive="both")]
        return df[["aid", "label", "confidence"]].drop_duplicates("aid", keep="last").reset_index(drop=True)
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        record["_line"] = line_no
        rows.append(record)
    df = pd.DataFrame(rows)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df[df["confidence"].between(0.0, 1.0, inclusive="both")]
    df = df.sort_values("_line")[["aid", "label", "confidence"]]
    return df.drop_duplicates("aid", keep="last").reset_index(drop=True)

def ys(df):
    return df["label"].values.astype(int), df["confidence"].values.astype(float)

def bootstrap_ci(y, s, n=N_BOOT, alpha=0.05):
    aucs = np.empty(n)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    for i in range(n):
        p = RNG.choice(pos_idx, size=len(pos_idx), replace=True)
        q = RNG.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, q])
        aucs[i] = roc_auc_score(y[idx], s[idx])
    lo, hi = np.quantile(aucs, [alpha/2, 1 - alpha/2])
    return float(roc_auc_score(y, s)), float(lo), float(hi)

# ── DeLong (Sun & Xu 2014 fast implementation) ────────────────────────────
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N)
    out[J] = T
    return out

def delong_var(scores_list, y):
    """scores_list: list of arrays of equal length aligned with y.
    Returns (AUC vector, covariance matrix) following DeLong."""
    pos = y == 1
    neg = ~pos
    m, n = pos.sum(), neg.sum()
    k = len(scores_list)
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    aucs = np.empty(k)
    for r, s in enumerate(scores_list):
        sx = s[pos]; sy = s[neg]
        tx[r] = _midrank(sx)
        ty[r] = _midrank(sy)
        tz[r] = _midrank(np.concatenate([sx, sy]))
        aucs[r] = (tz[r, :m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx_ = np.cov(v01, ddof=1)
    sy_ = np.cov(v10, ddof=1)
    if k == 1:
        sx_ = np.array([[sx_]]); sy_ = np.array([[sy_]])
    cov = sx_ / m + sy_ / n
    return aucs, cov

def delong_p(s1, s2, y):
    aucs, cov = delong_var([s1, s2], y)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = diff / np.sqrt(var)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(p)

# ── 1. compute stats ───────────────────────────────────────────────────────
def main():
    runs = {}  # name -> DataFrame(aid,label,confidence)
    for name, fn in {**ABLATION, **GENERATIONS}.items():
        p = RES / fn
        if not p.exists():
            print(f"skip missing: {fn}"); continue
        runs[name] = load(p)

    # bootstrap CI table
    rows = []
    for name, df in runs.items():
        y, s = ys(df)
        auc, lo, hi = bootstrap_ci(y, s)
        rows.append({"run": name, "n": len(y), "auc": round(auc, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
    pd.DataFrame(rows).to_csv(CSV_DIR / "stats.csv", index=False)
    print(f"→ {CSV_DIR / 'stats.csv'}")
    for r in rows: print(f"  {r['run']:35s} AUC={r['auc']:.4f} [{r['ci_lo']:.4f}-{r['ci_hi']:.4f}] n={r['n']}")

    # DeLong pairwise on aid intersection
    drows = []
    names = list(runs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            m = runs[n1].merge(runs[n2], on="aid", suffixes=("_1", "_2"))
            if len(m) < 50: continue
            assert (m["label_1"] == m["label_2"]).all()
            y = m["label_1"].values.astype(int)
            s1 = m["confidence_1"].values.astype(float)
            s2 = m["confidence_2"].values.astype(float)
            a1, a2, p = delong_p(s1, s2, y)
            drows.append({"a": n1, "b": n2, "n": len(m),
                          "auc_a": round(a1,4), "auc_b": round(a2,4),
                          "delta": round(a1-a2,4), "p": f"{p:.4g}"})
    pd.DataFrame(drows).to_csv(CSV_DIR / "delong.csv", index=False)
    print(f"→ {CSV_DIR / 'delong.csv'}")

    # ── Figure 2: ROC (a) Z/F ablation  (b) four Gemini generations under F3 ──
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True, sharey=True)
    for name in ABLATION:
        if name in runs:
            y, s = ys(runs[name]); fpr, tpr, _ = roc_curve(y, s)
            lw = 2.0 if name == "F3 20-shot rich+clinical" else 1.25
            alpha = 1.0 if name == "F3 20-shot rich+clinical" else 0.86
            axs[0].plot(fpr, tpr, lw=lw, alpha=alpha, color=color_for(name),
                        ls=linestyle_for(name),
                        label=f"{display_name(name)}  {fmt3(roc_auc_score(y, s))}")
    axs[0].plot([0, 1], [0, 1], "--", c="#9A9A9A", lw=0.9, label="Chance")
    axs[0].set(xlabel="1 - Specificity", ylabel="Sensitivity",
               title="Z/F prompt ablation")
    axs[0].legend(loc="lower right", handlelength=2.1, ncols=2, columnspacing=0.9)
    axs[0].set_box_aspect(1)
    add_panel_label(axs[0], "a")

    for name in GEMINI_GENERATIONS:
        if name in runs:
            y, s = ys(runs[name]); fpr, tpr, _ = roc_curve(y, s)
            axs[1].plot(fpr, tpr, lw=1.7, color=color_for(name),
                        ls=linestyle_for(name),
                        label=f"{display_name(name)}  {fmt3(roc_auc_score(y, s))}")
    axs[1].plot([0, 1], [0, 1], "--", c="#9A9A9A", lw=0.9, label="Chance")
    axs[1].set(xlabel="1 - Specificity", ylabel="Sensitivity",
               title="Gemini generations (F3)")
    axs[1].legend(loc="lower right", handlelength=2.6)
    axs[1].set_box_aspect(1)
    add_panel_label(axs[1], "b")
    for ax in axs:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    plt.tight_layout(w_pad=1.2)
    plt.savefig(FIGS / "fig2_roc.pdf")
    plt.close()
    print(f"→ {FIGS / 'fig2_roc.pdf'}")

    # ── Figure 3: adaptation spectrum on LUNA25 ──
    # DL AUCs are read from per-sample test predictions (single source);
    # paired-DeLong significance vs the primary single-run F3 is annotated
    # on each bar.
    best_name = "F3 20-shot rich+clinical"
    if best_name not in runs:
        raise RuntimeError(f"Missing required run for Figure 3: {best_name}")
    y, s = ys(runs[best_name])
    f3_auc, f3_lo, f3_hi = bootstrap_ci(y, s)
    dl_p = delong_p_vs_best(runs[best_name])

    items = []  # (label, auc, lo, hi, kind, p_vs_f3, value_text, is_range)
    for name, fn in DL_FILES.items():
        ddf = load_dl_test(LUNA25_DL_PRED_DIR / fn)
        a, l, h = bootstrap_ci(*ys(ddf))
        items.append((name, a, l, h, "DL", dl_p.get(name), fmt3(a), False))
    items.append(("Gemini F3", f3_auc, f3_lo, f3_hi, "VLM", None, fmt3(f3_auc), False))
    if "MedGemma 1.5-4B" in runs:
        amg, lmg, hmg = bootstrap_ci(*ys(runs["MedGemma 1.5-4B"]))
        items.append(("MedGemma 1.5-4B", amg, lmg, hmg, "MedVLM", None, fmt3(amg), False))
    ttt_range = load_20shot_ttt_range()
    if ttt_range is not None:
        lo, hi = ttt_range
        items.append((
            "20-shot Med3D ResNet18+TTT",
            hi,
            lo,
            hi,
            "WeakDL",
            None,
            f"{fmt3(lo)}-{fmt3(hi)}",
            True,
        ))
    items.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    y_pos = np.arange(len(items))[::-1]
    x_max = max(h for _, _, _, h, _, _, _, _ in items) + 0.055

    for yp, (n, a, l, h, kind, p, value_text, is_range) in zip(y_pos, items):
        ax.hlines(yp, l, h, color=color_for(n), lw=1.6, zorder=2)
        if is_range:
            ax.scatter(
                [l, h],
                [yp, yp],
                s=28,
                color=color_for(n),
                marker=marker_for(n),
                edgecolor="black",
                linewidth=0.35,
                zorder=3,
            )
        else:
            ax.scatter(
                a,
                yp,
                s=34,
                color=color_for(n),
                marker=marker_for(n),
                edgecolor="black",
                linewidth=0.35,
                zorder=3,
            )
        ax.text(h + 0.006, yp, value_text, va="center", ha="left")
        if p is not None:
            ax.text(x_max - 0.001, yp, sig_marker(p), va="center", ha="right", color="#555555")

    ax.axvline(f3_auc, ls=(0, (2, 2)), c=color_for("Gemini 3 Flash Preview (F3)"),
               lw=1.0, alpha=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([display_name(n) for n, *_ in items])
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0.55, x_max)
    ax.set_ylim(-0.8, len(items) - 0.2)
    ax.set_title("LUNA25 adaptation spectrum")
    ax.text(
        f3_auc + 0.003,
        len(items) - 0.45,
        "Gemini F3",
        color=color_for("Gemini 3 Flash Preview (F3)"),
        ha="left",
        va="bottom",
    )
    ax.text(
        0.0,
        -0.18,
        "Error bars: 95% bootstrap CI; 20-shot Med3D+TTT: observed range. DL markers: paired DeLong vs Gemini F3.",
        transform=ax.transAxes,
        color="#555555",
    )
    plt.tight_layout()
    plt.savefig(FIGS / "fig3_bar.pdf")
    plt.close()
    print(f"→ {FIGS / 'fig3_bar.pdf'}")

if __name__ == "__main__":
    main()
