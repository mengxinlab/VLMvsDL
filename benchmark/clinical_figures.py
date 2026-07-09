"""P1 clinically-oriented figures (no inference; from stored predictions).

  fig_decision_curve.pdf    — Vickers/Elkin net benefit 1-50% (+ treat-all/none)
  fig_pr_curves.pdf         — precision-recall curves (+ prevalence reference)
  fig_internal_external.pdf — internal vs external AUC dumbbell (LNDb)
  fig_ablation_waterfall.pdf— full Z/F ablation condition grid

A clinical-safety paper that argues PR-AUC / net benefit / external
validation matter more than ROC-AUC should show those as figures, not only
tables. All curves are computed from the same per-sample files the tables
use; reported table values are unchanged and the curves pass through them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/vlmvsdl_mplconfig"))
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from benchmark.figure_style import (
    LUNA25_PREVALENCE,
    apply_style,
    color_for,
    display_name,
    linestyle_for,
    rounded_3,
)
from benchmark.paths import LUNA25_DL_PRED_DIR, MANUSCRIPT_FIGURES_DIR, VLM_RESULTS_DIR

apply_style()
RES = VLM_RESULTS_DIR
FIGS = MANUSCRIPT_FIGURES_DIR
FIGS.mkdir(parents=True, exist_ok=True)

DL_FILES = {
    "STU-Net": "stunet_base_warmup_test_preds.csv",
    "DenseNet-121": "densenet121_baseline_test_preds.csv",
    "EfficientNet-B0": "efficientnet_b0_baseline_test_preds.csv",
}


def load_vlm(fn: str) -> pd.DataFrame:
    p = RES / fn
    if p.suffix == ".csv":
        df = pd.read_csv(p).rename(columns={"AnnotationID": "aid"})
    else:
        df = pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df[df["confidence"].between(0, 1)].drop_duplicates("aid", keep="last")
    return df[["aid", "label", "confidence"]].reset_index(drop=True)


def load_dl(fn: str) -> pd.DataFrame:
    df = pd.read_csv(LUNA25_DL_PRED_DIR / fn)
    df = df[df["split"] == "test"].copy()
    return df.rename(columns={"AnnotationID": "aid", "pred_prob": "confidence"})[
        ["aid", "label", "confidence"]
    ]


def net_benefit(y: np.ndarray, s: np.ndarray, pt: float) -> float:
    pred = s >= pt
    n = len(y)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return tp / n - (fp / n) * (pt / (1 - pt))


def write_ablation_condition_grid() -> None:
    canon = pd.read_csv(RES / "canonical_numbers.csv")
    cd = {r["name"]: float(r["auc"]) for _, r in
          canon[canon["group"] == "ablation"].iterrows()}
    conditions = [
        ("Z1", "zero-shot\nminimal", "Z1 zero-shot minimal", color_for("Z1 zero-shot minimal")),
        ("Z2", "zero-shot\nrich", "Z2 zero-shot rich", color_for("Z2 zero-shot rich")),
        ("Z3", "zero-shot\nrich+clinical", "Z3 zero-shot rich+clinical", color_for("Z3 zero-shot rich+clinical")),
        ("F1", "20-shot\nminimal", "F1 20-shot minimal", color_for("F1 20-shot minimal")),
        ("F2", "20-shot\nrich", "F2 20-shot rich", color_for("F2 20-shot rich")),
        ("F3", "20-shot\nrich+clinical", "F3 20-shot rich+clinical", color_for("F3 20-shot+clinical")),
    ]
    xs = np.array([0, 1, 2, 3.4, 4.4, 5.4])
    vals = [cd[key] for _, _, key, _ in conditions]
    colors = [color for *_, color in conditions]
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.bar(xs, vals, width=0.62, color=colors, edgecolor="black", lw=0.5)
    for x, (tag, _, _, _), val in zip(xs, conditions, vals):
        ax.text(x, val + 0.003, f"{tag}\n{rounded_3(val)}", ha="center",
                va="bottom", fontsize=8)
    ax.axvline(2.7, color="#bbbbbb", lw=0.9, ls=":")
    ax.text(1, 0.754, "Zero-shot", ha="center", va="bottom",
            fontsize=9, weight="bold")
    ax.text(4.4, 0.754, "20-shot", ha="center", va="bottom",
            fontsize=9, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([desc for _, desc, _, _ in conditions])
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.66, 0.765)
    ax.set_title("Gemini Z/F prompt ablation on LUNA25")
    fig.subplots_adjust(bottom=0.24, top=0.88)
    fig.text(0.5, 0.055,
             "Z3 nearly matches F3; adding structured clinical text changes AUC by "
             f"+{cd['Z3 zero-shot rich+clinical'] - cd['Z2 zero-shot rich']:.3f} "
             f"(zero-shot) and +{cd['F3 20-shot rich+clinical'] - cd['F2 20-shot rich']:.3f} "
             "(20-shot).",
             ha="center", fontsize=8, color="#555555", style="italic")
    fig.savefig(FIGS / "fig_ablation_waterfall.pdf")
    plt.close(fig)
    print(f"-> {FIGS / 'fig_ablation_waterfall.pdf'}")


def main() -> None:
    key = {
        "STU-Net": load_dl(DL_FILES["STU-Net"]),
        "DenseNet-121": load_dl(DL_FILES["DenseNet-121"]),
        "Gemini 3 Flash Preview (F3)": load_vlm("luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl"),
        "MedGemma 1.5-4B": load_vlm(
            "luna25_model_comparison/f3_medgemma15_4b.jsonl"),
    }
    prev = LUNA25_PREVALENCE

    # ── 1. Decision curve (net benefit) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    pts = np.linspace(0.01, 0.50, 99)
    for name, df in key.items():
        y = df["label"].values.astype(int)
        s = df["confidence"].values.astype(float)
        nb = [net_benefit(y, s, t) for t in pts]
        ax.plot(pts * 100, nb, lw=1.8, color=color_for(name),
                ls=linestyle_for(name), label=display_name(name))
    ax.plot(pts * 100, [prev - (1 - prev) * (t / (1 - t)) for t in pts],
            lw=1.2, color="#444444", ls=(0, (4, 2)), label="Treat all")
    ax.axhline(0.0, lw=1.2, color="#888888", ls=":", label="Treat none")
    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(1, 50)
    ax.set_ylim(-0.05, prev + 0.01)
    ax.set_xticks([5, 10, 15, 20, 30, 40, 50])
    ax.legend(loc="upper right", handlelength=2.6)
    fig.savefig(FIGS / "fig_decision_curve.pdf")
    plt.close(fig)
    print(f"-> {FIGS / 'fig_decision_curve.pdf'}")

    # ── 2. Precision-recall curves ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for name, df in key.items():
        y = df["label"].values.astype(int)
        s = df["confidence"].values.astype(float)
        prec, rec, _ = precision_recall_curve(y, s)
        ax.plot(rec, prec, lw=1.8, color=color_for(name),
                ls=linestyle_for(name),
                label=f"{display_name(name)}  {rounded_3(average_precision_score(y, s))}")
    ax.axhline(prev, lw=1.2, color="#888888", ls=":",
               label=f"Prevalence = {rounded_3(prev)}")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", handlelength=2.6)
    fig.savefig(FIGS / "fig_pr_curves.pdf")
    plt.close(fig)
    print(f"-> {FIGS / 'fig_pr_curves.pdf'}")

    # ── 3. Internal vs external AUC dumbbell (LNDb) ───────────────────────
    ext = pd.read_csv(RES / "lndb_external_metrics.csv")
    ext = ext.sort_values("external_auc", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.7, 4.6))
    for i, r in ext.iterrows():
        c = color_for(r["model"])
        ax.plot([r["internal_auc"], r["external_auc"]], [i, i], color=c, lw=2.0,
                zorder=1)
        ax.scatter([r["internal_auc"]], [i], color=c, s=42, marker="o",
                   zorder=2, label="Internal (LUNA25)" if i == 0 else None)
        ax.scatter([r["external_auc"]], [i], color=c, s=58, marker="D",
                   edgecolor="black", linewidth=0.4, zorder=3,
                   label="External (LNDb)" if i == 0 else None)
    ax.set_yticks(range(len(ext)))
    ax.set_yticklabels([display_name(str(m)) for m in ext["model"]])
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0.55, 0.92)
    ax.axvline(0.5, lw=0.8, color="#cccccc", ls=":")
    ax.legend(loc="lower right")
    fig.savefig(FIGS / "fig_internal_external.pdf")
    plt.close(fig)
    print(f"-> {FIGS / 'fig_internal_external.pdf'}")

    # ── 4. Ablation condition grid (kept at the historical filename) ───────
    write_ablation_condition_grid()


if __name__ == "__main__":
    main()
