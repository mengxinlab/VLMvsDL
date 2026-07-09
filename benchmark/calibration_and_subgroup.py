"""Calibration analysis + subgroup analysis for VLMvsDL.

The primary manuscript analysis uses the single-run F3 estimate.
This script therefore computes Fig. 5 / Supplementary Fig. 1 / Supplementary
Table calibration and subgroup outputs from that same primary F3 source.
The five-run averaged estimator F3A is reported separately in the manuscript
as a stability/sensitivity analysis and is not the default input here.

Figure PDFs are written straight into manuscript/figures/ (no manual copy);
the CSV side-products stay in results/figures/.

Outputs:
  manuscript/figures/fig5_calibration.pdf      — reliability + predicted-prob histogram
  manuscript/figures/fig6_subgroup_size.pdf    — AUC by nodule size bin (Supp Fig 1)
  manuscript/figures/fig6_subgroup_density.pdf — AUC by nodule density type (Supp Fig 1)
  results/figures/calibration_stats.csv         — ECE, MCE, Brier per model
  results/figures/subgroup_stats.csv            — AUC per (model, subgroup)
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
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

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
    LUNA25_CLINICAL_METADATA_CSV,
    MANUSCRIPT_FIGURES_DIR,
    PATIENT_SPLIT_JSON,
    VLM_RESULTS_DIR,
)

CSV_DIR = FIGURES_DIR          # CSV side-products
FIGS = MANUSCRIPT_FIGURES_DIR  # manuscript PDF figures (single destination)
CSV_DIR.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
apply_style()
RNG = np.random.default_rng(42)
N_BOOT = 1000
N_CAL_BINS = 10

# ── Models to analyse ──────────────────────────────────────────────────────
# F3 is the primary single-run estimate; Z1/F2 stay
# single-run as before. The five-run average (F3A) is reported separately as a sensitivity
# analysis in the manuscript and Supplementary Table 4.
VLM_RUNS = {
    "Gemini 3 Flash (F3)": "luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl",
    "Gemini 3 Flash (Z1)": "luna25_ablation/z1_gemini3flash_zeroshot_minimal.jsonl",
    "Gemini 3 Flash (F2)": "luna25_ablation/f2_gemini3flash_20shot_rich.jsonl",
    "MedGemma 1.5-4B":     "luna25_model_comparison/f3_medgemma15_4b.jsonl",
}
DL_FILES = {
    "STU-Net":        "stunet_base_warmup_test_preds.csv",
    "EfficientNet-B0": "efficientnet_b0_baseline_test_preds.csv",
    "ResNet-18":      "resnet18_baseline_test_preds.csv",
    "DenseNet-121":   "densenet121_baseline_test_preds.csv",
    "ResNet-50":      "resnet50_baseline_test_preds.csv",
    "Swin-UNETR":     "swin_unetr_final_gpu_test_preds.csv",
    "ViT-Base":       "vit_baseline_test_preds.csv",
}

DENSITY_MAP = {1: "Solid", 2: "Part-solid", 3: "GGN"}
SIZE_BINS = [0, 6, 8, 200]
SIZE_LABELS = ["<6 mm", "6–8 mm", ">8 mm"]


# ── Loaders ────────────────────────────────────────────────────────────────
def load_vlm(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path).rename(columns={"AnnotationID": "aid"})
    else:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(rows)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df[df["confidence"].between(0.0, 1.0)]
    return df.drop_duplicates("aid", keep="last")[["aid", "label", "confidence"]].reset_index(drop=True)


def load_dl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["split"] == "test"].copy()
    return df.rename(columns={"AnnotationID": "aid", "pred_prob": "confidence"})[["aid", "label", "confidence"]]


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(LUNA25_CLINICAL_METADATA_CSV)
    split = json.load(PATIENT_SPLIT_JSON.open())
    test_pids = [int(p) for p in split["test"]]
    meta = meta[meta["PatientID"].isin(test_pids)].copy()
    meta = meta.rename(columns={"AnnotationID": "aid"})
    meta["size_bin"] = pd.cut(meta["sct_long_dia"], bins=SIZE_BINS, labels=SIZE_LABELS, right=True)
    meta["density"] = meta["sct_pre_att"].map(DENSITY_MAP)
    return meta[["aid", "size_bin", "density", "sct_long_dia"]].copy()


# ── Calibration metrics ───────────────────────────────────────────────────
def expected_calibration_error(y_true, y_prob, n_bins=N_CAL_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        frac = mask.sum() / len(y_true)
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        gap = abs(acc - conf)
        ece += frac * gap
        mce = max(mce, gap)
    return ece, mce


def bootstrap_auc(y, s, n=N_BOOT, alpha=0.05):
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    aucs = np.empty(n)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return float(roc_auc_score(y, s)), float("nan"), float("nan")
    for i in range(n):
        p = RNG.choice(pos_idx, size=len(pos_idx), replace=True)
        q = RNG.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, q])
        try:
            aucs[i] = roc_auc_score(y[idx], s[idx])
        except ValueError:
            aucs[i] = float("nan")
    aucs = aucs[~np.isnan(aucs)]
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return float(roc_auc_score(y, s)), float(lo), float(hi)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Load all predictions
    preds = {}
    for name, fn in VLM_RUNS.items():
        p = VLM_RESULTS_DIR / fn
        if p.exists():
            preds[name] = load_vlm(p)
        else:
            print(f"  skip VLM (missing): {fn}")
    for name, fn in DL_FILES.items():
        p = LUNA25_DL_PRED_DIR / fn
        if p.exists():
            preds[name] = load_dl(p)
        else:
            print(f"  skip DL (missing): {fn}")

    meta = load_metadata()
    print(f"Loaded {len(preds)} models, metadata for {len(meta)} test annotations\n")

    # ── 1. Calibration analysis ────────────────────────────────────────────
    print("=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)

    cal_rows = []
    for name, df in preds.items():
        y = df["label"].values.astype(int)
        s = df["confidence"].values.astype(float)
        brier = brier_score_loss(y, s)
        ece, mce = expected_calibration_error(y, s)
        auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")
        # Keep 6-dp values for any downstream math, plus a single
        # round-half-up 3-dp display string the manuscript tables read
        # verbatim (prevents double-rounding drift, e.g. 0.69947 -> 0.6995
        # -> 0.700 instead of the correct 0.699).
        cal_rows.append({"model": name,
                         "auc": round(auc, 6), "brier": round(brier, 6),
                         "ece": round(ece, 6), "mce": round(mce, 6),
                         "auc_str": rounded_3(auc), "brier_str": rounded_3(brier),
                         "ece_str": rounded_3(ece), "mce_str": rounded_3(mce),
                         "n": len(y)})
        print(f"  {name:30s}  AUC={auc:.4f}  Brier={brier:.4f}  ECE={ece:.4f}  MCE={mce:.4f}")

    cal_df = pd.DataFrame(cal_rows).sort_values("brier")
    cal_df.to_csv(CSV_DIR / "calibration_stats.csv", index=False)
    print(f"\n-> {CSV_DIR / 'calibration_stats.csv'}")

    # ── Figure 5: reliability + predicted-prob histogram + ECE/Brier bars ──
    # Four key models on the reliability/histogram panels (the rest stay in
    # the full Supplementary calibration table); panel (c) keeps all models.
    key_models = ["Gemini 3 Flash (F3)", "Gemini 3 Flash (Z1)",
                  "STU-Net", "EfficientNet-B0"]
    key_models = [m for m in key_models if m in preds]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.8, 3.8),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]},
    )
    shared_handles = []

    # (a) reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#9A9A9A", ls="--", lw=0.9)
    ax.text(0.07, 0.90, "Perfect", color="#777777", transform=ax.transAxes)
    for name in key_models:
        df = preds[name]
        y = df["label"].values.astype(int)
        s = df["confidence"].values.astype(float)
        prob_true, prob_pred = calibration_curve(y, s, n_bins=N_CAL_BINS, strategy="uniform")
        ece, _ = expected_calibration_error(y, s)
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            markersize=4,
            lw=1.6,
            color=color_for(name),
            ls=linestyle_for(name),
        )
        shared_handles.append(
            Line2D(
                [0],
                [0],
                color=color_for(name),
                ls=linestyle_for(name),
                marker="o",
                markersize=4,
                lw=1.6,
                label=f"{display_name(name)} (ECE {rounded_3(ece)})",
            )
        )
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed malignancy")
    ax.set_title("Reliability")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    add_panel_label(ax, "a")

    # (b) predicted-probability histogram (score-scale view)
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    for name in key_models:
        s = preds[name]["confidence"].values.astype(float)
        ax.hist(s, bins=bins, histtype="step", lw=1.6, density=True,
                color=color_for(name), ls=linestyle_for(name))
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution")
    add_panel_label(ax, "b")

    # (c) ECE + Brier paired dot plot (all models, ascending ECE)
    ax = axes[2]
    cal_sorted = cal_df.sort_values("ece")
    y_pos = np.arange(len(cal_sorted))[::-1]
    for yp, row in zip(y_pos, cal_sorted.itertuples()):
        ax.hlines(yp, row.ece, row.brier, color="#D0D0D0", lw=0.9, zorder=1)
        ax.scatter(
            row.ece,
            yp,
            s=22,
            marker="o",
            color="#4C78A8",
            edgecolor="black",
            linewidth=0.25,
            zorder=3,
            label="ECE" if yp == y_pos[0] else None,
        )
        ax.scatter(
            row.brier,
            yp,
            s=24,
            marker="s",
            color="#F58518",
            edgecolor="black",
            linewidth=0.25,
            zorder=3,
            label="Brier" if yp == y_pos[0] else None,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([display_name(m) for m in cal_sorted["model"]])
    ax.set_xlabel("Score (lower is better)")
    ax.set_xlim(0, max(cal_sorted["brier"].max(), cal_sorted["ece"].max()) + 0.05)
    ax.set_title("Calibration metrics")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=1.3)
    add_panel_label(ax, "c")

    fig.legend(
        handles=shared_handles,
        loc="lower center",
        bbox_to_anchor=(0.35, 0.06),
        ncol=2,
        columnspacing=1.1,
        handletextpad=0.5,
        handlelength=2.2,
    )
    fig.subplots_adjust(top=0.84, bottom=0.34, left=0.07, right=0.985, wspace=0.65)
    plt.savefig(FIGS / "fig5_calibration.pdf")
    plt.close()
    print(f"-> {FIGS / 'fig5_calibration.pdf'}")

    # ── 2. Subgroup analysis ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUBGROUP ANALYSIS")
    print("=" * 60)

    sub_rows = []
    # All seven DL baselines (ResNet-50 and Swin-UNETR added) so the subgroup
    # table/figure covers the same baseline set as the main comparison.
    subgroup_models = ["Gemini 3 Flash (F3)", "STU-Net", "EfficientNet-B0",
                       "ResNet-18", "DenseNet-121", "ResNet-50", "Swin-UNETR",
                       "ViT-Base"]

    for sg_col, sg_name_prefix in [("size_bin", "Size"), ("density", "Density")]:
        for name in subgroup_models:
            if name not in preds:
                continue
            df = preds[name].merge(meta, on="aid", how="inner")
            for grp_val in df[sg_col].dropna().unique():
                sub = df[df[sg_col] == grp_val]
                y = sub["label"].values.astype(int)
                s = sub["confidence"].values.astype(float)
                n_pos = int(y.sum())
                n_total = len(y)
                auc, lo, hi = bootstrap_auc(y, s)
                sub_rows.append({
                    "subgroup_type": sg_name_prefix,
                    "subgroup": str(grp_val),
                    "model": name,
                    "n": n_total,
                    "n_pos": n_pos,
                    "auc": round(auc, 6) if not np.isnan(auc) else None,
                    "ci_lo": round(lo, 6) if not np.isnan(lo) else None,
                    "ci_hi": round(hi, 6) if not np.isnan(hi) else None,
                    "auc_str": rounded_3(auc) if not np.isnan(auc) else None,
                    "ci_str": (f"{rounded_3(lo)}--{rounded_3(hi)}"
                               if not (np.isnan(lo) or np.isnan(hi)) else None),
                })
                flag = " *" if n_pos < 10 else ""
                print(f"  {sg_name_prefix:8s} {str(grp_val):12s} {name:25s} "
                      f"AUC={auc:.3f} n={n_total} n+={n_pos}{flag}")

    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(CSV_DIR / "subgroup_stats.csv", index=False)
    print(f"\n-> {CSV_DIR / 'subgroup_stats.csv'}")

    # ── Subgroup figures (Supplementary Fig 1) ─────────────────────────────
    # Use a forest-style point+CI layout: the table carries the decimals,
    # while the figure emphasises relative ordering and uncertainty.
    AUC_DESC_ORDER = ["STU-Net", "EfficientNet-B0", "ResNet-18", "DenseNet-121",
                      "ResNet-50", "Swin-UNETR", "Gemini 3 Flash (F3)", "ViT-Base"]
    for sg_type, sg_order, fig_suffix in [
        ("Size", SIZE_LABELS, "size"),
        ("Density", ["Solid", "Part-solid", "GGN"], "density"),
    ]:
        sg_data = sub_df[sub_df["subgroup_type"] == sg_type].copy()
        if sg_data.empty:
            continue

        present = set(sg_data["model"].unique())
        models_in_plot = [m for m in AUC_DESC_ORDER if m in present]
        groups = [g for g in sg_order if g in sg_data["subgroup"].unique()]
        n_models = len(models_in_plot)
        n_groups = len(groups)
        centers = np.arange(n_groups)[::-1]
        offsets = np.linspace(-0.28, 0.28, n_models) if n_models > 1 else np.array([0.0])

        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        for sep in np.arange(n_groups - 1) + 0.5:
            ax.axhline(sep, color="#E5E5E5", lw=0.8, zorder=0)

        for j, model in enumerate(models_in_plot):
            for center, grp in zip(centers, groups):
                row = sg_data[(sg_data["model"] == model) & (sg_data["subgroup"] == grp)]
                if row.empty:
                    continue
                r = row.iloc[0]
                if pd.isna(r["auc"]):
                    continue
                auc = float(r["auc"])
                lo = float(r["ci_lo"]) if not pd.isna(r["ci_lo"]) else auc
                hi = float(r["ci_hi"]) if not pd.isna(r["ci_hi"]) else auc
                ax.errorbar(
                    auc,
                    center + offsets[j],
                    xerr=[[auc - lo], [hi - auc]],
                    fmt=marker_for(model),
                    markersize=4.2,
                    lw=1.05,
                    capsize=1.9,
                    color=color_for(model),
                    mec="black",
                    mew=0.28,
                    label=display_name(model) if center == centers[0] else None,
                    zorder=3,
                )

        ax.axvline(0.5, color="#C8C8C8", ls=":", lw=0.8, zorder=0)
        ax.set_yticks(centers)
        ax.set_yticklabels(groups)
        ax.set_xlabel("ROC-AUC")
        ax.set_xlim(0.35, 1.03)
        ax.set_ylim(-0.6, n_groups - 0.4)
        ax.set_title(f"Nodule {sg_type.lower()} strata")

        txt_transform = blended_transform_factory(ax.transAxes, ax.transData)
        for center, grp in zip(centers, groups):
            gr = sg_data[sg_data["subgroup"] == grp].iloc[0]
            star = "*" if gr["n_pos"] < 10 else ""
            ax.text(
                1.01,
                center,
                f"n={int(gr['n'])}, malignant={int(gr['n_pos'])}{star}",
                transform=txt_transform,
                ha="left",
                va="center",
                color="#555555",
            )

        fig.subplots_adjust(left=0.13, right=0.80, top=0.88, bottom=0.30)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.095),
            ncol=4,
            columnspacing=1.2,
            handletextpad=0.45,
            handlelength=1.7,
            frameon=False,
        )
        fig.text(
            0.5,
            0.035,
            "* fewer than 10 malignant nodules; AUC estimates are unstable and exploratory.",
            ha="center",
            color="#555555",
        )

        plt.savefig(FIGS / f"fig6_subgroup_{fig_suffix}.pdf")
        plt.close()
        print(f"-> {FIGS / f'fig6_subgroup_{fig_suffix}.pdf'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
