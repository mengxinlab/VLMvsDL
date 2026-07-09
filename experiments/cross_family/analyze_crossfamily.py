#!/usr/bin/env python3
"""Cross-family metadata-reliance analysis.

Aggregates the zero-shot metadata-control triad (Z2 image-only / Z3 image+text /
Z0 text-only) across VLM families and produces the manuscript table + figure.

Headline readout per family:
    metadata lift  = AUC(image+text) - AUC(image-only)   [paired DeLong]
    text recovery  = AUC(text-only)                       (how much survives w/o images)

Gemini 3 Flash is included as a matched zero-shot reference row using the
existing ablation files (Z2=image-only, Z3=image+text; filenames retain legacy
c2/c3 tags). Hosted-model rows come from run_crossfamily_api.py.

Reuses the bootstrap/DeLong/association helpers from the existing
experiments/metadata_controls/analyze_metadata_controls.py so the statistics are
identical to the rest of the paper.

Run:
    python experiments/cross_family/analyze_crossfamily.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.paths import VLM_RESULTS_DIR  # noqa: E402
from experiments.metadata_controls.analyze_metadata_controls import (  # noqa: E402
    RNG,
    bootstrap_ci,
    delong_p,
    load_jsonl,
    load_metadata,
    safe_spearman,
)

ABLATION = VLM_RESULTS_DIR / "luna25_ablation"
CF_DIR = VLM_RESULTS_DIR / "crossfamily"
OUT_DIR = CF_DIR  # outputs land alongside the per-model JSONLs

# family -> {condition: jsonl path}. Missing files are skipped gracefully.
# Open-family / API filenames match run_crossfamily_offline.py and
# run_crossfamily_api.py: crossfamily_<model-name>_<condition>.jsonl
def cf(model: str, cond: str) -> Path:
    return CF_DIR / f"crossfamily_{model}_{cond}.jsonl"


CONDITIONS = ("image-only", "image-text", "text-only")


def discover_families() -> dict[str, dict[str, Path]]:
    """Gemini reference (Z2/Z3 video files) + auto-discovered cross-family runs.

    Any `crossfamily_<tag>_image-only.jsonl` written by run_crossfamily_api.py
    or another compatible runner is picked up automatically,
    so the analysis works no matter which models you ended up running.
    """
    fams: dict[str, dict[str, Path]] = {
        "Gemini-3-Flash (ref)": {
            "image-only": ABLATION / "z2_gemini3flash_zeroshot_rich.jsonl",
            "image-text": ABLATION / "z3_gemini3flash_zeroshot_rich_metadata.jsonl",
        },
    }
    for p in sorted(CF_DIR.glob("crossfamily_*_image-only.jsonl")):
        tag = p.name[len("crossfamily_"):-len("_image-only.jsonl")]
        fams[tag] = {c: CF_DIR / f"crossfamily_{tag}_{c}.jsonl" for c in CONDITIONS}
    return fams


def to_md(df: pd.DataFrame, empty: str = "n/a") -> str:
    """Markdown table, falling back to plain text if `tabulate` is unavailable."""
    if df.empty:
        return empty
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def delta_auc_ci(y: np.ndarray, s_hi: np.ndarray, s_lo: np.ndarray, n: int = 1000):
    """Paired, label-stratified bootstrap CI for AUC(s_hi) - AUC(s_lo)."""
    from sklearn.metrics import roc_auc_score

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    deltas = np.empty(n)
    for i in range(n):
        idx = np.concatenate([
            RNG.choice(pos, size=len(pos), replace=True),
            RNG.choice(neg, size=len(neg), replace=True),
        ])
        deltas[i] = roc_auc_score(y[idx], s_hi[idx]) - roc_auc_score(y[idx], s_lo[idx])
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    point = roc_auc_score(y, s_hi) - roc_auc_score(y, s_lo)
    return float(point), float(lo), float(hi)


def load_available(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    out = {}
    for cond, path in paths.items():
        if path.exists():
            df = load_jsonl(path)
            if len(df):
                out[cond] = df
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()

    auc_rows: list[dict] = []
    summary_rows: list[dict] = []
    assoc_rows: list[dict] = []

    for family, paths in discover_families().items():
        available = load_available(paths)
        if "image-only" not in available or "image-text" not in available:
            if available:
                print(f"skip {family}: need image-only + image-text (have {list(available)})")
            continue

        # per-condition AUC + association
        for cond in CONDITIONS:
            if cond not in available:
                continue
            df = available[cond]
            y = df["label"].astype(int).to_numpy()
            s = df["confidence"].astype(float).to_numpy()
            auc, lo, hi = bootstrap_ci(y, s)
            auc_rows.append({"family": family, "condition": cond, "n": len(df),
                             "auc": auc, "ci_lo": lo, "ci_hi": hi})
            m = df.merge(meta, on="aid", how="left")
            rho_d, _, n_d = safe_spearman(m["confidence"], pd.to_numeric(m["sct_long_dia"], errors="coerce"))
            rho_b, _, n_b = safe_spearman(m["confidence"], m["brock_prob_correct"])
            assoc_rows.append({"family": family, "condition": cond,
                               "spearman_diameter": rho_d, "n_diameter": n_d,
                               "spearman_brock": rho_b, "n_brock": n_b})

        # headline: metadata lift (image+text vs image-only), paired on shared aids
        merged = available["image-text"].merge(available["image-only"], on="aid", suffixes=("_hi", "_lo"))
        assert (merged["label_hi"] == merged["label_lo"]).all()
        y = merged["label_hi"].astype(int).to_numpy()
        s_hi = merged["confidence_hi"].astype(float).to_numpy()
        s_lo = merged["confidence_lo"].astype(float).to_numpy()
        auc_hi, auc_lo, p_lift = delong_p(s_hi, s_lo, y)
        d_point, d_lo, d_hi = delta_auc_ci(y, s_hi, s_lo)

        row = {
            "family": family,
            "n_paired": len(merged),
            "auc_image_only": auc_lo,
            "auc_image_text": auc_hi,
            "metadata_lift": d_point,
            "lift_ci_lo": d_lo,
            "lift_ci_hi": d_hi,
            "lift_delong_p": p_lift,
        }
        # text-only recovery (optional)
        if "text-only" in available:
            mt = available["image-text"].merge(available["text-only"], on="aid", suffixes=("_it", "_to"))
            yt = mt["label_it"].astype(int).to_numpy()
            auc_it, auc_to, p_to = delong_p(
                mt["confidence_it"].astype(float).to_numpy(),
                mt["confidence_to"].astype(float).to_numpy(),
                yt,
            )
            row["auc_text_only"] = auc_to
            row["text_only_vs_imgtext_delong_p"] = p_to
            row["recovery_frac"] = (
                (auc_to - row["auc_image_only"]) / (auc_it - row["auc_image_only"])
                if (auc_it - row["auc_image_only"]) > 1e-9 else np.nan
            )
        summary_rows.append(row)

    auc_df = pd.DataFrame(auc_rows)
    summary = pd.DataFrame(summary_rows)
    assoc = pd.DataFrame(assoc_rows)
    auc_df.to_csv(OUT_DIR / "crossfamily_auc_by_condition.csv", index=False)
    summary.to_csv(OUT_DIR / "crossfamily_summary.csv", index=False)
    assoc.to_csv(OUT_DIR / "crossfamily_association.csv", index=False)

    # markdown summary
    lines = ["# Cross-family metadata-reliance summary", "",
             "## Metadata lift = AUC(image+text) - AUC(image-only)", ""]
    lines.append(to_md(summary, "No families ready yet."))
    lines += ["", "## Per-condition AUC", "", to_md(auc_df),
              "", "## Score-vs-structured-predictor association (Spearman)", "", to_md(assoc), ""]
    (OUT_DIR / "crossfamily_summary.md").write_text("\n".join(lines))

    # figure: metadata lift per family with bootstrap CI
    if not summary.empty:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            s = summary.iloc[::-1].reset_index(drop=True)
            yloc = np.arange(len(s))
            xerr = np.vstack([s["metadata_lift"] - s["lift_ci_lo"], s["lift_ci_hi"] - s["metadata_lift"]])
            fig, ax = plt.subplots(figsize=(6.4, 0.6 * len(s) + 1.2))
            ax.axvline(0.0, color="0.6", lw=1, ls="--")
            ax.errorbar(s["metadata_lift"], yloc, xerr=xerr, fmt="o", color="#b2182b",
                        capsize=3, lw=1.5, ms=6)
            ax.set_yticks(yloc)
            ax.set_yticklabels(s["family"])
            ax.set_xlabel(r"Metadata lift  $\Delta$AUC = AUC(image+text) $-$ AUC(image-only)")
            ax.set_title("Structured-text lift in apparent CT reading, by VLM family")
            for yi, (lift, p) in enumerate(zip(s["metadata_lift"], s["lift_delong_p"])):
                ax.annotate(f"+{lift:.3f} (P={p:.3g})", (lift, yi), textcoords="offset points",
                            xytext=(8, 0), va="center", fontsize=8)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(OUT_DIR / f"crossfamily_delta_auc.{ext}", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {(OUT_DIR / 'crossfamily_delta_auc.pdf').relative_to(ROOT)}")
        except Exception as exc:  # noqa: BLE001
            print(f"(figure skipped: {exc})")

    print(summary.to_string(index=False) if not summary.empty else "No families ready.")
    print(f"\nWrote {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
