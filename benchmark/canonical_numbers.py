"""Single source of truth for the manuscript's core point AUCs.

Conflict (1): AUCs were hand-copied into the text/tables/figures with
inconsistent rounding (for example, one ablation row appeared as both 0.699 and 0.700). This script
recomputes the core ablation, cross-generation, DL, Brock, and LNDb point AUCs
from the *one* per-sample prediction file that backs each value and prints them
round-half-up to 3 decimals. Text, tables and figure annotations for these
analyses must all match this output.

It recomputes only point AUCs (and the few deltas the text cites) from stored
predictions — no inference is run and reported bootstrap CIs are left as-is
(they come from seeded bootstraps and are unchanged by this reconciliation).

Run:  .venv/bin/python -m benchmark.canonical_numbers
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from benchmark.figure_style import rounded_3
from benchmark.paths import LUNA25_DL_PRED_DIR, VLM_RESULTS_DIR

RES = VLM_RESULTS_DIR
OUT_CSV = RES / "canonical_numbers.csv"

# ── per-sample source file for every reported AUC ──────────────────────────
ABLATION = {  # Gemini 3 Flash Preview, LUNA25 test n=917; filenames use the public Z/F tags.
    "Z1 zero-shot minimal": "luna25_ablation/z1_gemini3flash_zeroshot_minimal.jsonl",
    "Z2 zero-shot rich":    "luna25_ablation/z2_gemini3flash_zeroshot_rich.jsonl",
    "Z3 zero-shot rich+clinical": "luna25_ablation/z3_gemini3flash_zeroshot_rich_metadata.jsonl",
    "F1 20-shot minimal":   "luna25_ablation/f1_gemini3flash_20shot_minimal.jsonl",
    "F2 20-shot rich":      "luna25_ablation/f2_gemini3flash_20shot_rich.jsonl",
    "F3 20-shot rich+clinical": "luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl",
    "F3A mean-of-5":        "f3a_mean_of5_predictions.csv",
}
GENERATIONS = {  # condition F3, LUNA25 test n=917; filenames use the public F3 tags.
    "Gemini 2.5 Flash":               "luna25_model_comparison/f3_gemini25flash.jsonl",
    "Gemini 2.5 Pro":                 "luna25_model_comparison/f3_gemini25pro.jsonl",
    "Gemini 3 Flash Preview":         "luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl",
    "Gemini 3.1 Pro Preview":         "luna25_model_comparison/f3_gemini31pro.jsonl",
    "MedGemma 1.5-4B":                "luna25_model_comparison/f3_medgemma15_4b.jsonl",
    "Gemma 4 31B":                    "luna25_model_comparison/f3_gemma4_31b.jsonl",
    "Gemma 4 26B A4B":                "luna25_model_comparison/f3_gemma4_26b_a4b.jsonl",
}
DL_LUNA25 = {
    "STU-Net":         "stunet_base_warmup_test_preds.csv",
    "EfficientNet-B0": "efficientnet_b0_baseline_test_preds.csv",
    "ResNet-18":       "resnet18_baseline_test_preds.csv",
    "DenseNet-121":    "densenet121_baseline_test_preds.csv",
    "ResNet-50":       "resnet50_baseline_test_preds.csv",
    "Swin-UNETR":      "swin_unetr_final_gpu_test_preds.csv",
    "ViT-Base":        "vit_baseline_test_preds.csv",
}


def _load_vlm(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path).rename(columns={"AnnotationID": "aid"})
    else:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(rows)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df[df["confidence"].between(0.0, 1.0)]
    return df.drop_duplicates("aid", keep="last")[["aid", "label", "confidence"]].reset_index(drop=True)


def _load_dl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["split"] == "test"].copy()
    return df.rename(columns={"AnnotationID": "aid", "pred_prob": "confidence"})[["aid", "label", "confidence"]]


def _auc(df: pd.DataFrame) -> float:
    return roc_auc_score(df["label"].astype(int), df["confidence"].astype(float))


def main() -> None:
    rows = []

    def record(group, name, df):
        a = _auc(df)
        rows.append({"group": group, "name": name, "n": len(df),
                     "auc_exact": a, "auc": rounded_3(a)})
        return a

    for name, fn in ABLATION.items():
        record("ablation", name, _load_vlm(RES / fn))
    for name, fn in GENERATIONS.items():
        record("generation", name, _load_vlm(RES / fn))
    for name, fn in DL_LUNA25.items():
        record("dl_luna25", name, _load_dl(LUNA25_DL_PRED_DIR / fn))

    # Brock complete-metadata subset (701) + matched primary single-run F3.
    brock = pd.read_csv(RES / "brock_pancan_predictions.csv")
    f3 = _load_vlm(RES / "luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl")
    f3["AnnotationID"] = f3["aid"]
    sub = brock.merge(f3[["AnnotationID", "confidence"]], on="AnnotationID", how="inner")
    rows.append({"group": "brock", "name": "Brock (701)", "n": len(sub),
                 "auc_exact": roc_auc_score(sub["label"], sub["brock_prob_correct"]),
                 "auc": rounded_3(roc_auc_score(sub["label"], sub["brock_prob_correct"]))})
    rows.append({"group": "brock", "name": "Gemini F3 (primary run) on Brock-701", "n": len(sub),
                 "auc_exact": roc_auc_score(sub["label"], sub["confidence"]),
                 "auc": rounded_3(roc_auc_score(sub["label"], sub["confidence"]))})

    # LNDb external (row-level) — from the aligned 814-row metrics sheet.
    # Only the *external* AUC is taken here; the matched internal AUC for the
    # Gemini visual-only conditions is the LUNA25 Z3/Z2 value already computed
    # above from the per-sample files (the metrics-sheet internal_auc field is
    # a pre-rounded 4-dp mirror that would drift, e.g. F2 0.6995 -> 0.700 vs
    # the true 0.699). Reporting it once, from one source, keeps this tool
    # self-consistent with the manuscript.
    lndb = pd.read_csv(RES / "lndb_external_metrics.csv")
    for _, r in lndb.iterrows():
        rows.append({"group": "lndb", "name": f"{r['model']} (ext)", "n": int(r["n_rows"]),
                     "auc_exact": float(r["external_auc"]),
                     "auc": rounded_3(r["external_auc"])})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    # Key deltas the text cites (computed from the same rounded values).
    by = {r["name"]: r["auc"] for r in rows}
    deltas = {
        "Z1->Z2": float(by["Z2 zero-shot rich"]) - float(by["Z1 zero-shot minimal"]),
        "Z1->F2": float(by["F2 20-shot rich"]) - float(by["Z1 zero-shot minimal"]),
        "Z1->F1": float(by["F1 20-shot minimal"]) - float(by["Z1 zero-shot minimal"]),
        "F2->F3": float(by["F3 20-shot rich+clinical"]) - float(by["F2 20-shot rich"]),
        "Z3 vs F3": float(by["F3 20-shot rich+clinical"]) - float(by["Z3 zero-shot rich+clinical"]),
        "F3 - Brock (701)": float(by["Gemini F3 (primary run) on Brock-701"]) - float(by["Brock (701)"]),
    }

    print(f"→ {OUT_CSV}\n")
    for g in ["ablation", "generation", "dl_luna25", "brock", "lndb"]:
        print(f"[{g}]")
        for r in rows:
            if r["group"] == g:
                print(f"  {r['name']:38s} n={r['n']:>4}  AUC={r['auc']}  (exact {r['auc_exact']:.5f})")
        print()
    print("[deltas — round-half-up of rounded values]")
    for k, v in deltas.items():
        print(f"  {k:32s} Δ={v:+.3f}")


if __name__ == "__main__":
    main()
