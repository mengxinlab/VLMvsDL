"""Failure-mode coding of Gemini F3 discordant cases (Results 2.9 / Supp Note).

The audit is performed on the *representative single run* (00) at the
0.5 threshold, classifying each one-sentence model rationale with a fixed,
case-insensitive substring lexicon of radiological cue families. This script
is the single source of the failure-mode counts; the manuscript text and the
Supplementary "Failure-mode lexicon" note are generated from / verified
against its output (run with --check to compare to the published numbers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from benchmark.paths import VLM_RESULTS_DIR

AUDIT_RUN = "luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl"

# ── Fixed radiological-cue lexicon (case-insensitive substring match) ──────
LEXICON: dict[str, list[str]] = {
    "margin_spiculation": [
        "spicul", "margin", "lobulat", "irregular", "ill-defined",
        "ill defined", "well-defined", "well defined", "smooth", "border",
        "contour", "speculat",
    ],
    "attenuation": [
        "solid", "part-solid", "subsolid", "sub-solid", "ground-glass",
        "ground glass", "ggo", "attenuat", "density", "consistency",
        "soft tissue", "calcif",
    ],
    "history_smoking": [
        "smok", "history",
    ],
    "size": [
        "size", "diameter", "sized", "large",
    ],
}


def _hit(text: str, terms: list[str]) -> bool:
    t = str(text).lower()
    return any(term in t for term in terms)


def load_audit_run() -> pd.DataFrame:
    p = VLM_RESULTS_DIR / AUDIT_RUN
    df = pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df[df["confidence"].between(0, 1)].drop_duplicates("aid", keep="last")
    df["label"] = df["label"].astype(int)
    df["pred"] = (df["confidence"] >= 0.5).astype(int)
    return df


def code(df: pd.DataFrame) -> dict:
    fp = df[(df.pred == 1) & (df.label == 0)]
    fn = df[(df.pred == 0) & (df.label == 1)]

    def fam_counts(sub: pd.DataFrame) -> dict[str, int]:
        return {fam: int(sub["reasoning"].apply(lambda r: _hit(r, terms)).sum())
                for fam, terms in LEXICON.items()}

    return {
        "n_total": len(df),
        "n_discordant": int((df.pred != df.label).sum()),
        "n_fp": len(fp),
        "n_fn": len(fn),
        "fp_conf_median": round(float(fp.confidence.median()), 2),
        "fp_conf_mean": round(float(fp.confidence.mean()), 2),
        "fn_conf_median": round(float(fn.confidence.median()), 2),
        "fn_conf_mean": round(float(fn.confidence.mean()), 2),
        "fp": fam_counts(fp),
        "fn": fam_counts(fn),
        "fn_history_or_size": int(fn["reasoning"].apply(
            lambda r: _hit(r, LEXICON["history_smoking"]) or _hit(r, LEXICON["size"])
        ).sum()),
    }


def main() -> None:
    r = code(load_audit_run())
    nfp, nfn = r["n_fp"], r["n_fn"]
    print(f"Audit run: {AUDIT_RUN}")
    print(f"n={r['n_total']}  discordant={r['n_discordant']} "
          f"({r['n_discordant'] / r['n_total'] * 100:.1f}%)  "
          f"FP={nfp}  FN={nfn}")
    print(f"FP confidence median={r['fp_conf_median']} mean={r['fp_conf_mean']}; "
          f"FN median={r['fn_conf_median']} mean={r['fn_conf_mean']}")
    print("\nFalse positives (benign called malignant):")
    for fam, c in r["fp"].items():
        print(f"  {fam:20s} {c:3d}/{nfp}  ({c / nfp * 100:.1f}%)")
    print("False negatives (malignant called benign):")
    for fam, c in r["fn"].items():
        print(f"  {fam:20s} {c:3d}/{nfn}  ({c / nfn * 100:.1f}%)")
    print(f"  FN history-or-size   {r['fn_history_or_size']}/{nfn}")

    if "--check" in sys.argv:
        # Published Results 2.9 / Supplementary note targets.
        exp = {"FP": 258, "FN": 22, "fp_margin": 192, "fp_atten": 180,
               "fp_hist": 149, "fp_size_pct": 24.4}
        got = {"FP": nfp, "FN": nfn,
               "fp_margin": r["fp"]["margin_spiculation"],
               "fp_atten": r["fp"]["attenuation"],
               "fp_hist": r["fp"]["history_smoking"],
               "fp_size_pct": round(r["fp"]["size"] / nfp * 100, 1)}
        print("\n[check] expected vs reproduced:")
        ok = True
        for k in exp:
            flag = "OK" if exp[k] == got[k] else "DIFF"
            if flag == "DIFF":
                ok = False
            print(f"  {k:14s} expected={exp[k]:<6} got={got[k]:<6} {flag}")
        print("ALL MATCH" if ok else "MISMATCH — see above")


if __name__ == "__main__":
    main()
