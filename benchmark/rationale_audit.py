"""Rationale audit for the metadata-conditioned claim (Results 2.2 / Supp Note).

For every F3 case (primary run), it checks whether the model's
one-sentence rationale reproduces, verbatim, the structured nodule
descriptors that were supplied in the clinical text (the long-axis diameter
in mm and the margin descriptor). This is the supporting evidence that the
F3 gain is a shortcut to the supplied structured descriptors rather than new
visual perception. Numbers and sample cases are generated here; the
Supplementary "Rationale audit" note is verified against this output.

No inference is run — reads stored predictions and the clinical-text table.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from benchmark.paths import CLINICAL_TEXT_CSV, VLM_RESULTS_DIR

AUDIT_RUN = "luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl"

MARGIN_WORDS = ["spiculated", "irregular", "lobulated", "smooth", "well-defined"]


def load() -> pd.DataFrame:
    p = VLM_RESULTS_DIR / AUDIT_RUN
    pred = pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
    pred["confidence"] = pd.to_numeric(pred["confidence"], errors="coerce")
    pred = pred[pred["confidence"].between(0, 1)].drop_duplicates("aid", keep="last")

    txt = pd.read_csv(CLINICAL_TEXT_CSV)
    txt["aid"] = txt["AnnotationID"].astype(str).str.split("_", n=1).str[1]
    txt = txt.drop_duplicates("aid", keep="last")[["aid", "clinical_text"]]
    return pred.merge(txt, on="aid", how="inner")


def parse_clinical(t: str) -> dict:
    t = str(t)
    dia = re.search(r"long-axis diameter of\s+([0-9]+(?:\.[0-9]+)?)\s*mm", t, re.I)
    marg = re.search(r"Exhibiting\s+(.+?)\s+margins", t, re.I)
    return {
        "dia_mm": float(dia.group(1)) if dia else None,
        "margin": marg.group(1).strip().lower() if marg else None,
    }


def quotes_size(reason: str, dia_mm: float | None) -> bool:
    if dia_mm is None:
        return False
    r = str(reason).lower()
    iv = int(round(dia_mm))
    # model writes e.g. "22mm", "22 mm", "22-mm", "22.0 mm"
    pats = [rf"\b{iv}\s*-?\s*mm", rf"\b{dia_mm:g}\s*-?\s*mm"]
    return any(re.search(p, r) for p in pats)


def quotes_margin(reason: str, margin: str | None) -> bool:
    if not margin:
        return False
    r = str(reason).lower()
    return any(w in margin and w in r for w in MARGIN_WORDS)


def main() -> None:
    df = load()
    df["clin"] = df["clinical_text"].apply(parse_clinical)
    df["q_size"] = df.apply(lambda x: quotes_size(x["reasoning"], x["clin"]["dia_mm"]), axis=1)
    df["q_margin"] = df.apply(lambda x: quotes_margin(x["reasoning"], x["clin"]["margin"]), axis=1)
    df["q_any"] = df["q_size"] | df["q_margin"]

    n = len(df)
    print(f"Audit run: {AUDIT_RUN}   matched F3 cases with clinical text: {n}")
    for col, lbl in [("q_size", "verbatim size (diameter in mm)"),
                     ("q_margin", "verbatim margin descriptor"),
                     ("q_any", "verbatim size OR margin")]:
        c = int(df[col].sum())
        print(f"  {lbl:38s} {c:4d}/{n}  ({c / n * 100:.1f}%)")

    has_dia = df["clin"].apply(lambda c: c["dia_mm"] is not None)
    sd = df[has_dia]
    cs = int(sd["q_size"].sum())
    print(f"  size quoted | diameter present        {cs:4d}/{len(sd)}  "
          f"({cs / len(sd) * 100:.1f}%)")

    print("\nRepresentative cases (verbatim size + margin echoed from clinical text):")
    ex = df[df["q_size"] & df["q_margin"]].head(4)
    for _, r in ex.iterrows():
        print(f"\n  AnnotationID {r['aid']}  (label={int(r['label'])}, "
              f"confidence={float(r['confidence']):.2f})")
        print(f"    clinical text : {r['clinical_text']}")
        print(f"    model rationale: {r['reasoning']}")


if __name__ == "__main__":
    main()
