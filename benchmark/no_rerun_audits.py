"""Generate no-rerun audit tables used in the May 2026 manuscript revision.

This script performs no new model inference. It reads released metadata tables
plus stored per-sample predictions and writes the CSV side products used for
Supplementary Tables 11--13.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

try:
    from benchmark.paths import FIGURES_DIR, LUNA25_CLINICAL_METADATA_CSV, PATIENT_SPLIT_JSON, VLM_RESULTS_DIR
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paths import FIGURES_DIR, LUNA25_CLINICAL_METADATA_CSV, PATIENT_SPLIT_JSON, VLM_RESULTS_DIR  # type: ignore


def pid(annotation_id: str) -> int:
    return int(str(annotation_id).split("_")[0])


def load_vlm_jsonl_scores(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df.drop_duplicates("aid", keep="last")[["aid", "label", "confidence"]]


def load_test_metadata() -> pd.DataFrame:
    meta = pd.read_csv(LUNA25_CLINICAL_METADATA_CSV, low_memory=False)
    with open(PATIENT_SPLIT_JSON) as f:
        split = json.load(f)
    test_pids = {int(x) for x in split["test"]}
    meta = meta[meta["AnnotationID"].notna()].copy()
    meta["pid"] = meta["AnnotationID"].map(pid)
    meta = meta[meta["pid"].isin(test_pids)].drop_duplicates("AnnotationID").copy()
    return meta


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def density_label(value: float) -> str:
    mapping = {
        1.0: "Solid",
        2.0: "Part-solid",
        3.0: "GGN",
        4.0: "Mixed/other",
        6.0: "Fat",
        7.0: "Fluid/water",
        9.0: "Not determined",
    }
    if pd.isna(value):
        return "Missing"
    return mapping.get(float(value), f"Other ({value})")


def margin_label(value: float) -> str:
    mapping = {
        1.0: "Smooth",
        2.0: "Lobulated",
        3.0: "Spiculated/irregular",
        9.0: "Not determined",
    }
    if pd.isna(value):
        return "Missing"
    return mapping.get(float(value), f"Other ({value})")


def write_metadata_provenance(test_meta: pd.DataFrame) -> None:
    rows = [
        ("Age_at_StudyDate", "Age", "Released LUNA25 demographic field", "Near-objective"),
        ("Gender", "Sex", "Released LUNA25 demographic field", "Near-objective"),
        ("race", "Race", "Matched NLST participant field", "Self-reported category used verbatim"),
        ("cigsmok", "Smoking status", "Matched NLST screening-history field", "Self-reported/history-derived"),
        ("sct_ab_desc", "Nodule type", "Matched NLST CT-abnormality table", "Reader-derived structured nodule descriptor"),
        ("sct_long_dia", "Long-axis diameter", "Matched NLST CT-abnormality table", "Reader-derived structured CT measurement"),
        ("sct_perp_dia", "Perpendicular diameter", "Matched NLST CT-abnormality table", "Reader-derived structured CT measurement"),
        ("sct_epi_loc", "Lobar location", "Matched NLST CT-abnormality table", "Reader-derived structured CT descriptor"),
        ("sct_margins", "Margin category", "Matched NLST CT-abnormality table", "Reader-derived structured CT descriptor"),
        ("sct_pre_att", "Attenuation/density category", "Matched NLST CT-abnormality table", "Reader-derived structured CT descriptor"),
        ("label", "Binary malignancy label", "LUNA25 benchmark curation", "Curated mixed reference standard"),
    ]
    out = []
    denom = len(test_meta)
    for field, display, source, kind in rows:
        available_n = int(test_meta[field].notna().sum())
        out.append(
            {
                "field": field,
                "display": display,
                "source": source,
                "kind": kind,
                "available_n": available_n,
                "available_pct": pct(available_n, denom),
            }
        )
    pd.DataFrame(out).to_csv(FIGURES_DIR / "metadata_field_provenance.csv", index=False)


def write_brock_subset_profiles(test_meta: pd.DataFrame) -> pd.DataFrame:
    brock_path = VLM_RESULTS_DIR / "brock_pancan_predictions.csv"
    brock = pd.read_csv(brock_path)
    subset = test_meta[test_meta["AnnotationID"].isin(set(brock["AnnotationID"]))].copy()

    def summary_row(name: str, df: pd.DataFrame) -> dict[str, object]:
        n = len(df)
        age = pd.to_numeric(df["Age_at_StudyDate"], errors="coerce")
        smoker = pd.to_numeric(df["cigsmok"], errors="coerce").eq(1)
        male = df["Gender"].astype(str).eq("Male")
        malignant = df["label"].astype(int)
        return {
            "cohort": name,
            "annotations": n,
            "patients": int(df["pid"].nunique()),
            "malignant_n": int(malignant.sum()),
            "malignant_pct": pct(int(malignant.sum()), n),
            "age_mean": round(float(age.mean()), 1),
            "age_sd": round(float(age.std(ddof=1)), 1),
            "male_n": int(male.sum()),
            "male_pct": pct(int(male.sum()), n),
            "current_smoker_n": int(smoker.sum()),
            "current_smoker_pct": pct(int(smoker.sum()), n),
            "long_dia_avail_n": int(df["sct_long_dia"].notna().sum()),
            "long_dia_avail_pct": pct(int(df["sct_long_dia"].notna().sum()), n),
            "margin_avail_n": int(df["sct_margins"].notna().sum()),
            "margin_avail_pct": pct(int(df["sct_margins"].notna().sum()), n),
            "att_avail_n": int(df["sct_pre_att"].notna().sum()),
            "att_avail_pct": pct(int(df["sct_pre_att"].notna().sum()), n),
        }

    profile = pd.DataFrame(
        [
            summary_row("Full LUNA25 test split", test_meta),
            summary_row("Brock-computable subset", subset),
        ]
    )
    profile.to_csv(FIGURES_DIR / "brock_subset_profile.csv", index=False)

    category_rows = []
    for cohort_name, df in [("Full LUNA25 test split", test_meta), ("Brock-computable subset", subset)]:
        for category, count in df["sct_pre_att"].map(density_label).value_counts().items():
            category_rows.append({"cohort": cohort_name, "family": "Density", "category": category, "n": int(count)})
        for category, count in df["sct_margins"].map(margin_label).value_counts().items():
            category_rows.append({"cohort": cohort_name, "family": "Margin", "category": category, "n": int(count)})
    pd.DataFrame(category_rows).to_csv(FIGURES_DIR / "brock_subset_categories.csv", index=False)
    return subset


def write_metadata_dependence_posthoc(brock_subset: pd.DataFrame) -> None:
    brock_path = VLM_RESULTS_DIR / "brock_pancan_predictions.csv"
    brock = pd.read_csv(brock_path)
    brock_col = "brock_prob_correct"

    base = brock_subset[["AnnotationID", "sct_long_dia", "sct_margins"]].copy()
    base = base.merge(brock[["AnnotationID", brock_col]].rename(columns={brock_col: "brock_prob"}), on="AnnotationID", how="inner")
    base["diameter"] = pd.to_numeric(base["sct_long_dia"], errors="coerce")
    base["spic"] = pd.to_numeric(base["sct_margins"], errors="coerce").eq(3)

    model_files = {
        "Z1 (zero-shot, minimal)": "luna25_ablation/z1_gemini3flash_zeroshot_minimal.jsonl",
        "Z3 (zero-shot, rich + clinical)": "luna25_ablation/z3_gemini3flash_zeroshot_rich_metadata.jsonl",
        "F2 (20-shot, rich)": "luna25_ablation/f2_gemini3flash_20shot_rich.jsonl",
        "F3 (20-shot, rich + clinical)": "luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl",
    }

    rows = []
    for model, filename in model_files.items():
        pred = load_vlm_jsonl_scores(VLM_RESULTS_DIR / filename).rename(columns={"aid": "AnnotationID", "confidence": "score"})
        df = base.merge(pred[["AnnotationID", "score"]], on="AnnotationID", how="inner").dropna(subset=["brock_prob", "diameter", "score"])
        rho_brock, p_brock = spearmanr(df["score"], df["brock_prob"])
        rho_diam, p_diam = spearmanr(df["score"], df["diameter"])
        rows.append(
            {
                "model": model,
                "n": len(df),
                "rho_brock": round(float(rho_brock), 3),
                "p_brock": float(p_brock),
                "rho_diam": round(float(rho_diam), 3),
                "p_diam": float(p_diam),
                "mean_spic_yes": round(float(df.loc[df["spic"], "score"].mean()), 3),
                "mean_spic_no": round(float(df.loc[~df["spic"], "score"].mean()), 3),
            }
        )
    pd.DataFrame(rows).to_csv(FIGURES_DIR / "metadata_dependence_posthoc.csv", index=False)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    test_meta = load_test_metadata()
    write_metadata_provenance(test_meta)
    brock_subset = write_brock_subset_profiles(test_meta)
    write_metadata_dependence_posthoc(brock_subset)
    print("Wrote:")
    for name in [
        "metadata_field_provenance.csv",
        "brock_subset_profile.csv",
        "brock_subset_categories.csv",
        "metadata_dependence_posthoc.csv",
    ]:
        print(" -", FIGURES_DIR / name)


if __name__ == "__main__":
    main()
