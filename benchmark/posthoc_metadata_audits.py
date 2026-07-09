"""Post-hoc no-inference analyses of stored predictions and metadata.

Sections:
  1. Brock recompute & verification (beta_type direction, 701-vs-620, regen if swapped)
  2. Patient-level cluster bootstrap (F3 vs STU-Net, F3 vs Brock)
  3. LNDb finding-level AUC (814 rows -> 439 findings)
  4. Classical metadata-only baseline (LR / GBM, train->test) with race on/off
  5. Recalibrated decision-curve / calibration sensitivity (Platt + isotonic)

No Gemini/VLM inference is performed; everything reads existing files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    from benchmark.posthoc_metrics import (
        load_luna25_predictions,
        cluster_bootstrap_ci,
        stratified_bootstrap_ci,
        net_benefit,
    )
    from benchmark.f3_run_averaged import delong_p_value
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmark.posthoc_metrics import (  # noqa: E402
        load_luna25_predictions,
        cluster_bootstrap_ci,
        stratified_bootstrap_ci,
        net_benefit,
    )
    from benchmark.f3_run_averaged import delong_p_value  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VLM = ROOT / "results" / "vlm"
META = ROOT / "data" / "metadata" / "luna25_clinical_metadata.csv"
SPLIT = ROOT / "data" / "metadata" / "patient_split.json"
DL_LUNA = ROOT / "data" / "predictions" / "luna25_dl" / "files"
LNDB_PRED = VLM / "lndb_external_predictions_814.csv"

RNG = np.random.default_rng(20260515)
NB = 2000


def pid(aid: str) -> int:
    return int(str(aid).split("_")[0])


def auc(y, s):
    return roc_auc_score(y, s)


def paired_cluster_bootstrap(df, col_a, col_b, cluster_col="pid", n_boot=NB):
    """Return AUC_a, AUC_b, diff, 95% CI of diff, two-sided bootstrap p, per-model cluster CIs."""
    y = df["label"].to_numpy()
    a = df[col_a].to_numpy()
    b = df[col_b].to_numpy()
    cl = df[cluster_col].to_numpy()
    clusters = pd.unique(cl)
    idx_by = {c: np.where(cl == c)[0] for c in clusters}
    da = np.empty(n_boot)
    aa = np.empty(n_boot)
    bb = np.empty(n_boot)
    for i in range(n_boot):
        samp = RNG.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([idx_by[c] for c in samp])
        yy = y[idx]
        if yy.min() == yy.max():
            da[i] = aa[i] = bb[i] = np.nan
            continue
        Aa = auc(yy, a[idx])
        Bb = auc(yy, b[idx])
        aa[i], bb[i], da[i] = Aa, Bb, Aa - Bb
    lo, hi = np.nanpercentile(da, [2.5, 97.5])
    frac_neg = np.nanmean(da < 0)
    frac_pos = np.nanmean(da > 0)
    p = 2 * min(frac_neg, frac_pos)
    return {
        "auc_a": auc(y, a),
        "auc_b": auc(y, b),
        "diff": auc(y, a) - auc(y, b),
        "diff_ci": (lo, hi),
        "boot_p": p,
        "a_ci": tuple(np.nanpercentile(aa, [2.5, 97.5])),
        "b_ci": tuple(np.nanpercentile(bb, [2.5, 97.5])),
        "n_rows": len(df),
        "n_clusters": len(clusters),
    }


# ======================================================================
# Section 1 — Brock recompute & verification
# ======================================================================
def brock_eta(row, beta_type_map, fam=0, emph=0, ncount=1):
    age = row["age"]
    diam = row["diam"]
    if not np.isfinite(age) or not np.isfinite(diam) or diam <= 0:
        return np.nan
    eta = (-6.7892
           + 0.0287 * (age - 62.0)
           + 0.6011 * row["female"]
           + 0.2961 * fam
           + 0.2953 * emph
           - 5.3854 * ((diam / 10.0) ** -0.5 - 1.58113)
           + beta_type_map.get(row["att"], 0.0)
           - 0.0824 * ncount
           + 0.7729 * row["spic"]
           + 0.6581 * row["upper"])
    return eta


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def section1_brock():
    print("\n" + "=" * 72)
    print("SECTION 1 — Brock recompute & verification")
    print("=" * 72)
    m = pd.read_csv(META, low_memory=False)
    split = json.load(open(SPLIT))
    test_pids = set(int(p) for p in split["test"])

    m = m[m["AnnotationID"].notna()].copy()
    m["pid"] = m["AnnotationID"].map(pid)
    t = m[m["pid"].isin(test_pids)].drop_duplicates("AnnotationID").copy()
    print(f"Test annotations in metadata: {len(t)}  (unique patients {t.pid.nunique()})")

    t["age"] = pd.to_numeric(t["Age_at_StudyDate"], errors="coerce")
    t["female"] = (t["Gender"].astype(str).str.lower().str.startswith("f")).astype(int)
    t["diam"] = pd.to_numeric(t["sct_long_dia"], errors="coerce")
    t["att"] = pd.to_numeric(t["sct_pre_att"], errors="coerce")  # 1=Solid 2=Part-solid 3=GGN
    t["spic"] = (pd.to_numeric(t["sct_margins"], errors="coerce") == 3).astype(int)
    t["upper"] = pd.to_numeric(t["sct_epi_loc"], errors="coerce").isin([1, 4, 5]).astype(int)

    BETA_CORRECT = {1.0: 0.0, 2.0: 0.3770, 3.0: -0.1276}   # McWilliams: part-solid +0.377, GGN -0.1276
    BETA_SWAPPED = {1.0: 0.0, 2.0: -0.1276, 3.0: 0.3770}   # sensitivity coding check

    t["eta_correct"] = t.apply(lambda r: brock_eta(r, BETA_CORRECT), axis=1)
    t["eta_swapped"] = t.apply(lambda r: brock_eta(r, BETA_SWAPPED), axis=1)
    t["p_correct"] = sigmoid(t["eta_correct"])
    t["p_swapped"] = sigmoid(t["eta_swapped"])

    n_complete = t["p_correct"].notna().sum()
    n_diam = t["diam"].notna().sum()
    n_att123 = t["att"].isin([1, 2, 3]).sum()
    print(f"Test annotations with non-missing long-axis diameter (sct_long_dia): {n_diam}")
    print(f"Test annotations with Brock-computable inputs (age+diam present):    {n_complete}")
    print(f"Test annotations with att in {{1,2,3}} (solid/part-solid/GGN):        {n_att123}")

    # Full Brock-computable test set (not just stored 701)
    full = t.dropna(subset=["p_correct"]).copy()
    yf = full["label"].astype(int).to_numpy()
    print(f"\nAUC on FULL Brock-computable test set (n={len(full)}, prev {yf.mean():.3f}):")
    print(f"  CORRECT beta_type      AUC = {auc(yf, full['p_correct']):.4f}")
    print(f"  SWAPPED beta_type      AUC = {auc(yf, full['p_swapped']):.4f}")

    out = t[["AnnotationID", "label", "p_correct", "p_swapped"]].dropna(subset=["p_correct"])
    out = out.rename(columns={"p_correct": "brock_prob_correct",
                              "p_swapped": "brock_prob_swapped"})
    out.to_csv(VLM / "brock_pancan_predictions.csv", index=False)
    print(f"\nWrote {VLM/'brock_pancan_predictions.csv'} ({len(out)} rows)")
    return out


# ======================================================================
# Section 2 — Patient-level cluster bootstrap
# ======================================================================
def section2_cluster(brock_recomp):
    print("\n" + "=" * 72)
    print("SECTION 2 — Patient-level cluster bootstrap (318 patients)")
    print("=" * 72)
    preds = load_luna25_predictions()
    f3 = preds["Gemini 3 Flash Preview (F3 primary run)"].rename(columns={"aid": "AnnotationID"})
    stu = preds["STU-Net"].rename(columns={"aid": "AnnotationID"})
    eff = preds["EfficientNet-B0"].rename(columns={"aid": "AnnotationID"})

    base = f3[["AnnotationID", "label"]].copy()
    base["pid"] = base["AnnotationID"].map(pid)
    base = base.merge(f3[["AnnotationID", "score"]].rename(columns={"score": "f3"}), on="AnnotationID")
    base = base.merge(stu[["AnnotationID", "score"]].rename(columns={"score": "stu"}), on="AnnotationID")
    base = base.merge(eff[["AnnotationID", "score"]].rename(columns={"score": "eff"}), on="AnnotationID")
    print(f"Full LUNA25 test: {len(base)} annotations, {base.pid.nunique()} patients, "
          f"prev {base.label.mean():.3f}")

    for a, b, nm in [("f3", "stu", "F3 vs STU-Net"), ("f3", "eff", "F3 vs EfficientNet-B0")]:
        r = paired_cluster_bootstrap(base, a, b)
        da, db = delong_p_value(base["label"].to_numpy(),
                                base[a].to_numpy(), base[b].to_numpy())[:3] if False else (None, None)
        _, _, dp = delong_p_value(base["label"].to_numpy(), base[a].to_numpy(), base[b].to_numpy())
        print(f"\n{nm}")
        print(f"  AUC {a}={r['auc_a']:.4f} (patient-cluster 95% CI {r['a_ci'][0]:.3f}-{r['a_ci'][1]:.3f})")
        print(f"  AUC {b}={r['auc_b']:.4f} (patient-cluster 95% CI {r['b_ci'][0]:.3f}-{r['b_ci'][1]:.3f})")
        print(f"  ΔAUC={r['diff']:+.4f}  patient-cluster 95% CI [{r['diff_ci'][0]:+.4f},{r['diff_ci'][1]:+.4f}]"
              f"  bootstrap p={r['boot_p']:.4f}  | annotation-level DeLong p={dp:.2e}")

    # F3 vs Brock on the recomputed-correct Brock subset
    bk = brock_recomp.merge(f3[["AnnotationID", "score"]].rename(columns={"score": "f3"}),
                            on="AnnotationID", how="inner")
    bk["pid"] = bk["AnnotationID"].map(pid)
    bk = bk.rename(columns={"brock_prob_correct": "brock"})
    print(f"\nF3 vs Brock (recomputed CORRECT beta_type), n={len(bk)}, {bk.pid.nunique()} patients, "
          f"prev {bk.label.mean():.3f}")
    r = paired_cluster_bootstrap(bk, "f3", "brock")
    _, _, dp = delong_p_value(bk["label"].to_numpy(), bk["f3"].to_numpy(), bk["brock"].to_numpy())
    print(f"  AUC F3={r['auc_a']:.4f} (95% CI {r['a_ci'][0]:.3f}-{r['a_ci'][1]:.3f})")
    print(f"  AUC Brock={r['auc_b']:.4f} (95% CI {r['b_ci'][0]:.3f}-{r['b_ci'][1]:.3f})")
    print(f"  ΔAUC={r['diff']:+.4f}  patient-cluster 95% CI [{r['diff_ci'][0]:+.4f},{r['diff_ci'][1]:+.4f}]"
          f"  bootstrap p={r['boot_p']:.4f}  | annotation-level DeLong p={dp:.3f}")


# ======================================================================
# Section 3 — LNDb finding-level AUC
# ======================================================================
def section3_lndb():
    print("\n" + "=" * 72)
    print("SECTION 3 — LNDb finding-level AUC (814 rows -> 439 findings)")
    print("=" * 72)
    df = pd.read_csv(LNDB_PRED)
    score_cols = [c for c in df.columns if c.endswith("_score")]
    # label consistency within a finding
    grp = df.groupby("FindingID")
    consist = grp["label"].nunique()
    print(f"Rows={len(df)}  findings={df.FindingID.nunique()}  "
          f"findings with mixed labels={(consist > 1).sum()}")
    fin = grp.agg({"label": "max", **{c: "mean" for c in score_cols}}).reset_index()
    yr = df["label"].to_numpy()
    yf = fin["label"].to_numpy()
    print(f"Finding-level prevalence {yf.mean():.3f}  (row-level {yr.mean():.3f})\n")
    print(f"{'model':<34}{'row AUC':>9}{'find AUC':>10}   finding 95% CI (cluster by FindingID)")
    for c in score_cols:
        ra = auc(yr, df[c].to_numpy())
        fa = auc(yf, fin[c].to_numpy())
        lo, hi = cluster_bootstrap_ci(df["label"].to_numpy(), df[c].to_numpy(),
                                      df["FindingID"].to_numpy(), roc_auc_score)
        nm = c.replace("_score", "")
        print(f"{nm:<34}{ra:>9.4f}{fa:>10.4f}   [{lo:.3f}, {hi:.3f}]")


# ======================================================================
# Section 4 — Classical metadata-only baseline
# ======================================================================
def section4_metadata_only():
    print("\n" + "=" * 72)
    print("SECTION 4 — Classical metadata-only baseline (train -> test)")
    print("=" * 72)
    m = pd.read_csv(META, low_memory=False)
    split = json.load(open(SPLIT))
    tr_pids = set(int(p) for p in split["train"])
    te_pids = set(int(p) for p in split["test"])
    m = m[m["AnnotationID"].notna()].drop_duplicates("AnnotationID").copy()
    m["pid"] = m["AnnotationID"].map(pid)
    m["age"] = pd.to_numeric(m["Age_at_StudyDate"], errors="coerce")
    m["female"] = (m["Gender"].astype(str).str.lower().str.startswith("f")).astype(int)
    m["current_smoker"] = pd.to_numeric(m["cigsmok"], errors="coerce")
    m["long_dia"] = pd.to_numeric(m["sct_long_dia"], errors="coerce")
    m["perp_dia"] = pd.to_numeric(m["sct_perp_dia"], errors="coerce")
    m["att"] = pd.to_numeric(m["sct_pre_att"], errors="coerce")
    m["spiculation"] = (pd.to_numeric(m["sct_margins"], errors="coerce") == 3).astype(float)
    m["upper_lobe"] = pd.to_numeric(m["sct_epi_loc"], errors="coerce").isin([1, 4, 5]).astype(float)
    m["att_partsolid"] = (m["att"] == 2).astype(float)
    m["att_ggn"] = (m["att"] == 3).astype(float)
    m["race_white"] = (pd.to_numeric(m["race"], errors="coerce") == 1).astype(float)
    m["race_known"] = pd.to_numeric(m["race"], errors="coerce").isin([1, 2, 3, 4, 5, 6, 7]).astype(float)

    base_feats = ["age", "female", "current_smoker", "long_dia", "perp_dia",
                  "spiculation", "upper_lobe", "att_partsolid", "att_ggn"]
    race_feats = base_feats + ["race_white", "race_known"]

    tr = m[m["pid"].isin(tr_pids)].copy()
    te = m[m["pid"].isin(te_pids)].copy()
    # align test to the 917 evaluated AnnotationIDs
    f3 = load_luna25_predictions()["Gemini 3 Flash Preview (F3 primary run)"]
    te = te[te["AnnotationID"].isin(set(f3["aid"]))].copy()
    print(f"train annotations={len(tr)} (patients {tr.pid.nunique()}), "
          f"test annotations={len(te)} (patients {te.pid.nunique()}), "
          f"test prev={te['label'].mean():.3f}")

    def run(feats, tag):
        Xtr, ytr = tr[feats].to_numpy(), tr["label"].astype(int).to_numpy()
        Xte, yte = te[feats].to_numpy(), te["label"].astype(int).to_numpy()
        for name, clf in [
            ("LogReg", Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("sc", StandardScaler()),
                                 ("lr", LogisticRegression(max_iter=2000,
                                                           class_weight="balanced"))])),
            ("GBM", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("gb", GradientBoostingClassifier(random_state=0))])),
        ]:
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            a = auc(yte, p)
            lo, hi = stratified_bootstrap_ci(yte, p, roc_auc_score)
            ap = average_precision_score(yte, p)
            print(f"  [{tag:<14}] {name:<7} test AUC={a:.4f} (95% CI {lo:.3f}-{hi:.3f})  PR-AUC={ap:.3f}")

    print("\nReference: primary F3 AUC=0.739 ; F3A mean-of-5 AUC=0.747 ; "
          "zero-shot clinical-text 0.730 ; "
          "best DL (STU-Net) 0.872")
    run(base_feats, "no race")
    run(race_feats, "with race")


# ======================================================================
# Section 5 — Recalibration sensitivity (Platt + isotonic)
# ======================================================================
def ece(y, p, bins=10):
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def section5_recalibration():
    print("\n" + "=" * 72)
    print("SECTION 5 — Recalibration sensitivity (Platt / isotonic)")
    print("=" * 72)
    thr = [0.05, 0.10, 0.15, 0.20]

    # DL: fit on official validation split, apply to test
    dl = {
        "STU-Net": ("stunet_base_warmup_val_preds.csv", "stunet_base_warmup_test_preds.csv"),
        "EfficientNet-B0": ("efficientnet_b0_baseline_val_preds.csv", "efficientnet_b0_baseline_test_preds.csv"),
    }
    for nm, (vf, tf) in dl.items():
        v = pd.read_csv(DL_LUNA / vf)
        s = pd.read_csv(DL_LUNA / tf)
        yv, pv = v["label"].to_numpy(), v["pred_prob"].to_numpy()
        yt, pt = s["label"].to_numpy(), s["pred_prob"].to_numpy()
        platt = LogisticRegression(max_iter=1000).fit(pv.reshape(-1, 1), yv)
        p_platt = platt.predict_proba(pt.reshape(-1, 1))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(pv, yv)
        p_iso = iso.transform(pt)
        print(f"\n{nm} (val->test recalibration)")
        for tag, p in [("raw", pt), ("Platt", p_platt), ("isotonic", p_iso)]:
            nbs = "  ".join(f"NB{int(t*100)}={net_benefit(yt,p,t):+.4f}" for t in thr)
            print(f"  {tag:<9} AUC={auc(yt,p):.4f} Brier={brier_score_loss(yt,p):.4f} "
                  f"ECE={ece(yt,p):.4f} | {nbs}")

    # VLM F3: no val split -> 5-fold out-of-fold isotonic/Platt as in-sample sensitivity
    f3 = load_luna25_predictions()["Gemini 3 Flash Preview (F3 primary run)"]
    y = f3["label"].astype(int).to_numpy()
    p = f3["score"].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    p_iso = np.zeros_like(p)
    p_pl = np.zeros_like(p)
    for tr_i, te_i in skf.split(p, y):
        iso = IsotonicRegression(out_of_bounds="clip").fit(p[tr_i], y[tr_i])
        p_iso[te_i] = iso.transform(p[te_i])
        pl = LogisticRegression(max_iter=1000).fit(p[tr_i].reshape(-1, 1), y[tr_i])
        p_pl[te_i] = pl.predict_proba(p[te_i].reshape(-1, 1))[:, 1]
    print("\nGemini F3 primary run (5-fold out-of-fold in-sample recalibration; sensitivity only)")
    for tag, pp in [("raw", p), ("Platt-cv", p_pl), ("isotonic-cv", p_iso)]:
        nbs = "  ".join(f"NB{int(t*100)}={net_benefit(y,pp,t):+.4f}" for t in thr)
        print(f"  {tag:<11} AUC={auc(y,pp):.4f} Brier={brier_score_loss(y,pp):.4f} "
              f"ECE={ece(y,pp):.4f} | {nbs}")


if __name__ == "__main__":
    br = section1_brock()
    section2_cluster(br)
    section3_lndb()
    section4_metadata_only()
    section5_recalibration()
    print("\nDONE.")
