"""Shared data loading utilities: split + label CSV → train/test annotation rows."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from .paths import CLINICAL_TEXT_CSV, LUNA25_IMAGE_DIR, LUNA25_LABEL_CSV, PATIENT_SPLIT_JSON
except ImportError:  # allow running scripts directly from benchmark/
    from paths import CLINICAL_TEXT_CSV, LUNA25_IMAGE_DIR, LUNA25_LABEL_CSV, PATIENT_SPLIT_JSON

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = LUNA25_LABEL_CSV
DEFAULT_SPLIT = PATIENT_SPLIT_JSON
DEFAULT_IMAGE_DIR = LUNA25_IMAGE_DIR

ID_COL = "AnnotationID"
LABEL_COL = "label"
PID_COL = "PatientID"


def load_split(path: Path = DEFAULT_SPLIT) -> Dict[str, List[int]]:
    with open(path, "r") as f:
        return json.load(f)


def load_label_df(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure expected columns exist
    for col in (ID_COL, LABEL_COL, PID_COL):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} missing from {csv_path}")
    return df


def get_split_rows(
    df: pd.DataFrame,
    split: Dict[str, List[int]],
    name: str,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    require_npy: bool = True,
) -> pd.DataFrame:
    """Return label-CSV rows whose PatientID belongs to split[name].

    If `require_npy`, drop rows whose `<AnnotationID>.npy` is missing on disk.
    """
    pids = set(split[name])
    rows = df[df[PID_COL].isin(pids)].copy()
    if require_npy:
        existing = {p.stem for p in image_dir.glob("*.npy")}
        rows = rows[rows[ID_COL].isin(existing)].copy()
    rows.reset_index(drop=True, inplace=True)
    return rows


def npy_path(annotation_id: str, image_dir: Path = DEFAULT_IMAGE_DIR) -> Path:
    return image_dir / f"{annotation_id}.npy"


def video_path(annotation_id: str, fps: int, root: Path) -> Path:
    return root / f"fps{fps}" / f"{annotation_id}.mp4"


# ---------------------------------------------------------------------------
# Clinical text lookup
# ---------------------------------------------------------------------------
DEFAULT_CLINICAL_CSV = CLINICAL_TEXT_CSV

_clinical_cache: Dict[str, str] = {}
_clinical_loaded: bool = False


def _load_clinical_cache(csv_path: Path = DEFAULT_CLINICAL_CSV) -> None:
    """Load clinical_texts.csv into module-level dict (called once)."""
    global _clinical_cache, _clinical_loaded
    if _clinical_loaded:
        return
    if not csv_path.exists():
        _clinical_loaded = True
        return
    df = pd.read_csv(csv_path, usecols=["AnnotationID", "clinical_text"])
    for _, row in df.iterrows():
        # clinical_texts.csv AnnotationID format: PatientID_AnnotationID
        # Strip leading "PatientID_" prefix to get plain AnnotationID key
        raw = str(row["AnnotationID"])
        key = "_".join(raw.split("_")[1:])
        _clinical_cache[key] = str(row["clinical_text"]) if pd.notna(row["clinical_text"]) else ""
    _clinical_loaded = True


def get_clinical_text(annotation_id: str,
                      csv_path: Path = DEFAULT_CLINICAL_CSV) -> str:
    """Return pre-generated clinical text for *annotation_id*, or '' if missing.

    Args:
        annotation_id: plain AnnotationID as used in LUNA25 (e.g. '100570_1_19990102')
        csv_path: path to clinical_texts.csv
    """
    _load_clinical_cache(csv_path)
    return _clinical_cache.get(annotation_id, "")
