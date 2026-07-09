"""Central filesystem layout for the VLMvsDL project."""
from __future__ import annotations

from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
RAW_DATA_DIR = DATA_DIR / "raw"
PREDICTIONS_DIR = DATA_DIR / "predictions"

LUNA25_IMAGE_DIR = RAW_DATA_DIR / "luna25" / "image"
LNDB_ROI_DIR = RAW_DATA_DIR / "lndb_roi"

RESULTS_DIR = PROJECT_ROOT / "results"
VLM_RESULTS_DIR = RESULTS_DIR / "vlm"
# CSV/stat side-products still land here; manuscript PDFs are written straight
# into MANUSCRIPT_FIGURES_DIR so there is no manual copy step (the old
# results/figures -> manuscript/figures copy was a recurring drift source).
FIGURES_DIR = RESULTS_DIR / "figures"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
MANUSCRIPT_FIGURES_DIR = MANUSCRIPT_DIR / "figures"
VIDEOS_DIR = RESULTS_DIR / "videos"
CACHE_DIR = RESULTS_DIR / "cache"

LUNA25_LABEL_CSV = METADATA_DIR / "luna25_public_training_development_data.csv"
LUNA25_CLINICAL_METADATA_CSV = METADATA_DIR / "luna25_clinical_metadata.csv"
PATIENT_SPLIT_JSON = METADATA_DIR / "patient_split.json"
CLINICAL_TEXT_CSV = METADATA_DIR / "clinical_texts.csv"
LNDB_EVAL_CSV = METADATA_DIR / "lndb_10to1_eval.csv"

LUNA25_DL_PRED_DIR = PREDICTIONS_DIR / "luna25_dl" / "files"
LNDB_DL_PRED_DIR = PREDICTIONS_DIR / "lndb_dl"
