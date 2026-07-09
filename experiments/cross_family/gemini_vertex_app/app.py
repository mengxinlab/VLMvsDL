from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from vertex_runner import CONDITIONS, run_panel

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "data"
DEFAULT_OUTPUTS = APP_DIR / "outputs"


def count_jsonl(path: Path) -> tuple[int, int]:
    ok = bad = 0
    if not path.exists():
        return ok, bad
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
            confidence = float(record.get("confidence", -1))
            if 0 <= confidence <= 1:
                ok += 1
            else:
                bad += 1
        except Exception:
            bad += 1
    return ok, bad


st.set_page_config(page_title="Gemini Metadata Audit", layout="wide")
st.title("Gemini Metadata Audit")

with st.sidebar:
    st.header("Vertex AI")
    project = st.text_input("Google Cloud project", value=os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    location = st.text_input("Location", value=os.environ.get("VERTEX_LOCATION", "us-central1"))
    model = st.text_input("Model", value=os.environ.get("VERTEX_MODEL", "google/gemini-3.1-pro-preview"))
    model_tag = st.text_input("Output tag", value=os.environ.get("MODEL_TAG", "gemini31pro_vertex"))

    st.header("Data")
    frames_dir = Path(st.text_input("Frames dir", value=str(DEFAULT_DATA / "frames")))
    clinical_csv = Path(st.text_input("Clinical CSV", value=str(DEFAULT_DATA / "clinical_texts.csv")))
    out_dir = Path(st.text_input("Output dir", value=str(DEFAULT_OUTPUTS)))

    st.header("Run")
    conditions = st.multiselect("Conditions", list(CONDITIONS), default=list(CONDITIONS))
    num_shards = st.number_input("Num shards", min_value=1, max_value=64, value=12, step=1)
    shard_index = st.number_input("Shard index", min_value=0, max_value=int(num_shards) - 1, value=0, step=1)
    max_output_tokens = st.number_input("Max output tokens", min_value=32, max_value=512, value=96, step=16)
    limit = st.number_input("Limit", min_value=1, max_value=100000, value=100000, step=1)

st.write(
    "Runs the Z0/Z2/Z3 metadata-control triad with the same JSONL schema as "
    "the public API runner: `image-only`, `image-text`, and `text-only`."
)

frames_ok = (frames_dir / "manifest.csv").exists()
clinical_ok = clinical_csv.exists()

col1, col2, col3 = st.columns(3)
col1.metric("Frames manifest", "OK" if frames_ok else "missing")
col2.metric("Clinical CSV", "OK" if clinical_ok else "missing")
if frames_ok:
    manifest = pd.read_csv(frames_dir / "manifest.csv")
    rows = [i for i in range(len(manifest)) if i % int(num_shards) == int(shard_index)]
    col3.metric("Shard cases", len(rows))
else:
    col3.metric("Shard cases", "n/a")

if st.button("Run shard", type="primary", disabled=not (project and frames_ok and clinical_ok and conditions)):
    try:
        with st.spinner("Running Vertex AI Gemini calls..."):
            written = run_panel(
                project=project,
                location=location,
                model=model,
                model_tag=model_tag,
                frames_dir=frames_dir,
                clinical_csv=clinical_csv,
                out_dir=out_dir,
                conditions=tuple(conditions),
                num_shards=int(num_shards),
                shard_index=int(shard_index),
                max_output_tokens=int(max_output_tokens),
                limit=int(limit),
            )
        st.success("Run complete")
        for path in written:
            ok, bad = count_jsonl(path)
            st.write(f"`{path.name}`: ok={ok}, bad={bad}")
    except Exception as exc:
        st.error(str(exc))

st.subheader("Outputs")
out_dir.mkdir(parents=True, exist_ok=True)
for path in sorted(out_dir.glob("crossfamily_*.jsonl")):
    ok, bad = count_jsonl(path)
    with st.expander(f"{path.name}  ok={ok} bad={bad}"):
        st.download_button(
            "Download JSONL",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/jsonl",
            key=str(path),
        )
        st.code(path.read_text().splitlines()[:3])
