# VLMvsDL

VLMvsDL audits metadata reliance in Gemini-family vision-language models for
pulmonary nodule malignancy classification on LUNA25/NLST, and benchmarks them
against supervised deep-learning baselines with LNDb external validation.

This repository provides the public code, released predictions, and analysis
outputs for:

> Auditing metadata reliance in general-purpose vision-language models for CT
> lung nodule malignancy prediction

The current release centers the metadata-reliance audit, including the Gemini
Z1-Z3/F1-F3 prompt-input ablation, the matched F0 metadata-only control, and the
CF-Z0/CF-Z2/CF-Z3 cross-family hosted-model audit.

## Project Layout

```text
benchmark/      Canonical Python runners, evaluators, and plotting utilities
data/           Lightweight metadata and publishable per-sample prediction CSVs
results/        Released VLM outputs, statistical tables, and figure data
```

Large local assets are intentionally excluded from the public release:
`data/raw/`, `data/checkpoints/`, generated videos, model checkpoints, logs, API
caches, `.venv/`, third-party reference PDFs, and exploratory notebook archives.
See `DATA_AVAILABILITY.md`.

## Setup

Use Python 3.10-3.12 for the scientific Python and PyTorch stack.

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

## Common Commands

```bash
# Python syntax check for the core code.
./.venv/bin/python -m py_compile \
  benchmark/gemini_client.py \
  benchmark/run_sync.py \
  benchmark/run_lndb.py \
  benchmark/evaluate_lndb_external.py \
  benchmark/stats_and_figs.py

# Recompute summary tables and analysis figures from existing JSONL outputs.
./.venv/bin/python -m benchmark.summarize_luna25_results
./.venv/bin/python -m benchmark.stats_and_figs
./.venv/bin/python -m benchmark.evaluate_lndb_external
./.venv/bin/python experiments/metadata_controls/analyze_metadata_controls.py
./.venv/bin/python experiments/cross_family/analyze_crossfamily.py

```

## Canonical Result Locations

- VLM JSONL outputs: `results/vlm/` grouped by experiment
- Gemini Z1-Z3/F1-F3 prompt-input ablation outputs:
  `results/vlm/luna25_ablation/`
- F0 metadata-only control outputs: `results/vlm/luna25_controls/`
- Cross-family CF-Z0/CF-Z2/CF-Z3 hosted-model audit outputs:
  `results/vlm/crossfamily/`
- LUNA25 DL per-sample CSVs: `data/predictions/luna25_dl/files/`
- LNDb DL per-sample CSVs: `data/predictions/lndb_dl/`
- Analysis figures and stats CSVs: `results/figures/`
- Exploratory ResNet18+TTT summary: `results/ttt/`

Raw LUNA25/NLST/LNDb image arrays are not suitable for public GitHub storage.
This release instead documents data access and provides reproducible scripts
plus per-sample prediction tables.

## License

Code is released under the Apache License 2.0. Data and third-party source
datasets remain subject to their original providers' terms.
