# Benchmark Code

This directory contains the canonical Python entry points for VLMvsDL.

The project-level filesystem layout is centralized in `benchmark/paths.py`.
Durable inputs live under `data/`, generated outputs under `results/`, and
paper assets under `manuscript/`.

## Main Entry Points

```bash
# Convert LUNA25 nodule crops to MP4 videos.
python benchmark/convert.py --fps 20

# Run Gemini inference with resumable threaded workers.
python benchmark/run_sync.py --mode 20shot_clinical --model gemini-3-flash-preview --rich-prompt

# Run LNDb external-validation inference.
python benchmark/run_lndb.py --mode 20shot --model gemini-3-flash-preview --rich-prompt

# Summarize LUNA25 VLM runs and regenerate analysis figures.
python -m benchmark.summarize_luna25_results
python -m benchmark.stats_and_figs

# Evaluate LNDb external validation against matched DL predictions.
python -m benchmark.evaluate_lndb_external
```

## Data Layout

```text
data/
  metadata/                  # split files, clinical text, evaluation sheets
  predictions/luna25_dl/     # LUNA25 DL baseline per-sample CSVs
  predictions/lndb_dl/       # LNDb DL external per-sample CSVs
  raw/                       # local raw arrays/images; ignored by git

results/
  vlm/                       # VLM JSONL and summary CSV outputs; ignored by git
  figures/                   # analysis figures and stats CSVs; ignored by git
  videos/                    # generated MP4 inputs; ignored by git
```

## Notes

- `GEMINI_API_KEY` must be set for Gemini API runs.
- Gemini inference is exposed through the same resumable threaded runner for
  all public reruns. The manuscript primary F3 result uses the legacy Run `00`
  file, while `F3A` is the five-run averaged estimator across Run `00` plus
  four additional repeats (`01`-`04`).
- `few_shot_samples.json` is the fixed 20-shot exemplar list used by the main
  experiments.
- Local data, videos, checkpoints, logs, and API caches are intentionally
  ignored so the GitHub repository can stay light and publishable.
