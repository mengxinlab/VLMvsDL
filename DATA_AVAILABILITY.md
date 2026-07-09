# Data and Release Notes

This repository is intended to support the manuscript:

> Auditing metadata reliance in general-purpose vision-language models for CT lung nodule malignancy prediction

GitHub repository: https://github.com/mengxinlab/VLMvsDL

## Included in the public release

- Source code for CT-to-video conversion, prompt construction, VLM inference,
  DL-result evaluation, Brock/PanCan comparison, calibration, subgroup analyses,
  and figure/table generation.
- Fixed 20-shot exemplar IDs.
- Per-sample VLM outputs and model-score CSV/JSONL files used to recompute the
  reported tables.
- Gemini Z1-Z3/F1-F3 prompt-input ablation outputs, the matched F0 text-only
  metadata-control outputs, and summary CSVs used for the metadata-reliance
  audit.
- Cross-family Z0/Z2/Z3 hosted-model audit JSONL outputs and summary CSVs.
- Per-sample supervised DL prediction CSVs for LUNA25 and LNDb external
  validation.
- Manuscript and online-supplement figure files. Journal-specific submission
  packages, cover letters, title pages, Word files, and editorial
  correspondence are intentionally excluded.

## Not redistributed

The public release does not redistribute raw CT image data, raw NLST records,
LNDb image volumes, generated MP4 video inputs, local checkpoints, API caches,
or model weights. Obtain source data from the original providers and place them
under the local paths documented in `benchmark/paths.py`:

- LUNA25: https://luna25.grand-challenge.org/
- LNDb: https://lndb.grand-challenge.org/
- NLST: follow the applicable NLST data-access terms from the original data
  provider.

## Sensitive and local artifacts excluded from release

The release builder excludes:

- `data/raw/`
- `data/checkpoints/`
- `results/videos/`
- `results/cache/`
- `results/logs/`
- `results/training_runs/`
- `.venv/`
- `.claude/`
- `.vscode/`
- `docs/references/`
- archived exploratory notebooks and third-party template bundles
- model checkpoint files (`*.pth`) and generated MP4 files

The included per-sample outputs use benchmark annotation IDs and de-identified
structured fields only. Before making the repository public, rerun the release
audit for secrets, local paths, large binaries, raw imaging files, checkpoints,
generated videos, caches, logs, and notebook archives.
