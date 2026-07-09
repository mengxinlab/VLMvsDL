# VLM Result Files

This directory contains lightweight, publishable prediction outputs and summary
tables used by the manuscript. Raw API requests, response caches, generated CT
videos, repair backups, and batch job records are intentionally excluded from
the public release.

## Per-Sample VLM Outputs

- `luna25_ablation/`: LUNA25 Gemini 3 Flash Preview Z1-Z3/F1-F3 adaptation-grid runs.
- `luna25_replicates/`: additional Gemini F3 repeat runs used for the F3A
  mean-of-five sensitivity analysis.
- `luna25_sensitivity/`: exploratory F3 sensitivity runs, including the
  64-frame dense-input probe and temperature probes.
- `luna25_model_comparison/`: F3-style LUNA25 runs for other Gemini/Gemma and
  MedGemma models.
- `luna25_controls/`: F0 text-only metadata-control output and summary tables.
  The released full control is `f0_textonly_gemini3flash.jsonl`; intentionally
  metadata-permuted VLM inputs were not used in the manuscript.
- `lndb_external/`: LNDb external-validation VLM outputs.

## Summary Tables

- `luna25_model_metrics_main.csv`: main LUNA25 VLM comparison table inputs.
- `luna25_model_metrics_all.csv`: all retained LUNA25 VLM runs.
- `lndb_external_metrics.csv`: LNDb external-validation metrics.
- `f3_replicate_summary.csv`: five F3 repeat-run metrics.
- `f3a_mean_of5_predictions.csv` and `f3a_mean_of5_metrics.json`: F3A
  mean-of-five sensitivity outputs.
- `brock_pancan_predictions.csv`: verified Brock/McWilliams PanCan Model 2b
  comparator predictions.
- `canonical_numbers.csv`: single-source AUC reconciliation used for manuscript
  text, tables, and figures.
- `luna25_controls/metadata_control_metrics.csv`: AUCs for F2 image-only, F3
  image+text, and F0 text-only.
- `luna25_controls/metadata_control_delong.csv`: paired DeLong comparisons for
  F3 image+text versus F0 text-only and F2 image-only.
- `luna25_controls/metadata_control_association_audit.csv`: Spearman and margin
  association audit for the metadata-control analysis.

## F3 Temperature Sensitivity

The primary Gemini F3 estimate is the temperature-0 representative run:

- `luna25_ablation/f3_gemini3flash_20shot_rich_metadata_run00.jsonl`

Additional temperature sensitivity files are:

- `luna25_sensitivity/f3_gemini3flash_temperature1.jsonl`
- `luna25_sensitivity/f3_gemini3flash_temperature2.jsonl`

The early temperature-0 outlier disclosed in the supplement is retained as:

- `luna25_sensitivity/f3_gemini3flash_initial_t0_excluded.jsonl`

Each JSONL record also stores the numeric `temperature` used for that call.
