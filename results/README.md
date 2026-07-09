# Results Directory

The public release includes lightweight result files needed to reproduce the
manuscript tables and figures.

## Included

- `vlm/`: curated VLM JSONL outputs grouped by experiment, Brock comparison
  files, F3A mean-of-five predictions, classical metadata-only baseline metrics,
  F0 text-only metadata-control outputs, LNDb external-validation outputs, and
  manuscript summary CSVs.
- `figures/`: CSV/statistical side products used by the manuscript figure and
  table scripts.
- `ttt/`: summary metrics for exploratory Med3D ResNet18+TTT adaptation runs.

## Excluded

Generated videos, API request/response caches, batch job records, local logs,
training-run checkpoints, and temporary repair/back-up files are excluded from
the public release.
