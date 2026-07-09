# Data Directory

This directory contains lightweight, de-identified tabular artifacts needed to
reproduce the reported analyses from released model outputs.

## Included

- `metadata/`: split files, prompt clinical-text table, selected LUNA25/NLST
  clinical metadata fields used by the audits, LUNA25 public metadata, and LNDb
  evaluation sheet used by the scripts.
- `predictions/`: per-sample DL, MedGemma, and ResNet18 adaptation prediction
  files used by the manuscript analyses.

## Not included

Raw CT image data, NLST source records, LNDb image volumes, local checkpoints,
and generated MP4 videos are not redistributed. Obtain source data from LUNA25,
LNDb, and the applicable NLST provider, then place local files according to
`benchmark/paths.py`.
