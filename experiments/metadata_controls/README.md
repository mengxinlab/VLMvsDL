# Metadata-control experiments

This folder contains the F0 text-only metadata-control experiment for the
VLM-vs-DL paper. It is intentionally separate from the main benchmark scripts,
while reusing the existing Gemini client, upload cache, fixed 20-shot exemplars,
split files, and statistics conventions.

## Conditions

- `F0 text-only`: F3 without exemplar videos or target video. The request keeps
  the same 20 exemplar clinical texts, exemplar labels, target clinical text,
  rich system prompt, JSON schema, model, and temperature.

This control is named F0 because it retains the 20-shot/few-shot text scaffold
while removing the visual input. Prediction filenames retain the earlier
`f0_textonly_*` prefix for backward compatibility.

An earlier `permuted-text` scaffold can still be generated for request-shape
inspection, but deliberately mismatching another patient's clinical descriptors
to a target CT video was not used in the manuscript because it creates an
artificial input state rather than a clinically meaningful diagnostic request.

## Outputs

- `results/vlm/luna25_controls/f0_textonly_gemini3flash.jsonl`
- `results/vlm/luna25_controls/request_dumps/`
- `results/vlm/luna25_controls/metadata_control_metrics.csv`
- `results/vlm/luna25_controls/metadata_control_delong.csv`
- `results/vlm/luna25_controls/metadata_control_association_audit.csv`
- `results/vlm/luna25_controls/SUMMARY.md`

## Dry request inspection

Dry runs do not upload videos and do not call the model. They generate
request-shape dumps with placeholder file URIs.

```bash
python3 experiments/metadata_controls/run_metadata_controls.py \
  --condition text-only
```

## Paid smoke tests

Only add `--run-api` after inspecting the dry dumps. Smoke outputs are written
under `results/vlm/luna25_controls/smoke/`.

```bash
python3 experiments/metadata_controls/run_metadata_controls.py \
  --condition text-only \
  --run-api \
  --smoke-limit 10 \
  --workers 2 \
  --sleep 5
```

## Batch API for F0 text-only

`F0 text-only` is pure text, so it can also be run through Gemini Batch API.
Batch requests are asynchronous and priced at 50% of standard Gemini API cost.
The helper below keeps the same 20 exemplar texts, labels, rich system prompt,
temperature, and JSON schema, then converts the result back to the normal
prediction JSONL used by the analysis script.

Prepare a 10-case smoke batch without making API calls:

```bash
python3 experiments/metadata_controls/batch_textonly.py prepare \
  --limit 10 \
  --suffix smoke
```

Submit and monitor the smoke batch:

```bash
python3 experiments/metadata_controls/batch_textonly.py submit --suffix smoke
python3 experiments/metadata_controls/batch_textonly.py poll --suffix smoke
python3 experiments/metadata_controls/batch_textonly.py download --suffix smoke
python3 experiments/metadata_controls/batch_textonly.py convert --suffix smoke
```

For the full 917-case text-only control:

```bash
python3 experiments/metadata_controls/batch_textonly.py prepare
python3 experiments/metadata_controls/batch_textonly.py submit
python3 experiments/metadata_controls/batch_textonly.py poll
python3 experiments/metadata_controls/batch_textonly.py download
python3 experiments/metadata_controls/batch_textonly.py convert
```

## Full runs

Run this only after checking the smoke request dumps and parsed JSON outputs.

```bash
python3 experiments/metadata_controls/run_metadata_controls.py \
  --condition text-only \
  --run-api \
  --full \
  --workers 2 \
  --sleep 5
```

## Analysis

After the full text-only JSONL file exists:

```bash
python3 experiments/metadata_controls/analyze_metadata_controls.py
```
