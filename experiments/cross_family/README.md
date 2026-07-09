# Cross-Family Metadata-Reliance Audit

This folder reproduces the cross-family hosted-model audit used in the
manuscript. The goal is not to create a vendor leaderboard; it is to test
whether the metadata-reliance pattern observed in the Gemini video ablation also
appears under a portable sparse visual input across model families.

## Conditions

The public naming follows the manuscript notation:

| condition | runner name | input | role |
|---|---|---|---|
| Z2 | `image-only` | rich instruction + 3 sampled CT slices | visual-only sparse-montage baseline |
| Z3 | `image-text` | rich instruction + 3 sampled CT slices + structured metadata | metadata-assisted sparse-montage condition |
| Z0 | `text-only` | rich instruction + structured metadata only | metadata-only control |

For each family, the headline readout is:

```text
metadata lift = AUC(Z3 image+metadata) - AUC(Z2 image-only)
text recovery = AUC(Z0 metadata-only)
```

The sampled slices are frames 9, 29, and 49 from each 64-frame clip, matching
the nominal sparse frame positions observed in the hosted Gemini video sampling
probe. Matching frame indices does not make the montage identical to native
video ingestion: the montage is a single concatenated image and may be resized,
tokenized, or parsed differently.

## Released Results

The released JSONL files live under `results/vlm/crossfamily/` and are consumed
by `analyze_crossfamily.py`.

Current manuscript rows:

| row | input path | Z2 | Z3 | Z0 |
|---|---|---:|---:|---:|
| Gemini 3 Flash video reference | native video ablation | 0.682 | 0.730 | n/a |
| Gemini 3 Flash montage bridge | hosted sparse montage | 0.508 | 0.722 | 0.721 |
| Claude Opus 4.8 | hosted sparse montage | 0.497 | 0.730 | 0.719 |
| Gemini 3.1 Pro | hosted sparse montage | 0.516 | 0.729 | 0.713 |
| GPT-5.5 | hosted sparse montage | 0.500* | 0.712 | 0.703 |

`*` GPT-5.5 image-only responses were constant at 0.5 with no-image rationales
through the tested interface and are retained as an interface-provenance check.

Regenerate the summary tables and figure:

```bash
python experiments/cross_family/analyze_crossfamily.py
```

Outputs:

```text
results/vlm/crossfamily/crossfamily_summary.csv
results/vlm/crossfamily/crossfamily_auc_by_condition.csv
results/vlm/crossfamily/crossfamily_association.csv
results/vlm/crossfamily/crossfamily_summary.md
results/vlm/crossfamily/crossfamily_delta_auc.pdf
```

## Hosted API Runner

`run_crossfamily_api.py` is the public, provider-agnostic runner. It uses the
same prompt, JSON schema, clinical-text lookup, and JSONL output schema as the
manuscript audit. It supports native Anthropic calls and OpenAI-compatible
endpoints.

Example:

```bash
python experiments/cross_family/export_frames_local.py

export OPENROUTER_API_KEY=...
python experiments/cross_family/run_crossfamily_api.py \
  --provider openai \
  --base-url https://openrouter.ai/api/v1 \
  --model openai/gpt-5.5-2026-04-23 \
  --model-name openai_gpt-5.5-2026-04-23 \
  --frames-dir results/vlm/crossfamily/frames \
  --clinical-csv data/metadata/clinical_texts.csv \
  --workers 4
```

Smoke first with `--limit 5`. For native Anthropic:

```bash
export ANTHROPIC_API_KEY=...
python experiments/cross_family/run_crossfamily_api.py \
  --provider anthropic \
  --model anthropic/claude-opus-4-8@default \
  --model-name anthropic_claude-opus-4-8@default \
  --frames-dir results/vlm/crossfamily/frames \
  --clinical-csv data/metadata/clinical_texts.csv \
  --limit 5
```

The public release does not redistribute raw CT volumes or derived frame PNGs.
If frame PNGs are unavailable, regenerate the sparse frame bundle from the
source CT crops/videos using `export_frames.py` or `export_frames_local.py` in a
private data environment, then run the hosted-model panel.

## Files

- `run_crossfamily_api.py` - hosted-model API runner for Z0/Z2/Z3.
- `run_crossfamily_offline.py` - local/offline HF VLM runner using the same
  prompts and output schema.
- `export_frames.py` - export sampled frames where raw CT crops are available.
- `export_frames_local.py` - export sampled frames from existing local MP4 clips.
- `merge_shards.py` - merge shard folders, validate confidence values, and
  deduplicate by `aid`.
- `analyze_crossfamily.py` - compute AUCs, paired DeLong metadata lift,
  text-only recovery, and score-structured-predictor associations.
