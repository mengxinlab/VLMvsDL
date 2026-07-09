# Gemini Metadata Audit App

Minimal Cloud Run / Streamlit app for running the cross-family metadata-control
panel on Vertex AI Gemini. It exists so the Gemini 3.1 Pro Z0/Z2/Z3 run can be
executed through a reproducible Cloud Run app path as well as the command-line
API runner.

## What It Runs

Conditions match the public API runner:

- `image-only` (Z2): rich instruction + 3 sampled CT frames
- `image-text` (Z3): rich instruction + 3 sampled CT frames + structured metadata
- `text-only` (Z0): rich instruction + structured metadata only

Outputs match the existing JSONL schema consumed by
`experiments/cross_family/analyze_crossfamily.py`.

## Prepare Data

From the repository root:

```bash
cd experiments/cross_family/gemini_vertex_app
bash prepare_data.sh
```

This copies:

- `results/vlm/crossfamily/frames/` (917 cases, 2751 PNGs)
- `data/metadata/clinical_texts.csv`

into `gemini_vertex_app/data/`.

## Local Smoke Test

Requires Google Application Default Credentials with Vertex AI access:

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=<your-project-id>
export VERTEX_LOCATION=us-central1

python vertex_runner.py \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --location "$VERTEX_LOCATION" \
  --model google/gemini-3.1-pro-preview \
  --model-tag gemini31pro_vertex \
  --frames-dir data/frames \
  --clinical-csv data/clinical_texts.csv \
  --out-dir outputs \
  --num-shards 12 \
  --shard-index 0 \
  --limit 3 \
  --max-output-tokens 96
```

If the Vertex OpenAI-compatible endpoint expects a slightly different model
identifier, keep the app code unchanged and edit the model string in the UI or
CLI.

## Deploy To Cloud Run

```bash
PROJECT=<your-project-id>
REGION=us-central1
SERVICE=gemini-metadata-audit

gcloud config set project "$PROJECT"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com

# Use the default compute service account, or replace with your own.
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SA}" \
  --role="roles/aiplatform.user"

bash prepare_data.sh

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --service-account "$SA" \
  --no-allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT},VERTEX_LOCATION=${REGION},VERTEX_MODEL=google/gemini-3.1-pro-preview,MODEL_TAG=gemini31pro_vertex"
```

Open the Cloud Run URL while authenticated, select a shard, and click **Run
shard**. For full Gemini 3.1 Pro coverage, run shards `0..11`. If a shard takes
too long, run one condition at a time from the condition multiselect.

## Download And Merge

Download the three JSONL files from the app's output panel and place them under
the usual shard staging folder, for example:

```text
results/vlm/crossfamily/shards/gemini31pro_vertex_shard00/
```

Then merge:

```bash
python experiments/cross_family/merge_shards.py
python experiments/cross_family/analyze_crossfamily.py
```

## Notes

- `max_output_tokens` defaults to 96 because the output is a short JSON object
  and lower output caps reduce provider quota reservation.
- Data is bundled into the Cloud Run source for the first version. Moving frames
  and outputs to GCS is straightforward, but not needed to start.
- This app is not for clinical use; it is a reproducible benchmark runner for
  the manuscript audit.
