#!/usr/bin/env python3
"""Run the Z0/Z2/Z3 metadata-control panel with Vertex AI Gemini.

The module uses Vertex AI's OpenAI-compatible endpoint so it can run inside a
reproducible Google Cloud app (Cloud Run / Streamlit) or from the command line.

Output JSONL schema is identical to the public hosted-model API runner:
{aid, label, condition, mode, model, model_key, confidence, reasoning}
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import google.auth
import google.auth.transport.requests
import pandas as pd
from openai import OpenAI

CONDITIONS = ("image-only", "image-text", "text-only")

SYSTEM_PROMPT_RICH = (
    "You are an expert radiologist specializing in lung cancer CT diagnosis. "
    "You are shown a montage of axial CT slices of a single lung nodule (the "
    "nodule is centered in each slice). Predict whether the nodule is MALIGNANT "
    "or BENIGN.\n\n"
    "Key malignancy features to assess:\n"
    "1. Size/Volume: nodules >6 mm are higher risk; >15 mm strongly suspicious.\n"
    "2. Density: higher density (less ground-glass) correlates with invasiveness.\n"
    "3. Shape: irregular or lobulated shapes suggest malignancy; round/oval favors benign.\n"
    "4. Margins: spiculated or irregular margins are strongly malignant; smooth margins favor benign.\n"
    "5. Internal features: vacuolization and air bronchograms increase with malignancy.\n"
    "6. Vascular features: vessel penetration and internal vascular thickening indicate malignancy.\n"
    "7. Pleural attachment: pleural indentation or traction suggests invasiveness.\n\n"
    'Return ONLY valid JSON: {"confidence": <float 0-1, probability of malignancy>, '
    '"reasoning": <one sentence describing the key feature driving your decision>}'
)
TEXT_ONLY_NOTE = (
    "No CT images are provided for this case. Base your assessment only on the "
    "structured clinical and radiological description below."
)


def parse_generation(raw_text: str) -> tuple[float, str]:
    confidence, reasoning = -1.0, raw_text
    try:
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
        body = fenced.group(1) if fenced else raw_text
        match = re.search(r"\{[\s\S]*\}", body)
        parsed = json.loads(match.group(0) if match else body)
        confidence = float(parsed.get("confidence", -1))
        reasoning = str(parsed.get("reasoning", ""))
    except Exception:
        pass
    return confidence, reasoning


def load_clinical(clin_csv: Path) -> dict[str, str]:
    df = pd.read_csv(clin_csv)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        raw = str(row["AnnotationID"]).strip()
        text = "" if pd.isna(row["clinical_text"]) else str(row["clinical_text"])
        out[raw] = text
        out["_".join(raw.split("_")[1:])] = text
    return out


def b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def user_text(condition: str, clinical_text: str) -> str:
    if condition == "text-only":
        body = TEXT_ONLY_NOTE + ("\n" + clinical_text if clinical_text else "")
        return body + "\nClassify this case. Return ONLY the JSON."
    prefix = (clinical_text + "\n") if (condition == "image-text" and clinical_text) else ""
    return prefix + "Classify this case. Return ONLY the JSON."


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
            confidence = float(record.get("confidence", -1))
            if 0.0 <= confidence <= 1.0:
                done.add(str(record["aid"]))
        except Exception:
            pass
    return done


def vertex_openai_client(project: str, location: str) -> OpenAI:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(google.auth.transport.requests.Request())
    base_url = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/endpoints/openapi"
    )
    return OpenAI(api_key=credentials.token, base_url=base_url)


def build_messages(condition: str, frames_dir: Path, aid: str, n_frames: int, clinical_text: str):
    content = []
    if condition != "text-only":
        for j in range(n_frames):
            fp = frames_dir / aid / f"{j}.png"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_png(fp)}"},
            })
    content.append({"type": "text", "text": user_text(condition, clinical_text)})
    return [
        {"role": "system", "content": SYSTEM_PROMPT_RICH},
        {"role": "user", "content": content},
    ]


def call_vertex(client: OpenAI, model: str, messages, max_output_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_output_tokens,
    )
    return response.choices[0].message.content or ""


def run_panel(
    project: str,
    location: str,
    model: str,
    model_tag: str,
    frames_dir: Path,
    clinical_csv: Path,
    out_dir: Path,
    conditions: tuple[str, ...] = CONDITIONS,
    num_shards: int = 12,
    shard_index: int = 0,
    max_output_tokens: int = 96,
    limit: int = 100000,
    sleep_seconds: float = 0.0,
) -> list[Path]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_index < num_shards):
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if not (frames_dir / "manifest.csv").exists():
        raise FileNotFoundError(f"manifest.csv not found under {frames_dir}")
    if not clinical_csv.exists():
        raise FileNotFoundError(f"clinical CSV not found: {clinical_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(frames_dir / "manifest.csv")
    clinical = load_clinical(clinical_csv)
    client = vertex_openai_client(project, location)
    written: list[Path] = []

    rows = [
        row for i, (_, row) in enumerate(manifest.iterrows())
        if i % num_shards == shard_index
    ]

    for condition in conditions:
        result_path = out_dir / f"crossfamily_{model_tag}_{condition}.jsonl"
        done = load_done(result_path)
        todo = [row for row in rows if str(row["aid"]) not in done][:limit]
        print(
            f"=== {model} | {condition} | shard={shard_index}/{num_shards} "
            f"| done={len(done)} shard_rows={len(rows)} todo={len(todo)}",
            flush=True,
        )
        with result_path.open("a") as handle:
            for k, row in enumerate(todo, start=1):
                aid = str(row["aid"])
                label = int(row["label"])
                clinical_text = clinical.get(aid, "") if condition != "image-only" else ""
                messages = build_messages(
                    condition, frames_dir, aid, int(row["n_frames"]), clinical_text
                )
                raw = call_vertex(client, model, messages, max_output_tokens)
                confidence, reasoning = parse_generation(raw)
                record = {
                    "aid": aid,
                    "label": label,
                    "condition": condition,
                    "mode": "zeroshot",
                    "provider": "vertex-ai",
                    "model": model_tag,
                    "model_key": model,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(f"[{len(done)+k}] {aid} conf={confidence:.3f} {reasoning[:80]}", flush=True)
                if sleep_seconds:
                    time.sleep(sleep_seconds)
        written.append(result_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    parser.add_argument("--location", default=os.environ.get("VERTEX_LOCATION", "us-central1"))
    parser.add_argument("--model", default=os.environ.get("VERTEX_MODEL", "google/gemini-3.1-pro-preview"))
    parser.add_argument("--model-tag", default=os.environ.get("MODEL_TAG", "gemini31pro_vertex"))
    parser.add_argument("--frames-dir", default="data/frames")
    parser.add_argument("--clinical-csv", default="data/clinical_texts.csv")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--condition", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--num-shards", type=int, default=12)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if not args.project:
        raise SystemExit("--project or GOOGLE_CLOUD_PROJECT is required")
    run_panel(
        project=args.project,
        location=args.location,
        model=args.model,
        model_tag=args.model_tag,
        frames_dir=Path(args.frames_dir),
        clinical_csv=Path(args.clinical_csv),
        out_dir=Path(args.out_dir),
        conditions=tuple(args.condition),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_output_tokens=args.max_output_tokens,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()
