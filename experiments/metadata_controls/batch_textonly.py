#!/usr/bin/env python3
"""Gemini Batch API helper for the F0 text-only control.

Subcommands:
  prepare   Build JSONL requests and a key map. No API calls.
  submit    Upload the JSONL file and create a Batch API job.
  poll      Print the current Batch API job state.
  download  Download the raw JSONL result file after success.
  convert   Convert raw batch JSONL into the existing prediction JSONL schema.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH))

from benchmark.data_utils import ID_COL, LABEL_COL, get_split_rows, load_label_df, load_split  # noqa: E402
from benchmark.gemini_client import (  # noqa: E402
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT_RICH,
    build_request_contents,
)
from benchmark.paths import VLM_RESULTS_DIR  # noqa: E402
from benchmark.result_utils import parse_prediction_text  # noqa: E402
from experiments.metadata_controls.run_metadata_controls import (  # noqa: E402
    load_clinical_map,
    load_fixed_examples,
)

OUT_DIR = VLM_RESULTS_DIR / "luna25_controls"
BATCH_DIR = OUT_DIR / "batch_textonly"
DEFAULT_MODEL = "gemini-3-flash-preview"
REQUEST_BASENAME = "f0_textonly_batch_requests"
KEYMAP_BASENAME = "f0_textonly_batch_key_map"
JOB_BASENAME = "f0_textonly_batch_job"
RAW_RESULT_BASENAME = "f0_textonly_batch_results_raw"
FINAL_JSONL = OUT_DIR / "f0_textonly_gemini3flash.jsonl"
SMOKE_FINAL_JSONL = OUT_DIR / "smoke" / "f0_textonly_gemini3flash_batch_smoke.jsonl"


def suffix_part(suffix: str) -> str:
    return f"_{suffix}" if suffix else ""


def request_path(suffix: str) -> Path:
    return BATCH_DIR / f"{REQUEST_BASENAME}{suffix_part(suffix)}.jsonl"


def keymap_path(suffix: str) -> Path:
    return BATCH_DIR / f"{KEYMAP_BASENAME}{suffix_part(suffix)}.csv"


def job_path(suffix: str) -> Path:
    return BATCH_DIR / f"{JOB_BASENAME}{suffix_part(suffix)}.json"


def raw_result_path(suffix: str) -> Path:
    return BATCH_DIR / f"{RAW_RESULT_BASENAME}{suffix_part(suffix)}.jsonl"


def final_path(suffix: str) -> Path:
    return SMOKE_FINAL_JSONL if suffix else FINAL_JSONL


def get_test_rows(limit: int | None = None) -> pd.DataFrame:
    df = load_label_df()
    split = load_split()
    rows = get_split_rows(df, split, "test", require_npy=False)
    if limit is not None:
        rows = rows.head(limit).copy()
    return rows.reset_index(drop=True)


def generation_config(style: str, temperature: float) -> dict[str, Any]:
    config = {
        "temperature": temperature,
        "response_mime_type": "application/json",
        "response_schema": RESPONSE_SCHEMA,
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT_RICH}]},
    }
    if style == "config":
        return {"config": config}
    if style == "generation_config":
        # Older JSONL examples use generation_config. Keep the same system
        # instruction at request top level so the line remains self-contained.
        return {
            "generation_config": {
                "temperature": config["temperature"],
                "response_mime_type": config["response_mime_type"],
                "response_schema": config["response_schema"],
            },
            "system_instruction": config["system_instruction"],
        }
    if style == "plain_json":
        return {
            "generation_config": {
                "temperature": config["temperature"],
                "response_mime_type": config["response_mime_type"],
            },
            "system_instruction": config["system_instruction"],
        }
    raise ValueError(f"Unknown config style: {style}")


def build_batch_request(aid: str,
                        clinical_text: str,
                        examples: list[Any],
                        example_clinicals: list[str],
                        temperature: float,
                        config_style: str) -> dict[str, Any]:
    contents = build_request_contents(
        None,
        examples=examples,
        target_clinical=clinical_text,
        example_clinicals=example_clinicals,
        include_target_video=False,
        include_example_videos=False,
    )
    request: dict[str, Any] = {"contents": contents}
    request.update(generation_config(config_style, temperature))
    return {"key": aid, "request": request}


def command_prepare(args: argparse.Namespace) -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    rows = get_test_rows(args.limit)
    clinical = load_clinical_map()
    examples, example_clinicals = load_fixed_examples(
        "text-only", clinical, upload_cache={}, run_api=False
    )

    key_rows = []
    with request_path(args.suffix).open("w") as handle:
        for _, row in rows.iterrows():
            aid = str(row[ID_COL])
            label = int(row[LABEL_COL])
            req = build_batch_request(
                aid,
                clinical[aid],
                examples,
                example_clinicals,
                temperature=args.temperature,
                config_style=args.config_style,
            )
            handle.write(json.dumps(req) + "\n")
            key_rows.append({
                "key": aid,
                "aid": aid,
                "label": label,
                "condition": "F0 text-only",
                "model": args.model,
                "temperature": args.temperature,
                "config_style": args.config_style,
            })

    key_df = pd.DataFrame(key_rows)
    key_df.to_csv(keymap_path(args.suffix), index=False)

    size_mb = request_path(args.suffix).stat().st_size / (1024 * 1024)
    print(f"Wrote {request_path(args.suffix).relative_to(ROOT)}")
    print(f"Wrote {keymap_path(args.suffix).relative_to(ROOT)}")
    print(f"Requests: {len(key_df)} | JSONL size: {size_mb:.2f} MB")
    print("No API calls were made.")


def _client():
    from google import genai  # type: ignore

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required for Batch API submit/poll/download")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def command_submit(args: argparse.Namespace) -> None:
    from google.genai import types  # type: ignore

    req_path = request_path(args.suffix)
    if not req_path.exists():
        raise SystemExit(f"Missing request JSONL: {req_path}")
    client = _client()
    uploaded = client.files.upload(
        file=str(req_path),
        config=types.UploadFileConfig(
            display_name=f"vlmvsdl-f0-textonly{suffix_part(args.suffix)}",
            mime_type="jsonl",
        ),
    )
    job = client.batches.create(
        model=args.model,
        src=uploaded.name,
        config={"display_name": f"vlmvsdl-f0-textonly{suffix_part(args.suffix)}"},
    )
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "suffix": args.suffix,
        "request_jsonl": str(req_path.relative_to(ROOT)),
        "key_map": str(keymap_path(args.suffix).relative_to(ROOT)),
        "uploaded_file_name": uploaded.name,
        "uploaded_file_uri": getattr(uploaded, "uri", ""),
        "batch_job_name": job.name,
        "batch_job_state": getattr(getattr(job, "state", None), "name", str(getattr(job, "state", ""))),
    }
    job_path(args.suffix).write_text(json.dumps(record, indent=2))
    print(f"Uploaded file: {uploaded.name}")
    print(f"Created batch job: {job.name}")
    print(f"Wrote {job_path(args.suffix).relative_to(ROOT)}")


def load_job_name(args: argparse.Namespace) -> str:
    if args.job_name:
        return args.job_name
    path = job_path(args.suffix)
    if not path.exists():
        raise SystemExit(f"Missing job metadata: {path}")
    return json.loads(path.read_text())["batch_job_name"]


def command_poll(args: argparse.Namespace) -> None:
    client = _client()
    job = client.batches.get(name=load_job_name(args))
    state = getattr(getattr(job, "state", None), "name", str(getattr(job, "state", "")))
    print(f"Job: {job.name}")
    print(f"State: {state}")
    if getattr(job, "error", None):
        print(f"Error: {job.error}")
    if getattr(job, "dest", None):
        print(f"Dest: {job.dest}")


def command_download(args: argparse.Namespace) -> None:
    client = _client()
    job = client.batches.get(name=load_job_name(args))
    state = getattr(getattr(job, "state", None), "name", str(getattr(job, "state", "")))
    if state != "JOB_STATE_SUCCEEDED":
        raise SystemExit(f"Batch job is not succeeded yet: {state}")
    if not getattr(job, "dest", None) or not getattr(job.dest, "file_name", None):
        raise SystemExit("Batch job has no result file")
    downloaded = client.files.download(file=job.dest.file_name)
    if isinstance(downloaded, bytes):
        raw_result_path(args.suffix).write_bytes(downloaded)
    elif hasattr(downloaded, "read"):
        raw_result_path(args.suffix).write_bytes(downloaded.read())
    else:
        raw_result_path(args.suffix).write_text(str(downloaded))
    print(f"Downloaded {job.dest.file_name}")
    print(f"Wrote {raw_result_path(args.suffix).relative_to(ROOT)}")


def response_text_from_record(record: dict[str, Any]) -> tuple[str | None, str | None]:
    if record.get("error"):
        return None, json.dumps(record["error"], ensure_ascii=False)
    response = record.get("response") or record.get("inlineResponse", {}).get("response")
    if not response:
        return None, "missing response"
    try:
        parts = response["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        return None, f"malformed response: {exc}"
    texts = [str(part["text"]) for part in parts if "text" in part]
    if not texts:
        return None, "missing text part"
    return "\n".join(texts), None


def command_convert(args: argparse.Namespace) -> None:
    raw_path = raw_result_path(args.suffix)
    if not raw_path.exists():
        raise SystemExit(f"Missing raw batch result: {raw_path}")
    key_df = pd.read_csv(keymap_path(args.suffix))
    by_key = {str(row["key"]): row for _, row in key_df.iterrows()}

    out_rows = []
    errors = []
    for line_no, line in enumerate(raw_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = str(record.get("key", ""))
        meta = by_key.get(key)
        if meta is None:
            errors.append({"line": line_no, "key": key, "error": "unknown key"})
            continue
        text, err = response_text_from_record(record)
        if err is not None:
            errors.append({"line": line_no, "key": key, "error": err})
            confidence, reasoning = -1.0, err
        else:
            confidence, reasoning = parse_prediction_text(text or "")
        out_rows.append({
            "aid": str(meta["aid"]),
            "label": int(meta["label"]),
            "mode": "20shot_clinical",
            "prompt": "rich",
            "model": str(meta["model"]),
            "temperature": float(meta["temperature"]),
            "control_condition": "F0 text-only",
            "batch_api": True,
            "target_video_present": False,
            "example_videos_present": False,
            "confidence": confidence,
            "reasoning": reasoning,
        })

    out = final_path(args.suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in out_rows:
            handle.write(json.dumps(row) + "\n")

    if errors:
        err_path = BATCH_DIR / f"f0_textonly_batch_convert_errors{suffix_part(args.suffix)}.json"
        err_path.write_text(json.dumps(errors, indent=2))
        print(f"Conversion warnings: {len(errors)} -> {err_path.relative_to(ROOT)}")
    print(f"Wrote {out.relative_to(ROOT)} ({len(out_rows)} rows)")
    good = sum(0.0 <= float(row["confidence"]) <= 1.0 for row in out_rows)
    print(f"Valid confidence rows: {good}/{len(out_rows)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="build request JSONL without API calls")
    prep.add_argument("--model", default=DEFAULT_MODEL)
    prep.add_argument("--temperature", type=float, default=0.0)
    prep.add_argument("--limit", type=int, default=None,
                      help="Use e.g. --limit 10 for a smoke batch.")
    prep.add_argument("--suffix", default="",
                      help="File suffix, e.g. smoke. Empty suffix is the full run.")
    prep.add_argument(
        "--config-style",
        choices=("generation_config", "config", "plain_json"),
        default="generation_config",
        help=(
            "Batch JSONL config style. generation_config matches the Gemini "
            "Batch API docs; plain_json omits response_schema if schema "
            "validation is rejected by the service."
        ),
    )
    prep.set_defaults(func=command_prepare)

    submit = sub.add_parser("submit", help="upload JSONL and create Batch API job")
    submit.add_argument("--model", default=DEFAULT_MODEL)
    submit.add_argument("--suffix", default="")
    submit.set_defaults(func=command_submit)

    poll = sub.add_parser("poll", help="poll Batch API job state")
    poll.add_argument("--suffix", default="")
    poll.add_argument("--job-name", default="")
    poll.set_defaults(func=command_poll)

    down = sub.add_parser("download", help="download succeeded batch result JSONL")
    down.add_argument("--suffix", default="")
    down.add_argument("--job-name", default="")
    down.set_defaults(func=command_download)

    conv = sub.add_parser("convert", help="convert raw batch results to prediction JSONL")
    conv.add_argument("--suffix", default="")
    conv.set_defaults(func=command_convert)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
