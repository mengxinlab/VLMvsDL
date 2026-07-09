#!/usr/bin/env python3
"""Run F0/F3 metadata counterfactual controls.

The script defaults to a no-API dry run. Add ``--run-api`` to upload/call
Gemini. Full 917-case outputs require both ``--run-api`` and ``--full``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    UploadedVideo,
    build_request_contents,
    predict_one,
)
from benchmark.paths import CLINICAL_TEXT_CSV, METADATA_DIR, VLM_RESULTS_DIR  # noqa: E402
from benchmark.result_utils import parse_prediction_text  # noqa: E402
from benchmark.run_sync import (  # noqa: E402
    _extract_denied_file_id,
    _load_upload_cache,
    _refresh_request_videos,
    append_result,
    cached_upload,
    load_done,
)

SAMPLES_PATH = BENCH / "few_shot_samples.json"
OUT_DIR = VLM_RESULTS_DIR / "luna25_controls"
SMOKE_DIR = OUT_DIR / "smoke"
DUMP_DIR = OUT_DIR / "request_dumps"
MANIFEST_DIR = OUT_DIR / "manifests"
PERMUTED_CSV = METADATA_DIR / "clinical_texts_permuted_seed42.csv"

CONDITIONS = ("text-only", "permuted-text")
CONDITION_TO_FILE = {
    "text-only": "f0_textonly_gemini3flash.jsonl",
    "permuted-text": "f3_permuted_metadata_gemini3flash.jsonl",
}
CONDITION_LABELS = {
    "text-only": "F0 text-only",
    "permuted-text": "F3-permuted-text",
}


def plain_aid(raw: str) -> str:
    return "_".join(str(raw).split("_")[1:])


def raw_aid(aid: str) -> str:
    return f"{str(aid).split('_')[0]}_{aid}"


def load_clinical_map(csv_path: Path = CLINICAL_TEXT_CSV) -> dict[str, str]:
    df = pd.read_csv(csv_path, usecols=["AnnotationID", "clinical_text"])
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        out[plain_aid(row["AnnotationID"])] = (
            str(row["clinical_text"]) if pd.notna(row["clinical_text"]) else ""
        )
    return out


def generate_derangement(test_rows: pd.DataFrame,
                         clinical: dict[str, str],
                         seed: int = 42,
                         out_csv: Path = PERMUTED_CSV) -> pd.DataFrame:
    aids = [str(aid) for aid in test_rows[ID_COL].tolist()]
    labels = {str(row[ID_COL]): int(row[LABEL_COL]) for _, row in test_rows.iterrows()}
    rng = __import__("numpy").random.default_rng(seed)
    assigned = aids.copy()
    while True:
        rng.shuffle(assigned)
        if all(a != b for a, b in zip(aids, assigned)):
            break

    rows = []
    for target, source in zip(aids, assigned):
        rows.append({
            "AnnotationID": raw_aid(target),
            "target_plain_aid": target,
            "label": labels[target],
            "clinical_text": clinical.get(source, ""),
            "source_AnnotationID": raw_aid(source),
            "source_plain_aid": source,
            "source_label": labels[source],
            "perm_seed": seed,
        })
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def load_permuted_map(path: Path = PERMUTED_CSV) -> tuple[dict[str, str], dict[str, str]]:
    df = pd.read_csv(path)
    text_map = {
        str(row["target_plain_aid"]): str(row["clinical_text"])
        for _, row in df.iterrows()
    }
    source_map = {
        str(row["target_plain_aid"]): str(row["source_plain_aid"])
        for _, row in df.iterrows()
    }
    return text_map, source_map


def placeholder_video(aid: str, label: int | None = None) -> UploadedVideo:
    return UploadedVideo(
        annotation_id=aid,
        label=label,
        file_name=f"dryrun/{aid}",
        uri=f"dryrun://{aid}.mp4",
    )


def load_fixed_examples(condition: str,
                        clinical: dict[str, str],
                        upload_cache: dict[str, Any],
                        run_api: bool) -> tuple[list[UploadedVideo], list[str]]:
    samples = json.loads(SAMPLES_PATH.read_text())
    raw = samples["all_20"]
    examples: list[UploadedVideo] = []
    example_clinicals: list[str] = []

    include_example_videos = condition == "permuted-text"
    for sample in raw:
        aid = str(sample[ID_COL])
        label = int(sample[LABEL_COL])
        if include_example_videos and run_api:
            video = cached_upload(aid, upload_cache)
            video.label = label
        else:
            video = placeholder_video(aid, label)
        examples.append(video)
        example_clinicals.append(clinical.get(aid, ""))
    return examples, example_clinicals


def output_path(condition: str, full: bool) -> Path:
    if full:
        return OUT_DIR / CONDITION_TO_FILE[condition]
    return SMOKE_DIR / CONDITION_TO_FILE[condition].replace(".jsonl", "_smoke.jsonl")


def dump_requests(condition: str,
                  rows: pd.DataFrame,
                  clinical: dict[str, str],
                  permuted: dict[str, str],
                  source_map: dict[str, str],
                  examples: list[UploadedVideo],
                  example_clinicals: list[str],
                  count: int,
                  model: str,
                  temperature: float,
                  video_sample_fps: float | None) -> Path:
    include_videos = condition == "permuted-text"
    dumped = []
    for _, row in rows.head(count).iterrows():
        aid = str(row[ID_COL])
        label = int(row[LABEL_COL])
        target_clinical = clinical[aid] if condition == "text-only" else permuted[aid]
        target = None if condition == "text-only" else placeholder_video(aid)
        contents = build_request_contents(
            target,
            examples=examples,
            target_clinical=target_clinical,
            example_clinicals=example_clinicals,
            video_sample_fps=video_sample_fps,
            include_target_video=include_videos,
            include_example_videos=include_videos,
        )
        dumped.append({
            "condition": condition,
            "model": model,
            "temperature": temperature,
            "target_aid": aid,
            "target_label_not_in_prompt": label,
            "target_video_present": include_videos,
            "example_video_count": 20 if include_videos else 0,
            "target_text_source_aid": aid if condition == "text-only" else source_map[aid],
            "target_text_source_differs": (
                True if condition == "text-only" else aid != source_map[aid]
            ),
            "contents": contents,
        })
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    path = DUMP_DIR / f"{condition}_request_dump_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(dumped, indent=2))
    return path


def run_one(aid: str,
            label: int,
            condition: str,
            model: str,
            examples: list[UploadedVideo],
            example_clinicals: list[str],
            clinical: dict[str, str],
            permuted: dict[str, str],
            source_map: dict[str, str],
            upload_cache: dict[str, Any],
            result_path: Path,
            temperature: float,
            video_sample_fps: float | None,
            counter: list[int],
            counter_lock: threading.Lock) -> bool:
    include_videos = condition == "permuted-text"
    try:
        target = None
        if include_videos:
            target_video = cached_upload(aid, upload_cache)
            target = UploadedVideo(
                annotation_id=aid,
                label=None,
                file_name=target_video.file_name,
                uri=target_video.uri,
                mime_type=target_video.mime_type,
            )
        target_clinical = clinical[aid] if condition == "text-only" else permuted[aid]
        request_examples = [
            UploadedVideo(
                annotation_id=ex.annotation_id,
                label=ex.label,
                file_name=ex.file_name,
                uri=ex.uri,
                mime_type=ex.mime_type,
            )
            for ex in examples
        ]

        for request_attempt in range(3):
            try:
                result = predict_one(
                    target,
                    examples=request_examples,
                    model=model,
                    target_clinical=target_clinical,
                    example_clinicals=example_clinicals,
                    rich_prompt=True,
                    temperature=temperature,
                    video_sample_fps=video_sample_fps,
                    include_target_video=include_videos,
                    include_example_videos=include_videos,
                )
                break
            except Exception as exc:  # noqa: BLE001
                denied_file_id = _extract_denied_file_id(exc)
                if denied_file_id is None or not include_videos or target is None:
                    raise
                refreshed = _refresh_request_videos(
                    target, request_examples, denied_file_id, upload_cache
                )
                if refreshed == 0 or request_attempt == 2:
                    raise
                print(
                    f"  [predict_one] refreshed {refreshed} request file(s) after "
                    f"PERMISSION_DENIED on {denied_file_id}; retrying",
                    flush=True,
                )

        raw = result["text"]
        confidence, reasoning = parse_prediction_text(raw)
        record = {
            "aid": aid,
            "label": label,
            "mode": "20shot_clinical",
            "prompt": "rich",
            "model": model,
            "temperature": temperature,
            "control_condition": CONDITION_LABELS[condition],
            "target_video_present": include_videos,
            "example_videos_present": include_videos,
            "target_text_source_aid": aid if condition == "text-only" else source_map[aid],
            "confidence": confidence,
            "reasoning": reasoning,
        }
        append_result(result_path, record)
        with counter_lock:
            counter[0] += 1
            n = counter[0]
        print(f"  [{n}] {aid} conf={confidence:.3f}  {reasoning[:70]}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERROR] {aid}: {exc}", flush=True)
        return False


def write_manifest(args: argparse.Namespace,
                   result_path: Path,
                   dump_path: Path,
                   n_todo: int,
                   n_total: int) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "model": args.model,
        "temperature": args.temperature,
        "run_api": args.run_api,
        "full": args.full,
        "workers": args.workers,
        "sleep": args.sleep,
        "limit": args.limit,
        "smoke_limit": args.smoke_limit,
        "n_todo_this_invocation": n_todo,
        "n_total_test_rows": n_total,
        "result_path": str(result_path.relative_to(ROOT)),
        "request_dump": str(dump_path.relative_to(ROOT)),
        "permutation_csv": str(PERMUTED_CSV.relative_to(ROOT)),
    }
    path = MANIFEST_DIR / f"{args.condition}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--video-sampling-fps", type=float, default=None)
    parser.add_argument("--perm-seed", type=int, default=42)
    parser.add_argument("--run-api", action="store_true",
                        help="Actually upload/call Gemini. Omit for dry request dumps.")
    parser.add_argument("--full", action="store_true",
                        help="Write the full 917-case result file. Requires --run-api.")
    parser.add_argument("--smoke-limit", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--dump-requests", type=int, default=2)
    args = parser.parse_args()

    if args.full and not args.run_api:
        raise SystemExit("--full requires --run-api")
    if args.run_api and not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is required when --run-api is set")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_label_df()
    split = load_split()
    # The control runs reuse already encoded MP4 clips under results/videos/fps20.
    # Public/review working copies may not contain raw .npy files, so do not
    # filter the split by raw-image availability here.
    test_rows = get_split_rows(df, split, "test", require_npy=False)
    clinical = load_clinical_map()
    missing = [str(aid) for aid in test_rows[ID_COL] if str(aid) not in clinical]
    if missing:
        raise RuntimeError(f"{len(missing)} test cases missing clinical text; first={missing[:3]}")

    perm_df = generate_derangement(
        test_rows, clinical, seed=args.perm_seed, out_csv=PERMUTED_CSV
    )
    permuted, source_map = load_permuted_map(PERMUTED_CSV)
    deranged_ok = all(aid != source_map[aid] for aid in test_rows[ID_COL].astype(str))
    if not deranged_ok:
        raise RuntimeError("Permutation is not a full derangement")

    upload_cache = _load_upload_cache()
    examples, example_clinicals = load_fixed_examples(
        args.condition, clinical, upload_cache, run_api=args.run_api
    )
    dump_path = dump_requests(
        args.condition,
        test_rows,
        clinical,
        permuted,
        source_map,
        examples,
        example_clinicals,
        count=args.dump_requests,
        model=args.model,
        temperature=args.temperature,
        video_sample_fps=args.video_sampling_fps,
    )

    result_path = output_path(args.condition, full=args.full)
    done = load_done(result_path)
    limit = args.limit if args.limit is not None else (len(test_rows) if args.full else args.smoke_limit)
    todo = []
    for _, row in test_rows.iterrows():
        aid = str(row[ID_COL])
        if aid not in done:
            todo.append((aid, int(row[LABEL_COL])))
        if len(todo) >= limit:
            break

    print(f"Condition: {CONDITION_LABELS[args.condition]}")
    print(f"Model: {args.model} | temperature={args.temperature} | rich prompt | fixed 20-shot")
    print(f"Permutation CSV: {PERMUTED_CSV.relative_to(ROOT)} ({len(perm_df)} rows, seed={args.perm_seed})")
    print(f"Request dump: {dump_path.relative_to(ROOT)}")
    print(f"Output: {result_path.relative_to(ROOT)}")
    print(f"Already done: {len(done)} | queued now: {len(todo)} | full test rows: {len(test_rows)}")
    print(f"Run API: {args.run_api} | workers={args.workers} | sleep={args.sleep}")
    if args.condition == "text-only":
        print("Request shape: 0 exemplar videos, 0 target videos, true target clinical text.")
    else:
        print("Request shape: 20 exemplar videos + real target video, permuted target clinical text.")
        first_aid = str(test_rows.iloc[0][ID_COL])
        print(f"First target text source: {first_aid} <- {source_map[first_aid]}")

    manifest_path = write_manifest(args, result_path, dump_path, len(todo), len(test_rows))
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")

    if not args.run_api:
        print("\nDry run only. Add --run-api for paid smoke/full inference.")
        return

    counter = [0]
    counter_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for aid, label in todo:
            future = executor.submit(
                run_one,
                aid,
                label,
                args.condition,
                args.model,
                examples,
                example_clinicals,
                clinical,
                permuted,
                source_map,
                upload_cache,
                result_path,
                args.temperature,
                args.video_sampling_fps,
                counter,
                counter_lock,
            )
            futures[future] = aid
            if args.sleep > 0:
                time.sleep(args.sleep)
        for _ in as_completed(futures):
            pass

    total_done = len(load_done(result_path))
    print(f"\nDone. Total successful records in {result_path.name}: {total_done}")


if __name__ == "__main__":
    main()
