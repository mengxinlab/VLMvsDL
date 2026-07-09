#!/usr/bin/env python3
"""
Resumable concurrent benchmark runner with daily-limit awareness.

Modes:
  zeroshot         - video only, no examples, no clinical text
  5shot            - 5 few-shot examples, no clinical text
  20shot           - 20 few-shot examples, no clinical text
  20shot_clinical  - 20 few-shot examples + clinical text

Usage:
  python run_sync.py --mode zeroshot --model gemini-2.0-flash
  python run_sync.py --mode zeroshot --model gemini-2.0-flash --rich-prompt --workers 16

Results saved to results/sync_{mode}_{model}_{prompt}.jsonl (one JSON per line).
Re-running resumes from where it left off.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
BENCH = Path(__file__).parent
sys.path.insert(0, str(BENCH))

os.environ.setdefault("GEMINI_MODEL",   "gemini-3.1-pro-preview")

from data_utils import (get_clinical_text, get_split_rows, load_label_df,
                        load_split, ID_COL, LABEL_COL)
from gemini_client import can_access_file, upload_video, predict_one, UploadedVideo
from paths import CACHE_DIR, VIDEOS_DIR, VLM_RESULTS_DIR
from result_utils import is_success_record as _is_success_record, parse_prediction_text

VIDEO_DIR    = VIDEOS_DIR / "fps20"
SAMPLES_PATH = BENCH / "few_shot_samples.json"
RESULTS_DIR  = VLM_RESULTS_DIR
UPLOAD_CACHE = CACHE_DIR / "upload_cache.json"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODES = ("zeroshot", "5shot", "20shot", "20shot_clinical")

# ── thread-safe upload cache ────────────────────────────────────────────────
_cache_lock  = threading.Lock()
_result_lock = threading.Lock()
_verified_uploads = set()
_DENIED_FILE_RE = re.compile(r"access the file\s+([A-Za-z0-9]+)", re.IGNORECASE)

def _load_upload_cache():
    if UPLOAD_CACHE.exists():
        return json.loads(UPLOAD_CACHE.read_text())
    return {}

def _save_upload_cache(cache):
    UPLOAD_CACHE.write_text(json.dumps(cache, indent=2))

def cached_upload(aid: str, cache: dict, max_retries: int = 5) -> UploadedVideo:
    cached_video = None
    with _cache_lock:
        if aid in cache:
            c = cache[aid]
            cached_video = UploadedVideo(annotation_id=aid, label=None,
                                         file_name=c["file_name"], uri=c["uri"])
            if aid in _verified_uploads:
                return cached_video
    if cached_video is not None:
        if can_access_file(cached_video.file_name):
            with _cache_lock:
                _verified_uploads.add(aid)
            return cached_video
        with _cache_lock:
            current = cache.get(aid)
            if current and current.get("file_name") == cached_video.file_name:
                print(f"  [upload-cache] {aid} cached file is inaccessible; re-uploading", flush=True)
                cache.pop(aid, None)
                _save_upload_cache(cache)
                _verified_uploads.discard(aid)
    # Upload outside lock to avoid blocking other threads
    path = VIDEO_DIR / f"{aid}.mp4"
    last_err = None
    for attempt in range(max_retries):
        try:
            v = upload_video(path)
            break
        except Exception as e:
            last_err = e
            wait = min(60.0, 5.0 * (2 ** attempt))
            print(f"  [upload] {aid} attempt {attempt+1}/{max_retries} failed: {e}, retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(f"cached_upload failed after {max_retries} retries: {last_err}")
    with _cache_lock:
        # Double-check in case another thread uploaded same aid
        if aid not in cache:
            cache[aid] = {"file_name": v.file_name, "uri": v.uri}
            _save_upload_cache(cache)
        current = cache[aid]
        _verified_uploads.add(aid)
    return UploadedVideo(annotation_id=aid, label=None,
                         file_name=current["file_name"], uri=current["uri"])


def _extract_denied_file_id(error: Exception) -> str | None:
    match = _DENIED_FILE_RE.search(str(error))
    if not match:
        return None
    return match.group(1)


def _video_refs_file(video: UploadedVideo, file_id: str) -> bool:
    suffix = f"/{file_id}"
    return any(
        str(value or "").endswith(suffix) or str(value or "") == file_id
        for value in (video.file_name, video.uri)
    )


def _refresh_request_videos(target: UploadedVideo,
                            examples: list[UploadedVideo],
                            file_id: str,
                            cache: dict) -> int:
    refreshed = 0
    for video in [*examples, target]:
        if not _video_refs_file(video, file_id):
            continue
        aid = video.annotation_id
        with _cache_lock:
            current = cache.get(aid)
            if current and any(
                str(current.get(key, "")).endswith(f"/{file_id}")
                or str(current.get(key, "")) == file_id
                for key in ("file_name", "uri")
            ):
                print(f"  [upload-cache] {aid} request file {file_id} was rejected; re-uploading", flush=True)
                cache.pop(aid, None)
                _save_upload_cache(cache)
            _verified_uploads.discard(aid)
        replacement = cached_upload(aid, cache)
        video.file_name = replacement.file_name
        video.uri = replacement.uri
        refreshed += 1
    return refreshed

# ── few-shot examples ───────────────────────────────────────────────────────
def load_examples(mode: str, df, cache: dict):
    """Return (examples, example_clinicals) for the given mode."""
    if mode == "zeroshot":
        return [], []

    samples = json.loads(SAMPLES_PATH.read_text())
    if mode == "5shot":
        fs = samples["five_shot"]
        raw = fs["malignant_3"] + fs["benign_2"]
    else:  # 20shot / 20shot_clinical
        raw = samples["all_20"]

    examples, example_clinicals = [], []
    for s in raw:
        ex_aid   = s[ID_COL]
        ex_label = int(s[LABEL_COL])
        v = cached_upload(ex_aid, cache)
        v.label = ex_label
        examples.append(v)
        example_clinicals.append(
            get_clinical_text(ex_aid) if mode == "20shot_clinical" else ""
        )
    return examples, example_clinicals

# ── checkpoint helpers ──────────────────────────────────────────────────────
def load_done(result_path: Path) -> set:
    done = set()
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    if _is_success_record(record):
                        done.add(record["aid"])
                except Exception:
                    pass
    return done

def append_result(result_path: Path, record: dict):
    with _result_lock:
        with open(result_path, "a") as f:
            f.write(json.dumps(record) + "\n")


def normalize_result_suffix(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", raw.strip())
    return cleaned.strip("-_")

# ── worker function (runs in thread) ────────────────────────────────────────
def process_one(aid, label, mode, model_tag, prompt_tag, examples,
                example_clinicals, upload_cache, result_path, rich_prompt,
                temperature, video_sample_fps, counter, counter_lock):
    try:
        target_v = cached_upload(aid, upload_cache)
        target   = UploadedVideo(annotation_id=aid, label=None,
                                 file_name=target_v.file_name, uri=target_v.uri)
        request_examples = [
            UploadedVideo(annotation_id=ex.annotation_id,
                          label=ex.label,
                          file_name=ex.file_name,
                          uri=ex.uri,
                          mime_type=ex.mime_type)
            for ex in examples
        ]
        clin = get_clinical_text(aid) if mode == "20shot_clinical" else ""

        for request_attempt in range(3):
            try:
                result = predict_one(target, examples=request_examples,
                                     model=model_tag,
                                     target_clinical=clin,
                                     example_clinicals=example_clinicals,
                                     rich_prompt=rich_prompt,
                                     temperature=temperature,
                                     video_sample_fps=video_sample_fps)
                break
            except Exception as e:
                denied_file_id = _extract_denied_file_id(e)
                if denied_file_id is None:
                    raise
                refreshed = _refresh_request_videos(target, request_examples,
                                                    denied_file_id, upload_cache)
                if refreshed == 0 or request_attempt == 2:
                    raise
                print(
                    f"  [predict_one] refreshed {refreshed} request file(s) after PERMISSION_DENIED on {denied_file_id}; retrying",
                    flush=True,
                )
        raw = result["text"]
        conf, reasoning = parse_prediction_text(raw)

        record = {"aid": aid, "label": label, "mode": mode,
                  "prompt": prompt_tag, "model": model_tag, "temperature": temperature,
                  "confidence": conf, "reasoning": reasoning}
        append_result(result_path, record)

        with counter_lock:
            counter[0] += 1
            n = counter[0]
        print(f"  [{n}] {aid} conf={conf:.3f}  {reasoning[:70]}", flush=True)
        return True

    except Exception as e:
        print(f"  [ERROR] {aid}: {e}", flush=True)
        return False

# ── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=MODES, required=True)
    parser.add_argument("--model",   type=str, default=None,
                        help="Override GEMINI_MODEL (e.g. gemini-2.0-flash)")
    parser.add_argument("--limit",   type=int, default=917,
                        help="Max API calls this run (daily budget)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent threads (default: 8)")
    parser.add_argument("--sleep",   type=float, default=0.0,
                        help="Sleep between submitting tasks (default: 0)")
    parser.add_argument("--rich-prompt", action="store_true",
                        help="Use SYSTEM_PROMPT_RICH instead of minimal prompt")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 = deterministic)")
    parser.add_argument("--video-sampling-fps", type=float, default=None,
                        help="Optional explicit Gemini video sampling FPS. Use 20 for the ~64-frame ablation on fps20 videos.")
    parser.add_argument("--result-suffix", type=str, default="",
                        help="Optional suffix appended to the result filename, e.g. samplefps20.")
    args = parser.parse_args()

    if args.model:
        os.environ["GEMINI_MODEL"] = args.model

    model_tag   = os.environ.get("GEMINI_MODEL", "unknown").replace("/", "-")
    prompt_tag  = "rich" if args.rich_prompt else "minimal"
    temp_tag    = f"t{args.temperature:.1f}".replace(".", "p")
    suffix_tag  = normalize_result_suffix(args.result_suffix)
    suffix_part = f"_{suffix_tag}" if suffix_tag else ""
    result_path = RESULTS_DIR / f"sync_{args.mode}_{model_tag}_{prompt_tag}_{temp_tag}{suffix_part}.jsonl"
    done        = load_done(result_path)
    print(f"Mode: {args.mode} | Model: {model_tag} | Prompt: {prompt_tag} | Temp: {args.temperature}")
    if args.video_sampling_fps is not None:
        print(f"Explicit Gemini video sampling FPS: {args.video_sampling_fps}")
    print(f"Already done: {len(done)} | Limit: {args.limit} | Workers: {args.workers}")

    df           = load_label_df()
    split        = load_split()
    test_rows    = get_split_rows(df, split, "test")
    upload_cache = _load_upload_cache()

    print("Loading few-shot examples...")
    examples, example_clinicals = load_examples(args.mode, df, upload_cache)
    print(f"  {len(examples)} examples ready.")

    # Build todo list respecting limit
    todo = []
    for _, row in test_rows.iterrows():
        aid   = row[ID_COL]
        label = int(row[LABEL_COL])
        if aid not in done:
            todo.append((aid, label))
        if len(todo) >= args.limit:
            break

    print(f"Submitting {len(todo)} tasks with {args.workers} workers...\n")

    counter      = [0]
    counter_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for aid, label in todo:
            fut = executor.submit(
                process_one,
                aid, label, args.mode, model_tag, prompt_tag,
                examples, example_clinicals, upload_cache,
                result_path, args.rich_prompt, args.temperature,
                args.video_sampling_fps,
                counter, counter_lock
            )
            futures[fut] = aid
            if args.sleep > 0:
                time.sleep(args.sleep)

        for fut in as_completed(futures):
            pass  # results already written inside process_one

    total_done = len(load_done(result_path))
    print(f"\nDone. Total completed for '{args.mode}': {total_done} / {len(test_rows)}")

if __name__ == "__main__":
    main()
