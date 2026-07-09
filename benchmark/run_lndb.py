#!/usr/bin/env python3
"""External-validation inference runner on LNDb.

Reuses gemini_client.predict_one + few-shot exemplars from LUNA25.
Modes supported: zeroshot, 20shot (no clinical text on LNDb).
This script writes one prediction per unique LNDb finding; use
evaluate_lndb_external.py to map those predictions onto the DL-matched
814-row lndb_10to1 evaluation sheet.

Results: results/lndb_{mode}_{model}_{prompt}_{temp}.jsonl
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).parent
sys.path.insert(0, str(BENCH))

os.environ.setdefault("GEMINI_MODEL",   "gemini-3-flash-preview")

from data_utils import ID_COL, LABEL_COL
from gemini_client import upload_video, predict_one, UploadedVideo
from paths import CACHE_DIR, LNDB_ROI_DIR, VIDEOS_DIR, VLM_RESULTS_DIR

LNDB_VIDEO_DIR = VIDEOS_DIR / "lndb_fps20"
LNDB_LABEL_CSV = LNDB_ROI_DIR / "lndb_labels.csv"
SAMPLES_PATH   = BENCH / "few_shot_samples.json"
RESULTS_DIR    = VLM_RESULTS_DIR
UPLOAD_CACHE_LUNA = CACHE_DIR / "upload_cache.json"        # for few-shot exemplars (LUNA25)
UPLOAD_CACHE_LNDB = CACHE_DIR / "upload_cache_lndb.json"   # LNDb target videos
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODES = ("zeroshot", "20shot")


def _is_success_record(record: dict) -> bool:
    confidence = record.get("confidence")
    return isinstance(confidence, (int, float)) and 0.0 <= float(confidence) <= 1.0

_cache_lock_lndb = threading.Lock()
_cache_lock_luna = threading.Lock()
_result_lock     = threading.Lock()


def _load_cache(p: Path):
    return json.loads(p.read_text()) if p.exists() else {}


def _save_cache(p: Path, c: dict):
    p.write_text(json.dumps(c, indent=2))


def cached_upload(aid: str, path: Path, cache: dict, lock: threading.Lock,
                  cache_path: Path, max_retries: int = 5) -> UploadedVideo:
    with lock:
        if aid in cache:
            c = cache[aid]
            return UploadedVideo(annotation_id=aid, label=None,
                                 file_name=c["file_name"], uri=c["uri"])
    last_err = None
    for attempt in range(max_retries):
        try:
            v = upload_video(path)
            break
        except Exception as e:
            last_err = e
            wait = min(60.0, 5.0 * (2 ** attempt))
            print(f"  [upload] {aid} attempt {attempt+1}/{max_retries} failed: {e}, retry in {wait:.0f}s", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(f"cached_upload failed: {last_err}")
    with lock:
        if aid not in cache:
            cache[aid] = {"file_name": v.file_name, "uri": v.uri}
            _save_cache(cache_path, cache)
    return v


def load_examples(mode: str, luna_cache: dict):
    if mode == "zeroshot":
        return [], []
    samples = json.loads(SAMPLES_PATH.read_text())
    raw = samples["all_20"]
    examples = []
    luna_video_dir = VIDEOS_DIR / "fps20"
    for s in raw:
        ex_aid   = s[ID_COL]
        ex_label = int(s[LABEL_COL])
        v = cached_upload(ex_aid, luna_video_dir / f"{ex_aid}.mp4",
                          luna_cache, _cache_lock_luna, UPLOAD_CACHE_LUNA)
        v.label = ex_label
        examples.append(v)
    return examples, [""] * len(examples)


def load_done(p: Path) -> set:
    done = set()
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    if _is_success_record(record):
                        done.add(record["aid"])
                except Exception:
                    pass
    return done


def append_result(p: Path, record: dict):
    with _result_lock:
        with open(p, "a") as f:
            f.write(json.dumps(record) + "\n")


def process_one(fid, label, mode, model_tag, prompt_tag, examples,
                example_clinicals, lndb_cache, result_path, rich_prompt,
                temperature, counter, counter_lock):
    try:
        target_v = cached_upload(fid, LNDB_VIDEO_DIR / f"{fid}.mp4",
                                 lndb_cache, _cache_lock_lndb, UPLOAD_CACHE_LNDB)
        target = UploadedVideo(annotation_id=fid, label=None,
                               file_name=target_v.file_name, uri=target_v.uri)
        result = predict_one(target, examples=examples,
                             model=model_tag,
                             target_clinical="",
                             example_clinicals=example_clinicals,
                             rich_prompt=rich_prompt,
                             temperature=temperature)
        raw = result["text"]
        try:
            parsed    = json.loads(raw)
            conf      = float(parsed.get("confidence", -1))
            reasoning = parsed.get("reasoning", "")
        except Exception:
            conf, reasoning = -1.0, raw

        record = {"aid": fid, "label": label, "mode": mode,
                  "prompt": prompt_tag, "model": model_tag,
                  "temperature": temperature,
                  "confidence": conf, "reasoning": reasoning,
                  "dataset": "lndb"}
        append_result(result_path, record)
        with counter_lock:
            counter[0] += 1
            n = counter[0]
        print(f"  [{n}] {fid} conf={conf:.3f}  {reasoning[:70]}", flush=True)
        return True
    except Exception as e:
        print(f"  [ERROR] {fid}: {e}", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=MODES, required=True)
    parser.add_argument("--model",   type=str, default=None)
    parser.add_argument("--limit",   type=int, default=10000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sleep",   type=float, default=0.0)
    parser.add_argument("--rich-prompt", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if args.model:
        os.environ["GEMINI_MODEL"] = args.model

    model_tag   = os.environ.get("GEMINI_MODEL", "unknown").replace("/", "-")
    prompt_tag  = "rich" if args.rich_prompt else "minimal"
    temp_tag    = f"t{args.temperature:.1f}".replace(".", "p")
    result_path = RESULTS_DIR / f"lndb_{args.mode}_{model_tag}_{prompt_tag}_{temp_tag}.jsonl"
    done        = load_done(result_path)
    print(f"[LNDb] Mode={args.mode} | Model={model_tag} | Prompt={prompt_tag} | Temp={args.temperature}")
    print(f"Already done: {len(done)} | Limit: {args.limit} | Workers: {args.workers}")

    df = pd.read_csv(LNDB_LABEL_CSV)
    df = df.drop_duplicates("FindingID").reset_index(drop=True)
    print(f"LNDb test set: {len(df)} unique findings ({(df['label']==1).sum()} malig / {(df['label']==0).sum()} benign)")

    luna_cache = _load_cache(UPLOAD_CACHE_LUNA)
    lndb_cache = _load_cache(UPLOAD_CACHE_LNDB)

    print("Loading few-shot examples (LUNA25 exemplars)...")
    examples, example_clinicals = load_examples(args.mode, luna_cache)
    print(f"  {len(examples)} examples ready.")

    todo = []
    for _, row in df.iterrows():
        fid   = str(row["FindingID"])
        label = int(row["label"])
        if fid not in done:
            todo.append((fid, label))
        if len(todo) >= args.limit:
            break

    print(f"Submitting {len(todo)} tasks with {args.workers} workers...\n")
    counter = [0]
    counter_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for fid, label in todo:
            f = ex.submit(process_one, fid, label, args.mode, model_tag, prompt_tag,
                          examples, example_clinicals, lndb_cache, result_path,
                          args.rich_prompt, args.temperature, counter, counter_lock)
            futs[f] = fid
            if args.sleep > 0:
                time.sleep(args.sleep)
        for f in as_completed(futs):
            pass

    total = len(load_done(result_path))
    print(f"\nDone. Total completed for LNDb '{args.mode}': {total} / {len(df)}")


if __name__ == "__main__":
    main()
