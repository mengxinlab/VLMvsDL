"""Hosted-model API runner for the cross-family Z0/Z2/Z3 metadata triad.

Runs on any machine with internet and the exported frame PNGs (see
export_frames.py). It does not need the raw CT data once the sparse frame bundle
has been generated in a private data environment.

Uses the same rich prompt, the same 3 sampled frames, the same clinical text,
and the same three conditions as run_crossfamily_offline.py, and writes the same
prediction JSONL schema consumed by analyze_crossfamily.py.

Examples
--------
    export OPENAI_API_KEY=...
    python experiments/cross_family/run_crossfamily_api.py \
        --provider openai --model openai/gpt-5.5-2026-04-23 --model-name openai_gpt-5.5-2026-04-23 \
        --frames-dir frames --clinical-csv data/metadata/clinical_texts.csv

    export ANTHROPIC_API_KEY=...
    python experiments/cross_family/run_crossfamily_api.py \
        --provider anthropic --model anthropic/claude-opus-4-8@default --model-name anthropic_claude-opus-4-8@default \
        --frames-dir frames --clinical-csv data/metadata/clinical_texts.csv
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "rcf", str(Path(__file__).resolve().parent / "run_crossfamily_offline.py")
)
rcf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcf)

SYSTEM_PROMPT_RICH = rcf.SYSTEM_PROMPT_RICH
TEXT_ONLY_NOTE = rcf.TEXT_ONLY_NOTE
CONDITIONS = rcf.CONDITIONS

# Set in main(): lets the OpenAI-compatible path target OpenRouter / DeepInfra /
# SiliconFlow / native OpenAI with one client.
_BASE_URL: str | None = None
_API_KEY: str | None = None


def b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def frame_paths(frames_dir: Path, aid: str, n: int) -> list[Path]:
    return [frames_dir / aid / f"{j}.png" for j in range(n)]


def user_text(condition: str, clinical_text: str) -> str:
    if condition == "text-only":
        body = TEXT_ONLY_NOTE + ("\n" + clinical_text if clinical_text else "")
        return body + "\nClassify this case. Return ONLY the JSON."
    prefix = (clinical_text + "\n") if (condition == "image-text" and clinical_text) else ""
    return prefix + "Classify this case. Return ONLY the JSON."


def call_openai(model, condition, frames, clinical_text):
    from openai import OpenAI

    client = OpenAI(base_url=_BASE_URL or None, api_key=_API_KEY)
    content = []
    if condition != "text-only":
        for fp in frames:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_png(fp)}"}})
    content.append({"type": "text", "text": user_text(condition, clinical_text)})
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT_RICH},
                  {"role": "user", "content": content}],
    )
    return resp.choices[0].message.content


def call_anthropic(model, condition, frames, clinical_text):
    import anthropic

    client = anthropic.Anthropic(api_key=_API_KEY)
    content = []
    if condition != "text-only":
        for fp in frames:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b64_png(fp)}})
    content.append({"type": "text", "text": user_text(condition, clinical_text)})
    resp = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0.0,
        system=SYSTEM_PROMPT_RICH,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


def predict_with_retry(provider, model, condition, frames, clinical_text, max_retries=5):
    fn = call_openai if provider == "openai" else call_anthropic
    last = None
    for attempt in range(max_retries):
        try:
            text = fn(model, condition, frames, clinical_text)
            return rcf.parse_generation(text or "")
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc).lower()
            if any(k in msg for k in ("429", "rate", "quota", "overloaded", "503", "timeout", "529")):
                time.sleep(min(60.0, 4.0 * (2 ** attempt)) + random.uniform(0, 2))
                continue
            time.sleep(2.0)
    return -1.0, f"FAILED: {last}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = Path(__file__).resolve().parents[2]
    p.add_argument("--provider", choices=("openai", "anthropic"), default="openai",
                   help="'openai' covers any OpenAI-compatible endpoint via --base-url "
                        "(OpenRouter / DeepInfra / SiliconFlow / OpenAI). 'anthropic' = native Claude.")
    p.add_argument("--model", required=True,
                   help="API model id, e.g. qwen/qwen3-vl-30b-a3b-instruct (OpenRouter), z-ai/glm-4.6v, "
                        "openai/gpt-4o, or gpt-4o for native OpenAI.")
    p.add_argument("--model-name", required=True, help="short tag for output files, e.g. qwen3-vl-30b-a3b")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible endpoint, e.g. https://openrouter.ai/api/v1 . Omit for native OpenAI.")
    p.add_argument("--api-key-env", default=None,
                   help="Env var holding the key (default: OPENROUTER_API_KEY if --base-url set, "
                        "else OPENAI_API_KEY / ANTHROPIC_API_KEY).")
    p.add_argument("--frames-dir", default=str(repo / "results/vlm/crossfamily/frames"),
                   help="folder from export_frames_local.py (has manifest.csv)")
    p.add_argument("--clinical-csv", default=str(repo / "data/metadata/clinical_texts.csv"))
    p.add_argument("--condition", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--out-dir", default=str(repo / "results/vlm/crossfamily"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=100000)
    args = p.parse_args()

    global _BASE_URL, _API_KEY
    _BASE_URL = args.base_url
    if args.provider == "anthropic":
        key_env = args.api_key_env or "ANTHROPIC_API_KEY"
    elif args.base_url:
        key_env = args.api_key_env or "OPENROUTER_API_KEY"
    else:
        key_env = args.api_key_env or "OPENAI_API_KEY"
    _API_KEY = os.environ.get(key_env)
    if not _API_KEY:
        raise SystemExit(f"{key_env} not set")
    print(f"provider={args.provider} base_url={_BASE_URL or 'native'} key_env={key_env} model={args.model}")

    frames_dir = Path(args.frames_dir)
    manifest = pd.read_csv(frames_dir / "manifest.csv")
    clin_csv = Path(args.clinical_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition in args.condition:
        result_path = out_dir / f"crossfamily_{args.model_name}_{condition}.jsonl"
        done = rcf.load_done(result_path)
        todo = [r for _, r in manifest.iterrows() if str(r["aid"]) not in done][: args.limit]
        print(f"\n=== {args.model_name} | {condition} | done={len(done)} todo={len(todo)} ===", flush=True)

        def work(row):
            aid = str(row["aid"])
            label = int(row["label"])
            n = int(row["n_frames"])
            clinical_text = rcf.get_clinical_text(aid, clin_csv) if condition != "image-only" else ""
            frames = [] if condition == "text-only" else frame_paths(frames_dir, aid, n)
            conf, reason = predict_with_retry(args.provider, args.model, condition, frames, clinical_text)
            return {"aid": aid, "label": label, "condition": condition, "mode": "zeroshot",
                    "provider": args.provider, "model": args.model_name,
                    "confidence": conf, "reasoning": reason}

        with open(result_path, "a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(work, row): row for row in todo}
            for k, fut in enumerate(as_completed(futures), start=1):
                rec = fut.result()
                handle.write(json.dumps(rec) + "\n")
                handle.flush()
                print(f"[{len(done)+k}/{len(manifest)}] {rec['aid']} conf={rec['confidence']:.3f}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
