"""Cross-family metadata triad via Kaggle Benchmarks (kbench) — free quota path.

Runs INSIDE a Kaggle notebook that has `kaggle_benchmarks` available, using the
notebook's model quota (the same one that let you call Opus). Lets us reach
non-Google families (Anthropic/Meta/OpenAI/Qwen, whatever your kbench.llms
exposes) for free instead of paying an API.

It is self-contained (no repo imports) so you can upload it + the tiny frame
bundle as a Kaggle dataset and run it. Input it needs, all uploadable as small
datasets:
  - frames/  : from export_frames_local.py  (917 x 3 PNGs, ~56 MB) + manifest.csv
  - clinical_texts.csv

Output: one JSONL per (model, condition) with the standard schema
{aid,label,condition,model,confidence,reasoning}, same as the rest of the paper,
analysed by analyze_crossfamily.py back in the repo.

The 3 sampled slices are stitched into ONE montage image per case, so the model
call is an unambiguous single-image `llm.prompt(text, image=...)` (avoids any
multi-image API differences). The within-model image-vs-text contrast — the
whole point — is unaffected.

USAGE (Kaggle cell):
    # 1) see what models you can call, paste the list back to me:
    import kaggle_benchmarks as kbench; print(sorted(kbench.llms.keys()))
    # 2) smoke 3 cases on one model, check output, then full run:
    %run run_crossfamily_kbench.py --models anthropic/claude-opus-4.1 --limit 3
    %run run_crossfamily_kbench.py --models anthropic/claude-opus-4.1 meta/llama-4-...

For multi-account quota splitting, use deterministic shards:
    %run run_crossfamily_kbench.py --models anthropic/claude-opus-4.1 --num-shards 3 --shard-index 0
    %run run_crossfamily_kbench.py --models anthropic/claude-opus-4.1 --num-shards 3 --shard-index 1
    %run run_crossfamily_kbench.py --models anthropic/claude-opus-4.1 --num-shards 3 --shard-index 2
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

CONDITIONS = ("image-only", "image-text", "text-only")
STOP_ERROR_PATTERNS = (
    "403",
    "quota",
    "cost",
    "credit",
    "budget",
    "rate limit",
    "rate-limit",
    "too many requests",
    "insufficient",
    "permission",
    "forbidden",
)

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


# ----- clinical text (dual-key: predictions use the trimmed AID) -------------
_clin: dict[str, str] = {}


def load_clinical(clin_csv: Path) -> None:
    df = pd.read_csv(clin_csv)
    for _, row in df.iterrows():
        raw = str(row["AnnotationID"]).strip()
        text = "" if pd.isna(row["clinical_text"]) else str(row["clinical_text"])
        _clin[raw] = text
        _clin["_".join(raw.split("_")[1:])] = text


def montage_b64(frame_paths: list[Path]) -> str:
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    h = max(im.height for im in imgs)
    w = sum(im.width for im in imgs) + 4 * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + 4
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def user_text(condition: str, clinical_text: str) -> str:
    if condition == "text-only":
        body = TEXT_ONLY_NOTE + ("\n" + clinical_text if clinical_text else "")
        return SYSTEM_PROMPT_RICH + "\n\n" + body + "\nClassify this case. Return ONLY the JSON."
    prefix = (clinical_text + "\n") if (condition == "image-text" and clinical_text) else ""
    return SYSTEM_PROMPT_RICH + "\n\n" + prefix + "Classify this case. Return ONLY the JSON."


# ----- the one kbench-specific call; tweak here after the smoke test ----------
def prompt_with_token_cap(model, text: str, max_output_tokens: int, **kwargs) -> str:
    """Call kbench prompt with a small output cap when supported.

    Some providers reserve quota from `max_output_tokens`; GPT-5.5 in kbench can
    fail before generation if the default reservation is large. The target output
    is a tiny JSON object, so 96-128 tokens is enough in practice.
    """
    try:
        return str(model.prompt(text, max_output_tokens=max_output_tokens, **kwargs))
    except TypeError:
        try:
            return str(model.prompt(text, max_tokens=max_output_tokens, **kwargs))
        except TypeError:
            return str(model.prompt(text, **kwargs))


def call_model(kbench, model, text: str, image_b64: str | None, max_output_tokens: int) -> str:
    """Return the model's raw text response.

    Documented kbench image API (cookbook): build an image then either
    `llm.prompt(text, image=img)` or `user.send(img); llm.prompt(text)`.
    We use the single-image kwarg with a montage. If your kbench build wants a
    different shape, this is the only function to adjust.
    """
    if image_b64 is None:
        return prompt_with_token_cap(model, text, max_output_tokens)
    images = kbench.content_types.images
    img = images.from_base64(image_b64, format="png")
    try:
        return prompt_with_token_cap(model, text, max_output_tokens, image=img)
    except TypeError:
        # fallback to the conversational form
        kbench.user.send(img)
        return prompt_with_token_cap(model, text, max_output_tokens)


def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    conf = float(record.get("confidence", -1))
                    if 0.0 <= conf <= 1.0:
                        done.add(str(record["aid"]))
                except Exception:
                    pass
    return done


def should_stop_for_error(exc: Exception | str) -> bool:
    text = str(exc).lower()
    return any(pattern in text for pattern in STOP_ERROR_PATTERNS)


def main() -> None:
    import kaggle_benchmarks as kbench  # only available inside Kaggle

    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="kbench.llms keys, e.g. anthropic/claude-opus-4.1 meta/llama-4-maverick")
    p.add_argument("--frames-dir", default="/kaggle/input/luna25-crossfamily-frames/frames")
    p.add_argument("--clinical-csv", default="/kaggle/input/luna25-clinical-texts/clinical_texts.csv")
    p.add_argument("--condition", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--out-dir", default="/kaggle/working/crossfamily")
    p.add_argument("--limit", type=int, default=100000, help="use 3 for a smoke test")
    p.add_argument("--num-shards", type=int, default=1,
                   help="deterministically split manifest rows across accounts/jobs")
    p.add_argument("--shard-index", type=int, default=0,
                   help="0-based shard index; run all indices 0..num-shards-1 to cover all rows")
    p.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=True,
                   help="stop immediately on quota/cost/rate-limit/API errors, preserving partial JSONL")
    p.add_argument("--max-errors", type=int, default=0,
                   help="maximum non-quota per-case errors before stopping; 0 means no tolerance")
    p.add_argument("--max-output-tokens", type=int, default=96,
                   help="small output cap for the JSON response; lowers provider quota reservation")
    args = p.parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    load_clinical(Path(args.clinical_csv))
    manifest = pd.read_csv(frames_dir / "manifest.csv")

    for model_key in args.models:
        if model_key not in kbench.llms:
            print(f"!! {model_key} not in kbench.llms; available e.g.: {sorted(kbench.llms.keys())[:8]} ...")
            continue
        model = kbench.llms[model_key]
        tag = model_key.replace("/", "_")
        error_count = 0
        for condition in args.condition:
            result_path = out_dir / f"crossfamily_{tag}_{condition}.jsonl"
            done = load_done(result_path)
            rows = [
                r for i, (_, r) in enumerate(manifest.iterrows())
                if i % args.num_shards == args.shard_index
            ]
            todo = [r for r in rows if str(r["aid"]) not in done][: args.limit]
            print(
                f"\n=== {model_key} | {condition} | shard={args.shard_index}/{args.num_shards} "
                f"| done={len(done)} shard_rows={len(rows)} todo={len(todo)} ===",
                flush=True,
            )
            with open(result_path, "a") as handle:
                for k, row in enumerate(todo, start=1):
                    aid = str(row["aid"])
                    label = int(row["label"])
                    n = int(row["n_frames"])
                    clinical_text = _clin.get(aid, "") if condition != "image-only" else ""
                    image_b64 = None
                    if condition != "text-only":
                        fps = [frames_dir / aid / f"{j}.png" for j in range(n)]
                        image_b64 = montage_b64(fps)
                    try:
                        raw = call_model(
                            kbench, model, user_text(condition, clinical_text),
                            image_b64, args.max_output_tokens,
                        )
                        conf, reason = parse_generation(raw)
                    except Exception as exc:  # noqa: BLE001
                        if args.stop_on_error and should_stop_for_error(exc):
                            marker = out_dir / "STOPPED_ON_ERROR.txt"
                            marker.write_text(
                                f"model={model_key}\ncondition={condition}\n"
                                f"aid={aid}\nshard={args.shard_index}/{args.num_shards}\n"
                                f"error={exc}\n"
                            )
                            print(
                                "\n!! STOPPING: quota/cost/rate-limit/API error detected.\n"
                                f"!! Partial JSONL is preserved at: {result_path}\n"
                                f"!! Last attempted AID: {aid}\n"
                                f"!! Error: {exc}\n",
                                flush=True,
                            )
                            sys.exit(2)
                        error_count += 1
                        conf, reason = -1.0, f"ERROR: {exc}"
                        if args.stop_on_error and error_count > args.max_errors:
                            marker = out_dir / "STOPPED_ON_ERROR.txt"
                            marker.write_text(
                                f"model={model_key}\ncondition={condition}\n"
                                f"aid={aid}\nshard={args.shard_index}/{args.num_shards}\n"
                                f"error={exc}\n"
                            )
                            print(
                                "\n!! STOPPING: per-case error limit exceeded.\n"
                                f"!! Partial JSONL is preserved at: {result_path}\n"
                                f"!! Last attempted AID: {aid}\n"
                                f"!! Error: {exc}\n",
                                flush=True,
                            )
                            sys.exit(2)
                    handle.write(json.dumps({
                        "aid": aid, "label": label, "condition": condition, "mode": "zeroshot",
                        "model": tag, "model_key": model_key, "confidence": conf, "reasoning": reason,
                    }) + "\n")
                    handle.flush()
                    print(f"[{len(done)+k}] {aid} conf={conf:.3f} {reason[:80]}", flush=True)

    print("\nDone. Download /kaggle/working/crossfamily/*.jsonl and drop into results/vlm/crossfamily/")


if __name__ == "__main__":
    main()
