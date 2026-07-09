"""Cross-family open-weight VLM inference for the metadata-reliance audit.

Purpose
-------
Reviewer concern at npj DM: the metadata-reliance finding was Gemini-only and
therefore "too narrow to judge generalizability." This script replicates the
core *metadata-control contrast* on additional open-weight VLM families
(Qwen2.5-VL, InternVL3, GLM-4.5V, ...) so the same pattern can be shown to hold
across model lineages.

It deliberately mirrors the hosted-model audit schema (same frame sampling,
same clinical-text dual-key handling, same resume/lock logic), so it slots into
the existing analysis pipeline with no new output format.

Design choices (kept identical across all cross-family models for comparability)
--------------------------------------------------------------------------------
* Zero-shot, not 20-shot. The Gemini ablation already showed the metadata effect
  is present zero-shot (Z3 ~= F3), so the cross-family test uses the lighter
  zero-shot regime that fits every model's context window.
* Three conditions form the metadata-control triad:
    image-only : rich prompt + sampled CT slices,        no clinical text
    image-text : rich prompt + sampled CT slices +       structured clinical text
    text-only  : rich prompt +                           structured clinical text (no slices)
  The key within-model readout is AUC(image-text) - AUC(image-only) (the metadata
  lift) and how much of image-text is recovered by text-only.
* Same 3-frame sampling as the hosted Gemini video API (indices [9,29,49] of 64
  via the 1-fps-of-20-fps "gemini" anchor), passed as multiple still images,
  which every open VLM family ingests.

Output
------
One JSONL per (model, condition) with the existing prediction schema
{aid, label, condition, model, confidence, reasoning, ...}, consumable by
experiments/cross_family/analyze_crossfamily.py.

Usage (on a private GPU machine with local model snapshots)
----------------------------------------------------------
    python run_crossfamily_offline.py \
        --model-path /path/to/qwen2.5-vl-72b-instruct-snapshot \
        --model-name qwen2.5-vl-72b \
        --condition image-only image-text text-only \
        --load-in-4bit

Run all three conditions for one model in a single invocation (the model is
loaded once). Repeat with a different --model-path/--model-name per family.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from PIL import Image

ID_COL = "AnnotationID"
LABEL_COL = "label"
PATIENT_COL = "PatientID"
ROOT = Path(__file__).resolve().parents[2]

CONDITIONS = ("image-only", "image-text", "text-only")

# Local data paths (override via env or CLI).
DEFAULT_IMG_DIR = os.environ.get("IMG_DIR", str(ROOT / "data/raw/luna25/image"))
DEFAULT_CSV_PATH = os.environ.get(
    "CSV_PATH", str(ROOT / "data/metadata/luna25_public_training_development_data.csv")
)
DEFAULT_SPLIT_PATH = os.environ.get("SPLIT_PATH", str(ROOT / "data/metadata/patient_split.json"))
DEFAULT_CLIN_CSV = os.environ.get("CLIN_CSV", str(ROOT / "data/metadata/clinical_texts.csv"))
DEFAULT_OUT_DIR = os.environ.get("OUT_DIR", str(ROOT / "results/vlm/crossfamily"))

# Single rich prompt shared by every cross-family model and every condition, so
# the only thing that varies within a model is the presence of image / text.
# Multi-image phrasing (the cross-family input is sampled still slices).
SYSTEM_PROMPT_RICH = (
    "You are an expert radiologist specializing in lung cancer CT diagnosis. "
    "You are shown one or more axial CT slices of a lung nodule; the nodule is "
    "centered in each slice. Predict whether the nodule is MALIGNANT or BENIGN.\n\n"
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

_clin_cache: dict[str, str] = {}
_clin_loaded = False


# --------------------------------------------------------------------------- #
# Data / frame handling (copied from the MedGemma offline runner for parity)
# --------------------------------------------------------------------------- #
def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@contextmanager
def exclusive_run_lock(result_path: Path):
    lock_path = result_path.with_suffix(result_path.suffix + ".lock")
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another run is already writing to {result_path.name}. "
            f"If no job is running, delete the stale lock: {lock_path}"
        ) from exc
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def load_done(result_path: Path) -> set[str]:
    done: set[str] = set()
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["aid"]))
            except Exception:
                pass
    return done


def _load_clinical_texts(clin_csv: Path) -> None:
    global _clin_loaded
    if _clin_loaded:
        return
    _clin_loaded = True
    if not clin_csv.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {clin_csv}")
    df = pd.read_csv(clin_csv)
    if ID_COL not in df.columns or "clinical_text" not in df.columns:
        raise KeyError(f"Clinical CSV must contain {ID_COL} and clinical_text: {clin_csv}")
    for _, row in df.iterrows():
        raw = str(row[ID_COL]).strip()
        text = "" if pd.isna(row["clinical_text"]) else str(row["clinical_text"])
        _clin_cache[raw] = text
        # predictions key on the trimmed AID (drop the duplicated patient prefix)
        trimmed = "_".join(raw.split("_")[1:])
        _clin_cache[trimmed] = text


def get_clinical_text(aid: str, clin_csv: Path) -> str:
    _load_clinical_texts(clin_csv)
    return _clin_cache.get(str(aid), "")


def load_test_rows(csv_path: Path, split_path: Path, img_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    with open(split_path) as handle:
        split = json.load(handle)
    test_pids = {str(pid) for pid in split["test"]}
    test_df = df[df[PATIENT_COL].astype(str).isin(test_pids)].copy()
    test_df = test_df[
        test_df[ID_COL].apply(lambda aid: (img_dir / f"{aid}.npy").exists())
    ].reset_index(drop=True)
    return test_df


def sample_slice_indices(
    depth: int,
    n_frames: int | None = None,
    source_video_fps: float | None = None,
    sample_fps: float | None = None,
) -> np.ndarray:
    """Match the hosted Gemini video sampling: 1 fps of a 20 fps clip over 64
    frames -> indices [9, 29, 49] (the 'gemini' window-center anchor)."""
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    if sample_fps is not None:
        if source_video_fps is None or source_video_fps <= 0:
            raise ValueError("source_video_fps must be > 0 when sample_fps is set")
        frames_per_window = float(source_video_fps) / float(sample_fps)
        complete_windows = int(np.floor(depth / frames_per_window + 1e-9))
        if complete_windows <= 0:
            return np.array([max(0, min(depth - 1, (depth - 1) // 2))], dtype=int)
        starts = np.arange(complete_windows, dtype=np.float64) * frames_per_window
        indices = np.floor(starts + (frames_per_window - 1.0) * 0.5 + 1e-9).astype(int)
        indices = np.unique(np.clip(indices, 0, depth - 1))
        return indices if indices.size else np.array([0], dtype=int)
    if n_frames is None or n_frames <= 0 or n_frames >= depth:
        return np.arange(depth, dtype=int)
    return np.linspace(0, depth - 1, n_frames, dtype=int)


def npy_to_pil(
    aid: str,
    img_dir: Path,
    n_frames: int | None,
    source_video_fps: float | None = None,
    sample_fps: float | None = None,
) -> list[Image.Image]:
    path = img_dir / f"{aid}.npy"
    if not path.exists():
        raise FileNotFoundError(f"NPY not found: {path}")
    volume = np.load(str(path)).astype(np.float32)
    volume = np.clip(volume, -1000, 400)
    volume = ((volume + 1000) / 1400 * 255).astype(np.uint8)
    indices = sample_slice_indices(
        volume.shape[0], n_frames=n_frames,
        source_video_fps=source_video_fps, sample_fps=sample_fps,
    )
    return [Image.fromarray(volume[i]).convert("RGB") for i in indices]


# --------------------------------------------------------------------------- #
# Message construction (unified transformers chat-template content format)
# --------------------------------------------------------------------------- #
def build_messages(condition: str, frames: list[Image.Image], clinical_text: str):
    """Return a chat-format message list with inline content parts.

    Uses {'type': 'image', 'image': <PIL>} and {'type': 'text', 'text': ...},
    which transformers' AutoProcessor.apply_chat_template resolves for
    Qwen2.5-VL, InternVL3, GLM-4.x-V and other modern VLMs.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    system_part = {"type": "text", "text": SYSTEM_PROMPT_RICH}

    user_parts: list[dict] = []
    if condition == "text-only":
        user_parts.append({"type": "text", "text": TEXT_ONLY_NOTE})
        if clinical_text:
            user_parts.append({"type": "text", "text": clinical_text})
        user_parts.append({"type": "text", "text": "Classify this case. Return ONLY the JSON."})
    else:
        for frame in frames:
            user_parts.append({"type": "image", "image": frame})
        if condition == "image-text" and clinical_text:
            user_parts.append({"type": "text", "text": clinical_text})
        user_parts.append({"type": "text", "text": "Classify this case. Return ONLY the JSON."})

    return [
        {"role": "system", "content": [system_part]},
        {"role": "user", "content": user_parts},
    ]


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


# --------------------------------------------------------------------------- #
# Model loading + inference (generic across HF image-text-to-text VLMs)
# --------------------------------------------------------------------------- #
def resolve_dtype() -> torch.dtype:
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def resolve_model_dir(root: str) -> Path:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Model path not found: {root_path}")
    if (root_path / "config.json").exists():
        return root_path
    candidates = []
    for cfg in root_path.rglob("config.json"):
        parent = cfg.parent
        if any(parent.glob("*.safetensors")) or (parent / "model.safetensors.index.json").exists():
            candidates.append(parent)
    if not candidates:
        raise FileNotFoundError(f"No HF snapshot (config.json + safetensors) under {root_path}")
    return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]


def load_model(model_path: Path, load_in_4bit: bool, attn_impl: str | None):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = resolve_dtype()
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True
    )
    kwargs: dict = dict(
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        dtype=dtype,
    )
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    if load_in_4bit:
        missing = [n for n in ("accelerate", "bitsandbytes") if not module_available(n)]
        if missing:
            raise RuntimeError("4-bit requested but missing: " + ", ".join(missing))
        from transformers import BitsAndBytesConfig

        kwargs.pop("dtype", None)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForImageTextToText.from_pretrained(str(model_path), **kwargs)
    model.eval()
    return model, processor, dtype


def model_device(model) -> torch.device:
    if getattr(model, "hf_device_map", None):
        for mapped in model.hf_device_map.values():
            if isinstance(mapped, str) and mapped not in ("cpu", "disk"):
                return torch.device(mapped)
    return next(model.parameters()).device


def predict_one(model, processor, messages, max_new_tokens: int) -> tuple[float, str, str]:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model_device(model))
    pad_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id
        )
    input_len = inputs["input_ids"].shape[-1]
    raw_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    confidence, reasoning = parse_generation(raw_text)
    del inputs, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return confidence, reasoning, raw_text


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def result_path_for(out_dir: Path, model_name: str, condition: str) -> Path:
    tag = model_name.replace("/", "_").replace(" ", "_")
    return out_dir / f"crossfamily_{tag}_{condition}.jsonl"


def run_condition(args, model, processor, condition, test_rows, img_dir, clin_csv):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_path_for(out_dir, args.model_name, condition)
    done = load_done(result_path)
    remaining = [r for _, r in test_rows.iterrows() if str(r[ID_COL]) not in done][: args.limit]
    print(
        f"\n=== condition={condition} | model={args.model_name} | file={result_path.name} ===\n"
        f"done={len(done)} total={len(test_rows)} remaining={len(remaining)}",
        flush=True,
    )

    with exclusive_run_lock(result_path):
        for i, row in enumerate(remaining, start=1):
            aid = str(row[ID_COL])
            label = int(row[LABEL_COL])
            try:
                clinical_text = get_clinical_text(aid, clin_csv) if condition != "image-only" else ""
                frames = (
                    []
                    if condition == "text-only"
                    else npy_to_pil(
                        aid, img_dir, args.n_frames,
                        source_video_fps=args.source_video_fps, sample_fps=args.sample_fps,
                    )
                )
                messages = build_messages(condition, frames, clinical_text)
                confidence, reasoning, _ = predict_one(model, processor, messages, args.max_new_tokens)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                confidence, reasoning = -1.0, f"CUDA OOM: {exc}"
            except Exception as exc:  # noqa: BLE001
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                confidence, reasoning = -1.0, str(exc)

            record = {
                "aid": aid,
                "label": label,
                "condition": condition,
                "mode": "zeroshot",
                "model": args.model_name,
                "model_path": str(args.model_path),
                "n_frames": 0 if condition == "text-only" else (
                    args.n_frames if args.sample_fps is None else None
                ),
                "sample_fps": args.sample_fps,
                "source_video_fps": args.source_video_fps,
                "load_in_4bit": args.load_in_4bit,
                "confidence": confidence,
                "reasoning": reasoning,
            }
            with open(result_path, "a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"[{len(done)+i}/{len(test_rows)}] {aid} conf={confidence:.3f} {reasoning[:90]}", flush=True)

    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="Local HF snapshot directory.")
    parser.add_argument("--model-name", required=True, help="Short tag for output files, e.g. qwen2.5-vl-72b.")
    parser.add_argument("--condition", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--split-path", default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--clinical-csv", default=DEFAULT_CLIN_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    # Default sampling reproduces the hosted Gemini video API frames [9,29,49].
    parser.add_argument("--n-frames", type=int, default=3, help="Uniform N frames if --sample-fps unset.")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Sample fps (set with --source-video-fps).")
    parser.add_argument("--source-video-fps", type=float, default=20.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--load-in-4bit", action="store_true", help="Recommended for 72B/106B on a single GPU.")
    parser.add_argument("--attn-impl", default=None, help="e.g. flash_attention_2 or sdpa.")
    parser.add_argument("--limit", type=int, default=100000)
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    clin_csv = Path(args.clinical_csv)
    model_root = resolve_model_dir(args.model_path)
    for path in [img_dir, Path(args.csv_path), Path(args.split_path), clin_csv, model_root]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    test_rows = load_test_rows(Path(args.csv_path), Path(args.split_path), img_dir)
    print(f"Model dir: {model_root} | test samples: {len(test_rows)} | conditions: {args.condition}")

    model, processor, dtype = load_model(model_root, args.load_in_4bit, args.attn_impl)
    print(f"Loaded {args.model_name} | dtype={dtype} | 4bit={args.load_in_4bit}", flush=True)

    for condition in args.condition:
        run_condition(args, model, processor, condition, test_rows, img_dir, clin_csv)

    print("\nAll requested conditions complete.")


if __name__ == "__main__":
    main()
