"""Local MedGemma inference client using HuggingFace Transformers.

Mirrors the interface of gemini_client.predict_one() so run_medgemma.py
can share the same result format / JSONL schema.

Input: .npy CT crops  (D, H, W) int16 HU values — read directly, no video needed.
Representative axial slices are extracted and passed as PIL images to the processor.

Supports:
  - zeroshot         : npy slices only
  - zeroshot_clinical: npy slices + clinical text
  - 20shot_clinical  : 20 few-shot examples (npy) + clinical text

Usage:
    from medgemma_client import load_model, predict_one_local
    load_model("google/medgemma-1.5-4b-it")
    result = predict_one_local(aid, mode="zeroshot_clinical", clinical_text="...")
"""
from __future__ import annotations

import json as _json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from paths import LUNA25_IMAGE_DIR

# ── paths ──────────────────────────────────────────────────────────────────
BENCH     = Path(__file__).parent

# Set MEDGEMMA_IMAGE_DIR and MEDGEMMA_MODEL_PATH env vars, or fall back to the
# local repo layout / HuggingFace model id.
import os as _os
IMAGE_DIR  = Path(_os.environ.get("MEDGEMMA_IMAGE_DIR",
                                  str(LUNA25_IMAGE_DIR)))
MODEL_PATH = _os.environ.get("MEDGEMMA_MODEL_PATH", "google/medgemma-1.5-4b-it")

# ── prompt templates ────────────────────────────────────────────────────────
SYSTEM_PROMPT_RICH = (
    "You are an expert radiologist specializing in lung cancer CT diagnosis. "
    "You are shown axial CT slices of a lung nodule (centered in each image). "
    "Predict whether the nodule is MALIGNANT or BENIGN.\n\n"
    "Key features: size (>15 mm suspicious), density (solid > GGO), "
    "margins (spiculated/lobulated = malignant, smooth = benign), "
    "shape (irregular = malignant), pleural attachment, vascular involvement.\n\n"
    'Return ONLY valid JSON: {"confidence": <float 0-1, probability of malignancy>, '
    '"reasoning": <one sentence>}'
)

SYSTEM_PROMPT_MINIMAL = (
    "You are an expert radiologist. "
    "Analyze the CT slice and classify the lung nodule as malignant or benign. "
    'Return ONLY valid JSON: {"confidence": <float 0-1>, "reasoning": <one sentence>}'
)

# ── global model state ──────────────────────────────────────────────────────
_model     = None
_processor = None
_model_id  = None


def load_model(model_id: str = None) -> None:
    """Load model + processor once; subsequent calls are no-ops."""
    global _model, _processor, _model_id
    if model_id is None:
        model_id = MODEL_PATH
    if _model is not None and _model_id == model_id:
        return

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading {model_id} ...", flush=True)

    if torch.backends.mps.is_available():
        device_map = {"": "mps"}
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.bfloat16
    else:
        device_map = {"": "cpu"}
        dtype = torch.float32

    _processor = AutoProcessor.from_pretrained(model_id)
    _model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device_map
    )
    _model.eval()
    _model_id = model_id
    print(f"  loaded on {device_map}", flush=True)


# ── npy → PIL slices ────────────────────────────────────────────────────────

def sample_slice_indices(
    depth: int,
    n_frames: Optional[int] = None,
    source_video_fps: Optional[float] = None,
    sample_fps: Optional[float] = None,
    frame_anchor: str = "gemini",
) -> np.ndarray:
    """Return axial slice indices using either count-based or FPS-based sampling.

    When ``sample_fps`` is provided, sampling follows the synthetic video timeline
    implied by ``source_video_fps``. This makes MedGemma frame selection explicit
    instead of relying on an opaque video backend default.

    ``frame_anchor="gemini"`` matches the empirical Gemini 3 Flash default-video
    probe used in this repo: use complete sample windows only and take the
    middle-left frame of each window. For 64 slices rendered as 20fps video and
    sampled at 1fps, this returns [9, 29, 49].
    """
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")

    if sample_fps is not None:
        if sample_fps <= 0:
            raise ValueError(f"sample_fps must be > 0, got {sample_fps}")
        if source_video_fps is None or source_video_fps <= 0:
            raise ValueError(
                "source_video_fps must be > 0 when sample_fps is provided"
            )
        if frame_anchor not in {"start", "middle", "gemini"}:
            raise ValueError(
                f"frame_anchor must be 'start', 'middle', or 'gemini', got {frame_anchor!r}"
            )

        source_video_fps = float(source_video_fps)
        sample_fps = float(sample_fps)

        if frame_anchor == "gemini":
            frames_per_window = source_video_fps / sample_fps
            complete_windows = int(np.floor(depth / frames_per_window + 1e-9))
            if complete_windows <= 0:
                fallback = max(0, min(depth - 1, int(np.floor((depth - 1) / 2))))
                return np.array([fallback], dtype=int)
            starts = np.arange(complete_windows, dtype=np.float64) * frames_per_window
            indices = np.floor(
                starts + (frames_per_window - 1.0) * 0.5 + 1e-9
            ).astype(int)
        else:
            duration = depth / source_video_fps
            step = 1.0 / sample_fps
            first_offset = 0.0 if frame_anchor == "start" else 0.5 * step
            sample_times = np.arange(first_offset, duration, step, dtype=np.float64)
            if sample_times.size == 0:
                midpoint = min(max(duration * 0.5, 0.0), (depth - 1) / source_video_fps)
                sample_times = np.array([midpoint], dtype=np.float64)
            indices = np.floor(sample_times * source_video_fps + 1e-9).astype(int)

        indices = np.clip(indices, 0, depth - 1)
        indices = np.unique(indices)
        if indices.size == 0:
            return np.array([0], dtype=int)
        return indices

    if n_frames is None or n_frames >= depth:
        return np.arange(depth, dtype=int)
    if n_frames <= 0:
        raise ValueError(f"n_frames must be > 0 when provided, got {n_frames}")
    return np.linspace(0, depth - 1, n_frames, dtype=int)


def _npy_to_pil(
    aid: str,
    n_frames: int = None,
    source_video_fps: Optional[float] = None,
    sample_fps: Optional[float] = None,
    frame_anchor: str = "gemini",
) -> List[Image.Image]:
    """Load (D,H,W) int16 HU npy, lung-window, return sampled axial PIL slices.

    ``sample_fps`` overrides count-based ``n_frames`` and samples on the synthetic
    timeline implied by ``source_video_fps``.
    """
    path = IMAGE_DIR / f"{aid}.npy"
    if not path.exists():
        raise FileNotFoundError(f"NPY not found: {path}")

    vol = np.load(str(path)).astype(np.float32)          # (D, H, W)
    vol = np.clip(vol, -1000, 400)                        # lung window
    vol = ((vol + 1000) / 1400 * 255).astype(np.uint8)  # → [0,255]

    D = vol.shape[0]
    idxs = sample_slice_indices(
        D,
        n_frames=n_frames,
        source_video_fps=source_video_fps,
        sample_fps=sample_fps,
        frame_anchor=frame_anchor,
    )
    return [Image.fromarray(vol[i], mode="L").convert("RGB") for i in idxs]


# ── build conversation ───────────────────────────────────────────────────────

def _build_messages(
    aid: str,
    mode: str,
    clinical_text: str = "",
    examples: Optional[List[Dict]] = None,
    rich_prompt: bool = True,
    n_frames: Optional[int] = None,
    example_n_frames: Optional[int] = None,
    source_video_fps: Optional[float] = None,
    sample_fps: Optional[float] = None,
    frame_anchor: str = "gemini",
) -> Tuple[List[dict], List[Image.Image]]:
    """Return (messages_list, images_list).

    Images are listed in the order they are referenced by <image> placeholders.
    """
    system_text = SYSTEM_PROMPT_RICH if rich_prompt else SYSTEM_PROMPT_MINIMAL
    examples = examples or []
    messages: List[dict] = []
    all_images: List[Image.Image] = []

    # ── system turn ──────────────────────────────────────────────────────────
    messages.append({"role": "user", "content": [{"type": "text", "text": system_text}]})
    messages.append({"role": "assistant", "content": [
        {"type": "text", "text": "Understood. I will analyze the CT slices and return only JSON."}
    ]})

    # ── few-shot examples ────────────────────────────────────────────────────
    for ex in examples:
        try:
            frames = _npy_to_pil(
                ex["aid"],
                n_frames=example_n_frames if example_n_frames is not None else n_frames,
                source_video_fps=None if example_n_frames is not None else source_video_fps,
                sample_fps=None if example_n_frames is not None else sample_fps,
                frame_anchor=frame_anchor,
            )
        except FileNotFoundError:
            raise RuntimeError(f"Missing few-shot example volume: {ex['aid']}") from None
        all_images.extend(frames)
        label_text = "malignant" if ex["label"] == 1 else "benign"
        gt_conf    = 0.95 if ex["label"] == 1 else 0.05
        gt_reason  = (
            "Irregular margins and solid component indicate malignancy."
            if ex["label"] == 1 else
            "Smooth margins and stable appearance indicate benign."
        )
        user_parts = [{"type": "image"} for _ in frames]
        if ex.get("clinical_text"):
            user_parts.append({"type": "text", "text": ex["clinical_text"]})
        user_parts.append({"type": "text", "text": f"Example. Ground truth: {label_text}."})
        messages.append({"role": "user", "content": user_parts})
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": f'{{"confidence": {gt_conf}, "reasoning": "{gt_reason}"}}'}
        ]})

    # ── target case ──────────────────────────────────────────────────────────
    frames = _npy_to_pil(
        aid,
        n_frames=n_frames,
        source_video_fps=source_video_fps,
        sample_fps=sample_fps,
        frame_anchor=frame_anchor,
    )
    all_images.extend(frames)
    target_parts = [{"type": "image"} for _ in frames]
    if clinical_text and mode in ("zeroshot_clinical", "20shot_clinical"):
        target_parts.append({"type": "text", "text": clinical_text})
    target_parts.append({"type": "text", "text": "Classify this case. Return ONLY the JSON."})
    messages.append({"role": "user", "content": target_parts})

    return messages, all_images


# ── inference ────────────────────────────────────────────────────────────────

def predict_one_local(
    aid: str,
    mode: str = "zeroshot",
    clinical_text: str = "",
    examples: Optional[List[Dict]] = None,
    rich_prompt: bool = True,
    n_frames: Optional[int] = None,
    example_n_frames: Optional[int] = None,
    source_video_fps: Optional[float] = None,
    sample_fps: Optional[float] = None,
    frame_anchor: str = "gemini",
    max_new_tokens: int = 48,
) -> Dict:
    """Run MedGemma inference for one annotation ID.

    Returns {"text": raw_str, "confidence": float, "reasoning": str}
    """
    if _model is None or _processor is None:
        raise RuntimeError("Call load_model() first")

    import torch

    messages, images = _build_messages(
        aid, mode, clinical_text=clinical_text,
        examples=examples,
        rich_prompt=rich_prompt,
        n_frames=n_frames,
        example_n_frames=example_n_frames,
        source_video_fps=source_video_fps,
        sample_fps=sample_fps,
        frame_anchor=frame_anchor,
    )

    prompt = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _processor(text=prompt, images=images or None, return_tensors="pt")

    device = next(_model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.inference_mode():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
        )

    input_len = inputs["input_ids"].shape[-1]
    raw_text = _processor.decode(out[0][input_len:], skip_special_tokens=True).strip()

    conf, reasoning = -1.0, raw_text
    try:
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
        parsed = _json.loads(fence.group(1) if fence else raw_text)
        conf      = float(parsed.get("confidence", -1))
        reasoning = parsed.get("reasoning", "")
    except Exception:
        pass

    return {"text": raw_text, "confidence": conf, "reasoning": reasoning}
