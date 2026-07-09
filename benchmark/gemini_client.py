"""Thin wrapper over `google-genai` for video upload and synchronous inference.

Requires:
    pip install google-genai
    Set GEMINI_API_KEY in the environment.

Design notes:
- Uses Files API (`client.files.upload`) for few-shot videos so they can be
  referenced by URI in many requests without re-uploading.
- Provides a synchronous `predict_one()` for ad-hoc use, with retry +
  exponential backoff.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
# NOTE: official model id at this writing is "gemini-3-pro-preview". User may
# override via GEMINI_MODEL env var if Google rebrands to gemini-3.1.

SYSTEM_PROMPT = (
    "You are an expert radiologist. "
    "The CT video is a nodule-centered crop: the lung nodule appears at the center of each frame. "
    "Analyze the video and output your assessment of whether the nodule is malignant or benign. "
    'Return ONLY valid JSON: {"confidence": <float 0-1, probability of malignancy>, '
    '"reasoning": <one sentence>}'
)

SYSTEM_PROMPT_RICH = (
    "You are an expert radiologist specializing in lung cancer CT diagnosis. "
    "Analyze the provided lung CT scan video — each frame is one axial slice, and the nodule is centered in the crop. "
    "Your task: predict whether the lung nodule is MALIGNANT or BENIGN.\n\n"
    "Key malignancy features to assess:\n"
    "1. Size/Volume: nodules >6 mm are higher risk; >15 mm strongly suspicious.\n"
    "2. Density: higher density (less ground-glass) correlates with invasiveness.\n"
    "3. Shape: irregular or lobulated shapes suggest malignancy; round/oval favors benign.\n"
    "4. Margins: spiculated or irregular margins are strongly malignant; smooth margins favor benign.\n"
    "5. Internal features: vacuolization and air bronchograms increase with malignancy.\n"
    "6. Vascular features: vessel penetration and internal vascular thickening indicate malignancy.\n"
    "7. Pleural attachment: pleural indentation or traction suggests invasiveness.\n\n"
    "The CT video is a nodule-centered crop: the lung nodule appears at the center of each frame. "
    "Focus your analysis on the central structure across all slices. "
    'Return ONLY valid JSON: {"confidence": <float 0-1, probability of malignancy>, '
    '"reasoning": <one sentence describing the key feature driving your decision>}'
)

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "confidence": {"type": "NUMBER"},
        "reasoning": {"type": "STRING"},
    },
    "required": ["confidence", "reasoning"],
}


@dataclass
class UploadedVideo:
    annotation_id: str
    label: Optional[int]   # None for unknown / test target
    file_name: str         # e.g. "files/abc123" - what the API expects
    uri: str               # convenience full URI
    mime_type: str = "video/mp4"


def _client():
    from google import genai  # type: ignore

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var not set")
    return genai.Client(api_key=api_key)


def upload_video(path: Path, display_name: Optional[str] = None,
                 wait_active: bool = True, timeout_s: int = 300) -> UploadedVideo:
    """Upload an MP4 to Files API and (optionally) wait until state == ACTIVE."""
    from google.genai import types  # type: ignore

    client = _client()
    f = client.files.upload(file=str(path), config={"mime_type": "video/mp4",
                                                      "display_name": display_name or path.stem})
    if wait_active:
        deadline = time.time() + timeout_s
        while getattr(f, "state", None) and str(f.state).endswith("PROCESSING"):
            if time.time() > deadline:
                raise TimeoutError(f"File {f.name} did not become ACTIVE in {timeout_s}s")
            print(f"  [upload] {f.name} still PROCESSING, waiting...", flush=True)
            time.sleep(2.0)
            f = client.files.get(name=f.name)
        state = str(getattr(f, "state", ""))
        print(f"  [upload] {f.name} state={state}", flush=True)
        if not state.endswith("ACTIVE"):
            raise RuntimeError(f"File {f.name} ended in state={state}")
    return UploadedVideo(
        annotation_id=path.stem,
        label=None,
        file_name=f.name,
        uri=getattr(f, "uri", f.name),
    )


def can_access_file(file_name: str) -> bool:
    """Return whether the current API key can access an uploaded Files API object."""
    client = _client()
    normalized = str(file_name or "").strip()
    if not normalized:
        return False
    if "/" in normalized and not normalized.startswith("files/"):
        normalized = f"files/{normalized.rsplit('/', 1)[-1]}"
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            client.files.get(name=normalized)
            return True
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            if any(token in msg for token in ("403", "404", "permission_denied", "not exist", "not found")):
                return False
            if any(token in msg for token in ("ssl", "eof", "connecterror", "connection", "timeout", "unavailable", "503")) and attempt < 2:
                time.sleep(1.0 + attempt)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return False


def _backoff_sleep(attempt: int, base: float = 2.0, cap: float = 60.0) -> None:
    delay = min(cap, base * (2 ** attempt)) + random.uniform(0, 1)
    logger.info("Rate limited / transient error; sleeping %.1fs", delay)
    time.sleep(delay)


def _video_part(video: UploadedVideo, video_sample_fps: Optional[float] = None) -> Dict[str, Any]:
    part: Dict[str, Any] = {
        "file_data": {"file_uri": video.uri, "mime_type": video.mime_type},
    }
    if video_sample_fps is not None:
        part["video_metadata"] = {"fps": float(video_sample_fps)}
    return part


def _build_contents(target_video: Optional[UploadedVideo],
                    examples: List[UploadedVideo],
                    target_clinical: str = "",
                    example_clinicals: Optional[List[str]] = None,
                    video_sample_fps: Optional[float] = None,
                    include_target_video: bool = True,
                    include_example_videos: bool = True) -> List[Dict[str, Any]]:
    """Build the `contents` list interleaving few-shot examples and target.

    Each example contributes: user(video [+ clinical text]) + model(JSON answer)
    Final turn: user(target video [+ clinical text]) -> model answers.

    Args:
        target_video: the test sample video
        examples: few-shot example videos (with labels)
        target_clinical: pre-generated clinical description for the target (may be "")
        example_clinicals: one clinical string per example (parallel list, may be None/"")
    """
    example_clinicals = example_clinicals or [""] * len(examples)
    contents: List[Dict[str, Any]] = []
    for ex, clin in zip(examples, example_clinicals):
        label_text = "malignant" if ex.label == 1 else "benign"
        user_text = (clin + "\n" if clin else "") + f"Example case. Ground truth label: {label_text}."
        parts: List[Dict[str, Any]] = []
        if include_example_videos:
            parts.append(_video_part(ex, video_sample_fps=video_sample_fps))
        parts.append({"text": user_text})
        contents.append({"role": "user", "parts": parts})
        # Provide an answer turn so the model learns the desired JSON format
        gt_conf = 0.95 if ex.label == 1 else 0.05
        gt_reason = (
            "Ground-truth malignant; likely shows irregular margins or solid component."
            if ex.label == 1
            else "Ground-truth benign; likely smooth margins and stable appearance."
        )
        contents.append({
            "role": "model",
            "parts": [{"text": json.dumps({"confidence": gt_conf, "reasoning": gt_reason})}],
        })
    target_text = (
        (target_clinical + "\n" if target_clinical else "")
        + "Now classify this case. Return ONLY the JSON object."
    )
    parts = []
    if include_target_video:
        if target_video is None:
            raise ValueError("target_video is required when include_target_video=True")
        parts.append(_video_part(target_video, video_sample_fps=video_sample_fps))
    parts.append({"text": target_text})
    contents.append({"role": "user", "parts": parts})
    return contents


def build_request_contents(target_video: Optional[UploadedVideo],
                           examples: Optional[List[UploadedVideo]] = None,
                           target_clinical: str = "",
                           example_clinicals: Optional[List[str]] = None,
                           video_sample_fps: Optional[float] = None,
                           include_target_video: bool = True,
                           include_example_videos: bool = True) -> List[Dict[str, Any]]:
    """Public wrapper for smoke-test request inspection.

    This mirrors the request body used by :func:`predict_one` without calling
    the hosted model. It is intended for control-condition dry runs and request
    dumps.
    """
    return _build_contents(
        target_video,
        examples or [],
        target_clinical=target_clinical,
        example_clinicals=example_clinicals,
        video_sample_fps=video_sample_fps,
        include_target_video=include_target_video,
        include_example_videos=include_example_videos,
    )


def _generation_config(rich_prompt: bool = False, temperature: float = 0.0) -> Dict[str, Any]:
    return {
        "temperature": temperature,
        "response_mime_type": "application/json",
        "response_schema": RESPONSE_SCHEMA,
        "system_instruction": SYSTEM_PROMPT_RICH if rich_prompt else SYSTEM_PROMPT,
    }


def predict_one(target_video: Optional[UploadedVideo],
                examples: Optional[List[UploadedVideo]] = None,
                model: str = DEFAULT_MODEL,
                max_retries: int = 5,
                target_clinical: str = "",
                example_clinicals: Optional[List[str]] = None,
                rich_prompt: bool = False,
                temperature: float = 0.0,
                video_sample_fps: Optional[float] = None,
                include_target_video: bool = True,
                include_example_videos: bool = True) -> Dict[str, Any]:
    """Synchronous single-call inference, with retry/backoff."""
    from google.genai import types  # type: ignore

    client = _client()
    examples = examples or []
    contents = _build_contents(target_video, examples,
                               target_clinical=target_clinical,
                               example_clinicals=example_clinicals,
                               video_sample_fps=video_sample_fps,
                               include_target_video=include_target_video,
                               include_example_videos=include_example_videos)

    cfg = _generation_config(rich_prompt=rich_prompt, temperature=temperature)
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            print(f"  [predict_one] calling API (attempt {attempt+1}/{max_retries})...", flush=True)
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=cfg["temperature"],
                    response_mime_type=cfg["response_mime_type"],
                    response_schema=cfg["response_schema"],
                    system_instruction=cfg["system_instruction"],
                ),
            )
            return {"text": resp.text, "raw": resp}
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            if any(k in msg for k in ("429", "rate", "quota", "503", "unavailable", "timeout", "deadline")):
                delay = min(120.0, 5.0 * (2 ** attempt)) + random.uniform(0, 2)
                print(f"  [predict_one] attempt {attempt+1}/{max_retries} rate-limited, retrying in {delay:.1f}s… ({e})", flush=True)
                time.sleep(delay)
                continue
            raise
    raise RuntimeError(f"predict_one failed after {max_retries} retries: {last_err}")
