from __future__ import annotations

import json
import re
from typing import Any, Optional


_FENCE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def is_success_record(record: dict[str, Any]) -> bool:
    confidence = record.get("confidence")
    try:
        return 0.0 <= float(confidence) <= 1.0
    except (TypeError, ValueError):
        return False


def _strip_markdown_fences(text: str) -> str:
    fenced = _FENCE_BLOCK_RE.fullmatch(text)
    if fenced:
        return fenced.group(1).strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            if lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
            return "\n".join(lines[1:]).strip()

    if text.endswith("```"):
        return text[: text.rfind("```")].strip()

    return text


def _raw_decode_first_object(text: str) -> Optional[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_json_payload(text: str | None) -> Optional[dict[str, Any]]:
    if not text:
        return None

    stripped = text.strip()
    candidates = [stripped]

    unfenced = _strip_markdown_fences(stripped)
    if unfenced and unfenced != stripped:
        candidates.append(unfenced)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = _raw_decode_first_object(candidate)
        if isinstance(parsed, dict):
            return parsed

    return None


def parse_prediction_text(text: str) -> tuple[float, str]:
    parsed = parse_json_payload(text)
    if parsed is None:
        return -1.0, text

    try:
        confidence = float(parsed.get("confidence", -1))
    except (TypeError, ValueError):
        return -1.0, text

    reasoning = parsed.get("reasoning", "")
    if reasoning is None:
        reasoning = ""
    return confidence, str(reasoning)


def repair_result_record(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if is_success_record(record):
        return record, False

    reasoning = record.get("reasoning")
    if not isinstance(reasoning, str):
        return record, False

    confidence, parsed_reasoning = parse_prediction_text(reasoning)
    if not (0.0 <= confidence <= 1.0):
        return record, False

    repaired = dict(record)
    repaired["confidence"] = confidence
    repaired["reasoning"] = parsed_reasoning
    return repaired, True
