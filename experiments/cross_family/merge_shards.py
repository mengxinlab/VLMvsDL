#!/usr/bin/env python3
"""Merge cross-family hosted-model shard JSONLs.

Put downloaded shard folders under `results/vlm/crossfamily/shards/`, e.g.
`opus48_shard00/`, `opus48_shard01/`. This script concatenates matching
`crossfamily_<model>_<condition>.jsonl` files, drops invalid confidences,
deduplicates by `aid`, and writes merged JSONLs to `results/vlm/crossfamily/`.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARDS = ROOT / "results/vlm/crossfamily/shards"
DEFAULT_OUT = ROOT / "results/vlm/crossfamily"
CONDITIONS = ("image-only", "image-text", "text-only")


def parse_file_name(path: Path) -> tuple[str, str] | None:
    name = path.name
    if not name.startswith("crossfamily_") or not name.endswith(".jsonl"):
        return None
    stem = name[len("crossfamily_"):-len(".jsonl")]
    for cond in CONDITIONS:
        suffix = f"_{cond}"
        if stem.endswith(suffix):
            return stem[:-len(suffix)], cond
    return None


def valid_record(line: str) -> dict | None:
    try:
        record = json.loads(line)
        confidence = float(record.get("confidence", -1))
        if not (0.0 <= confidence <= 1.0):
            return None
        if not record.get("aid"):
            return None
        record["confidence"] = confidence
        return record
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", default=str(DEFAULT_SHARDS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--expected", type=int, default=917)
    args = parser.parse_args()

    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.out_dir)
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    invalid_counts: dict[tuple[str, str], int] = defaultdict(int)
    source_counts: dict[tuple[str, str], int] = defaultdict(int)

    for path in sorted(shards_dir.rglob("crossfamily_*.jsonl")):
        parsed = parse_file_name(path)
        if parsed is None:
            continue
        key = parsed
        for line in path.read_text().splitlines():
            source_counts[key] += 1
            record = valid_record(line)
            if record is None:
                invalid_counts[key] += 1
                continue
            grouped[key][str(record["aid"])] = record

    out_dir.mkdir(parents=True, exist_ok=True)
    for (model, condition), by_aid in sorted(grouped.items()):
        out_path = out_dir / f"crossfamily_{model}_{condition}.jsonl"
        records = [by_aid[aid] for aid in sorted(by_aid)]
        with out_path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        missing = args.expected - len(records)
        print(
            f"{out_path.name}: n={len(records)} missing={missing} "
            f"raw={source_counts[(model, condition)]} invalid={invalid_counts[(model, condition)]}"
        )


if __name__ == "__main__":
    main()
