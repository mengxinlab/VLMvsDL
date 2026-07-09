#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from data_utils import ID_COL, get_split_rows, load_label_df, load_split


def is_valid_confidence(value: Any) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= value <= 1.0


def expected_test_order() -> list[str]:
    split = load_split()
    df = load_label_df()
    rows = get_split_rows(df, split, "test")
    return [str(aid) for aid in rows[ID_COL].astype(str)]


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge MedGemma shard JSONLs into one validated result file.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing shard JSONLs.")
    parser.add_argument("--pattern", required=True, help="Glob pattern for shard files.")
    parser.add_argument("--output", type=Path, required=True, help="Output merged JSONL path.")
    args = parser.parse_args()

    shard_paths = sorted(args.input_dir.glob(args.pattern))
    if not shard_paths:
        raise SystemExit(f"No files matched {args.pattern!r} in {args.input_dir}")

    all_rows: list[tuple[str, int, dict[str, Any]]] = []
    for shard_path in shard_paths:
        rows = load_records(shard_path)
        print(f"{shard_path.name}: {len(rows)} lines")
        for lineno, row in enumerate(rows, 1):
            all_rows.append((shard_path.name, lineno, row))

    seen: dict[str, dict[str, Any]] = {}
    duplicate_hits: list[tuple[str, str, int]] = []
    invalid_hits: list[tuple[str, str, int]] = []
    raw_aids: list[str] = []

    for shard_name, lineno, row in all_rows:
        aid = str(row.get("aid", ""))
        if not aid:
            raise SystemExit(f"{shard_name}:{lineno}: missing aid")
        raw_aids.append(aid)
        if aid in seen:
            duplicate_hits.append((aid, shard_name, lineno))
        seen[aid] = row
        if not is_valid_confidence(row.get("confidence")):
            invalid_hits.append((aid, shard_name, lineno))

    expected = expected_test_order()
    expected_set = set(expected)
    merged_aids = set(seen)
    missing = [aid for aid in expected if aid not in merged_aids]
    extra = sorted(merged_aids - expected_set)

    if duplicate_hits:
        dup_counts = Counter(aid for aid, _, _ in duplicate_hits)
        dup_msg = ", ".join(f"{aid} (+{count})" for aid, count in sorted(dup_counts.items()))
        raise SystemExit(f"Found duplicate aids across shards: {dup_msg}")
    if invalid_hits:
        bad_msg = ", ".join(f"{aid}@{shard}:{lineno}" for aid, shard, lineno in invalid_hits[:10])
        raise SystemExit(f"Found invalid confidence values: {bad_msg}")
    if missing:
        raise SystemExit(f"Missing {len(missing)} expected test aids; first few: {', '.join(missing[:10])}")
    if extra:
        raise SystemExit(f"Found {len(extra)} unexpected aids; first few: {', '.join(extra[:10])}")

    merged_rows = [seen[aid] for aid in expected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Shard files: {len(shard_paths)}")
    print(f"Input lines: {len(all_rows)}")
    print(f"Unique aids: {len(merged_rows)}")
    print(f"Expected test aids: {len(expected)}")
    print(f"Merged file: {args.output}")


if __name__ == "__main__":
    main()
