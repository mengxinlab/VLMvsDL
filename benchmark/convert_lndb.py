"""Convert LNDb_ROI .npy volumes -> MP4 videos (same normalization as LUNA25).

Usage:
    python convert_lndb.py --fps 20

Outputs: <out_dir>/lndb_fps{fps}/<FindingID>.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from convert import normalize_volume, write_video
from paths import LNDB_ROI_DIR, VIDEOS_DIR
import numpy as np

LNDB_ROOT  = LNDB_ROI_DIR
LNDB_IMG   = LNDB_ROOT / "images"
LNDB_LABEL = LNDB_ROOT / "lndb_labels.csv"
DEFAULT_OUT = VIDEOS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(LNDB_LABEL)
    ids = df["FindingID"].astype(str).unique().tolist()
    print(f"LNDb: {len(ids)} unique findings -> mp4 fps{args.fps}")

    out_root = args.out / f"lndb_fps{args.fps}"
    out_root.mkdir(parents=True, exist_ok=True)
    n_skip = n_done = n_fail = 0
    for fid in tqdm(ids):
        out_path = out_root / f"{fid}.mp4"
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue
        npy = LNDB_IMG / f"{fid}.npy"
        if not npy.exists():
            print(f"[WARN] missing {npy}")
            n_fail += 1
            continue
        try:
            vol = np.load(npy)
            frames = normalize_volume(vol)
            write_video(frames, out_path, fps=args.fps)
            n_done += 1
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] {fid}: {e}")
            n_fail += 1
    print(f"done={n_done} skipped={n_skip} failed={n_fail}")


if __name__ == "__main__":
    main()
