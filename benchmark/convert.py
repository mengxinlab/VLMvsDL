"""Step 1: Convert LUNA25 ROI .npy volumes (64,128,128 HU int16) into MP4 videos.

Normalization follows the fixed CT preprocessing convention used for all
video-based VLM inputs:
    vol = clip(vol, -1000, 1000) / 1000.0       # [-1, 1]
    pix = ((vol + 1) / 2 * 255).astype(uint8)   # [0, 255]

Each axial slice along axis 0 (z) becomes one frame; gray -> BGR; 128x128 retained.
Outputs: <out_dir>/fps{fps}/<AnnotationID>.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from tqdm import tqdm

from data_utils import (
    DEFAULT_IMAGE_DIR,
    ID_COL,
    get_split_rows,
    load_label_df,
    load_split,
    npy_path,
)
from paths import VIDEOS_DIR

DEFAULT_OUT = VIDEOS_DIR


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """HU int16 (D,H,W) -> uint8 (D,H,W) using the fixed CT window."""
    vol = vol.astype(np.float32)
    vol = np.clip(vol, -1000.0, 1000.0) / 1000.0  # [-1, 1]
    vol = ((vol + 1.0) * 0.5 * 255.0).astype(np.uint8)
    return vol


def write_video(frames_uint8: np.ndarray, out_path: Path, fps: int) -> None:
    """frames_uint8: (D, H, W) uint8 -> mp4v MP4."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d, h, w = frames_uint8.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {out_path}")
    try:
        for i in range(d):
            bgr = cv2.cvtColor(frames_uint8[i], cv2.COLOR_GRAY2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def convert_one(annotation_id: str, fps: int, out_root: Path,
                image_dir: Path = DEFAULT_IMAGE_DIR, overwrite: bool = False) -> Path:
    out_path = out_root / f"fps{fps}" / f"{annotation_id}.mp4"
    if out_path.exists() and not overwrite:
        return out_path
    vol = np.load(npy_path(annotation_id, image_dir))
    if vol.ndim != 3:
        raise ValueError(f"{annotation_id}: expected 3D npy, got shape {vol.shape}")
    frames = normalize_volume(vol)  # (D,H,W) uint8
    write_video(frames, out_path, fps=fps)
    return out_path


def convert_many(ids: Iterable[str], fps: int, out_root: Path,
                 image_dir: Path = DEFAULT_IMAGE_DIR, overwrite: bool = False) -> List[Path]:
    paths: List[Path] = []
    for aid in tqdm(list(ids), desc=f"npy->mp4 fps{fps}"):
        try:
            paths.append(convert_one(aid, fps, out_root, image_dir, overwrite))
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] {aid}: {e}")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, nargs="+", default=[20, 60],
                    help="FPS values to render (default: 20 60)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--include-train", action="store_true",
                    help="Also convert train split (huge). Default: only test split.")
    ap.add_argument("--ids", type=Path, default=None,
                    help="Optional path to a text file with AnnotationIDs to convert (one per line).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.ids:
        with open(args.ids, "r") as f:
            ids = [ln.strip() for ln in f if ln.strip()]
    else:
        df = load_label_df()
        split = load_split()
        rows = get_split_rows(df, split, "test")
        ids = rows[ID_COL].tolist()
        if args.include_train:
            tr = get_split_rows(df, split, "train")
            ids = ids + tr[ID_COL].tolist()
        ids = sorted(set(ids))
    print(f"Converting {len(ids)} samples at fps={args.fps}")
    for fps in args.fps:
        convert_many(ids, fps=fps, out_root=args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
