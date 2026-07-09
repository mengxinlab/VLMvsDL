"""Export the 3 sampled CT slices per LUNA25 test case as PNGs.

Run this once where the .npy volumes live in a private data environment. The tiny
PNG bundle (~917 x 3 small grayscale images) can then be downloaded and used by
run_crossfamily_api.py to query closed-source models (GPT-4o / Claude) from a
machine with internet but without the raw CT data.

The frames use the exact same sampling as run_crossfamily_offline.py (the hosted
Gemini indices [9, 29, 49]), so the closed-source anchor sees identical pixels.

    python experiments/cross_family/export_frames.py --out-dir results/vlm/crossfamily/frames
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "rcf", str(Path(__file__).resolve().parent / "run_crossfamily_offline.py")
)
rcf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcf)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-dir", default=rcf.DEFAULT_IMG_DIR)
    p.add_argument("--csv-path", default=rcf.DEFAULT_CSV_PATH)
    p.add_argument("--split-path", default=rcf.DEFAULT_SPLIT_PATH)
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[2] / "results/vlm/crossfamily/frames"))
    p.add_argument("--n-frames", type=int, default=3)
    p.add_argument("--sample-fps", type=float, default=1.0)
    p.add_argument("--source-video-fps", type=float, default=20.0)
    args = p.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = rcf.load_test_rows(Path(args.csv_path), Path(args.split_path), img_dir)
    print(f"Exporting frames for {len(rows)} test cases -> {out_dir}")

    with (out_dir / "manifest.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["aid", "label", "n_frames"])
        for i, row in rows.iterrows():
            aid = str(row[rcf.ID_COL])
            label = int(row[rcf.LABEL_COL])
            frames = rcf.npy_to_pil(
                aid, img_dir, args.n_frames,
                source_video_fps=args.source_video_fps, sample_fps=args.sample_fps,
            )
            case_dir = out_dir / aid
            case_dir.mkdir(exist_ok=True)
            for j, frame in enumerate(frames):
                frame.save(case_dir / f"{j}.png")
            writer.writerow([aid, label, len(frames)])
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(rows)}", flush=True)
    print("Done. Use the frames/ folder for the hosted-model API anchor.")


if __name__ == "__main__":
    main()
