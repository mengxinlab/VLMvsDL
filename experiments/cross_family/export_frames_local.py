"""Export the 3 sampled CT slices per LUNA25 test case from the LOCAL fps20 MP4s.

No GPU and no raw .npy needed: the repo already has the fps20 nodule videos under
results/videos/fps20/<aid>.mp4 (the exact clips the hosted Gemini API ingested).
We pull frames [9, 29, 49] of 64 -- the same indices Gemini sampled -- so the
API models see identical pixels.

Output: <out-dir>/<aid>/{0,1,2}.png and <out-dir>/manifest.csv (aid,label,n_frames),
consumed by run_crossfamily_api.py.

    python experiments/cross_family/export_frames_local.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FRAME_INDICES = (9, 29, 49)  # matches the hosted Gemini 1-fps-of-20-fps sampling


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label-csv", default=str(ROOT / "data/metadata/luna25_public_training_development_data.csv"))
    p.add_argument("--split", default=str(ROOT / "data/metadata/patient_split.json"))
    p.add_argument("--videos", default=str(ROOT / "results/videos/fps20"))
    p.add_argument("--out-dir", default=str(ROOT / "results/vlm/crossfamily/frames"))
    args = p.parse_args()

    df = pd.read_csv(args.label_csv)
    split = json.loads(Path(args.split).read_text())
    test_pids = {str(pid) for pid in split["test"]}
    test = df[df["PatientID"].astype(str).isin(test_pids)].copy()

    videos = Path(args.videos)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_missing = 0, 0
    with (out_dir / "manifest.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["aid", "label", "n_frames"])
        for _, row in test.iterrows():
            aid = str(row["AnnotationID"])
            label = int(row["label"])
            vid = videos / f"{aid}.mp4"
            if not vid.exists():
                n_missing += 1
                continue
            reader = imageio.get_reader(str(vid))
            try:
                n = reader.count_frames()
                idxs = [i for i in FRAME_INDICES if i < n] or [min(n - 1, n // 2)]
                case_dir = out_dir / aid
                case_dir.mkdir(exist_ok=True)
                for j, i in enumerate(idxs):
                    Image.fromarray(reader.get_data(i)).convert("RGB").save(case_dir / f"{j}.png")
                writer.writerow([aid, label, len(idxs)])
                n_ok += 1
            finally:
                reader.close()
            if n_ok % 100 == 0:
                print(f"  {n_ok} done", flush=True)

    print(f"Exported {n_ok} cases ({n_missing} missing videos) -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
