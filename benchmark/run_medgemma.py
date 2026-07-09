#!/usr/bin/env python3
"""
Local MedGemma benchmark runner.

Mirrors run_sync.py but uses local HuggingFace inference instead of Gemini API.
No video upload needed — reads .npy CT crops directly.

Modes:
  zeroshot          - npy slices only, no clinical text
  zeroshot_clinical - npy slices + clinical text
  20shot_clinical   - 20 few-shot examples (npy) + clinical text

Usage:
  python run_medgemma.py --mode zeroshot_clinical --model google/medgemma-1.5-4b-it --rich-prompt
  python run_medgemma.py --mode zeroshot --model google/medgemma-1.5-4b-it

Results saved to results/medgemma_{mode}_{model_tag}_{prompt}.jsonl
Re-running resumes from where it left off.
"""
import argparse
import json
import sys
from pathlib import Path

BENCH = Path(__file__).parent
sys.path.insert(0, str(BENCH))

from data_utils import (ID_COL, LABEL_COL, get_clinical_text, get_split_rows,
                        load_label_df, load_split)
from medgemma_client import load_model, predict_one_local
from paths import VLM_RESULTS_DIR

SAMPLES_PATH = BENCH / "few_shot_samples.json"
RESULTS_DIR  = VLM_RESULTS_DIR
RESULTS_DIR.mkdir(exist_ok=True)

MODES = ("zeroshot", "zeroshot_clinical", "20shot_clinical")


def format_float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_done(result_path: Path) -> set:
    done = set()
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["aid"])
                except Exception:
                    pass
    return done


def load_examples(mode: str, df) -> list:
    """Load few-shot example dicts (with aid, label, clinical_text)."""
    if mode not in ("20shot_clinical",):
        return []
    samples = json.loads(SAMPLES_PATH.read_text())
    raw = samples["all_20"]
    examples = []
    for s in raw:
        ex_aid   = s[ID_COL]
        ex_label = int(s[LABEL_COL])
        ex_clin  = get_clinical_text(ex_aid) if mode == "20shot_clinical" else ""
        examples.append({"aid": ex_aid, "label": ex_label, "clinical_text": ex_clin})
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=MODES, required=True)
    parser.add_argument("--model",      default="google/medgemma-1.5-4b-it",
                        help="HuggingFace model ID (default: google/medgemma-1.5-4b-it)")
    parser.add_argument("--rich-prompt", action="store_true",
                        help="Use detailed radiologist prompt (recommended)")
    parser.add_argument("--n-frames",   type=int, default=64,
                        help="Axial slices per volume (default: 64; use all slices for fair comparison)")
    parser.add_argument("--example-n-frames", type=int, default=None,
                        help="Axial slices per few-shot example (default: same as --n-frames)")
    parser.add_argument("--source-video-fps", type=float, default=None,
                        help="Synthetic source video FPS for CT slice timeline. Required with --sample-fps.")
    parser.add_argument("--sample-fps", type=float, default=None,
                        help="If set, sample target slices by synthetic video time; few-shot examples inherit unless --example-n-frames is set.")
    parser.add_argument("--frame-anchor", choices=("start", "middle", "gemini"), default="gemini",
                        help="Which frame to take inside each FPS window when --sample-fps is used (default: gemini, e.g. 64@20fps -> [9,29,49]).")
    parser.add_argument("--max-new-tokens", type=int, default=48,
                        help="Max generated tokens (default: 48; enough for the requested JSON output)")
    parser.add_argument("--limit",      type=int, default=10000)
    args = parser.parse_args()

    if args.sample_fps is not None and args.source_video_fps is None:
        parser.error("--source-video-fps is required when --sample-fps is set")

    model_tag  = args.model.replace("/", "_").replace("-", "_")
    prompt_tag = "rich" if args.rich_prompt else "minimal"
    if args.sample_fps is not None:
        frame_tag = (
            f"_fps{format_float_tag(args.sample_fps)}"
            f"of{format_float_tag(args.source_video_fps)}_{args.frame_anchor}"
        )
    else:
        frame_tag = "" if args.n_frames == 64 else f"_f{args.n_frames}"
    example_frame_tag = ""
    if args.example_n_frames is not None and args.example_n_frames != args.n_frames:
        example_frame_tag = f"_exf{args.example_n_frames}"
    result_path = RESULTS_DIR / f"medgemma_{args.mode}_{model_tag}_{prompt_tag}{frame_tag}{example_frame_tag}.jsonl"

    done = load_done(result_path)
    sampling_desc = (
        f"{args.sample_fps:g}fps of {args.source_video_fps:g}fps ({args.frame_anchor})"
        if args.sample_fps is not None
        else str(args.n_frames)
    )
    print(f"Mode: {args.mode} | Model: {args.model} | Prompt: {prompt_tag} | Frames: {sampling_desc} | MaxNew: {args.max_new_tokens}")
    print(f"Already done: {len(done)} | Result: {result_path.name}")

    # load model once
    load_model(args.model)

    df       = load_label_df()
    split    = load_split()
    test_rows = get_split_rows(df, split, "test")
    examples  = load_examples(args.mode, df)
    print(f"Test samples: {len(test_rows)} | Examples: {len(examples)}")

    remaining = [r for _, r in test_rows.iterrows()
                 if r[ID_COL] not in done][:args.limit]
    print(f"Remaining: {len(remaining)}")

    for i, row in enumerate(remaining):
        aid   = row[ID_COL]
        label = int(row[LABEL_COL])
        clin  = get_clinical_text(aid) if args.mode in ("zeroshot_clinical", "20shot_clinical") else ""

        try:
            result = predict_one_local(
                aid,
                mode=args.mode,
                clinical_text=clin,
                examples=examples,
                rich_prompt=args.rich_prompt,
                n_frames=args.n_frames,
                example_n_frames=args.example_n_frames,
                source_video_fps=args.source_video_fps,
                sample_fps=args.sample_fps,
                frame_anchor=args.frame_anchor,
                max_new_tokens=args.max_new_tokens,
            )
            conf      = result["confidence"]
            reasoning = result["reasoning"]
        except Exception as e:
            print(f"  [ERROR] {aid}: {e}", flush=True)
            conf, reasoning = -1.0, str(e)

        record = {
            "aid": aid, "label": label, "mode": args.mode,
            "prompt": prompt_tag, "model": args.model,
            "source_video_fps": args.source_video_fps,
            "sample_fps": args.sample_fps,
            "frame_anchor": args.frame_anchor if args.sample_fps is not None else None,
            "confidence": conf, "reasoning": reasoning,
        }
        with open(result_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        n_done = len(done) + i + 1
        print(f"  [{n_done}] {aid} conf={conf:.3f}  {reasoning[:70]}", flush=True)

    total = len(load_done(result_path))
    print(f"\nDone. Total saved: {total} / {len(test_rows)}")


if __name__ == "__main__":
    main()
