#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/benchmark"
source "$ROOT/.venv/bin/activate"
python run_sync.py --mode 20shot_clinical --model gemini-3-flash-preview --rich-prompt --temperature 0 --workers 2 --sleep 5 --video-sampling-fps 20 --result-suffix samplefps20
