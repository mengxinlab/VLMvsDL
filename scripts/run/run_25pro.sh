#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/benchmark"
source "$ROOT/.venv/bin/activate"
python run_sync.py --mode 20shot_clinical --model gemini-2.5-pro --rich-prompt --workers 2 --sleep 10
