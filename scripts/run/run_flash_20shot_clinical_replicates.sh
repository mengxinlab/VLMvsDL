#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUFFIXES=()

while [[ $# -gt 0 ]]; do
	case "$1" in
		--suffixes)
			IFS=',' read -r -a SUFFIXES <<< "$2"
			shift 2
			;;
		*)
			echo "Unknown argument: $1" >&2
			exit 1
			;;
	esac
done

if [[ ${#SUFFIXES[@]} -eq 0 ]]; then
	SUFFIXES=(01 02 03 04)
fi

cd "$ROOT/benchmark"
source "$ROOT/.venv/bin/activate"

for suffix in "${SUFFIXES[@]}"; do
	echo "===== Gemini F3 repeat $suffix ====="
	"$ROOT/.venv/bin/python" run_sync.py \
		--mode 20shot_clinical \
		--model gemini-3-flash-preview \
		--rich-prompt \
		--temperature 0 \
		--workers 2 \
		--sleep 5 \
		--result-suffix "$suffix"
done
