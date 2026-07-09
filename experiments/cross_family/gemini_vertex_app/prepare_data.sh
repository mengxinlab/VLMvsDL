#!/usr/bin/env bash
# Copy the already-exported 917-case frame bundle into this app directory.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$APP_DIR/../../.." && pwd)"

mkdir -p "$APP_DIR/data"
rm -rf "$APP_DIR/data/frames"
cp -R "$REPO_ROOT/results/vlm/crossfamily/frames" "$APP_DIR/data/frames"
cp "$REPO_ROOT/data/metadata/clinical_texts.csv" "$APP_DIR/data/clinical_texts.csv"

echo "Prepared app data:"
du -sh "$APP_DIR/data"
wc -l "$APP_DIR/data/frames/manifest.csv"
find "$APP_DIR/data/frames" -name '*.png' | wc -l
