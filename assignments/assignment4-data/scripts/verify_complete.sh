#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f local-shared-data/classifiers/lid.176.bin ]]; then
  echo "Downloading offline assets (first run)..."
  uv run python scripts/download_data.py --offline-only
fi

echo "== pytest =="
uv run pytest -q

REQUIRED=(
  cs336_data/extract.py
  cs336_data/langid.py
  cs336_data/pii.py
  cs336_data/toxicity.py
  cs336_data/quality.py
  cs336_data/deduplication.py
  writeup.md
)

echo ""
echo "== required files =="
for f in "${REQUIRED[@]}"; do
  [[ -f "$f" ]] || { echo "MISSING $f"; exit 1; }
  echo "  ok  $f"
done
echo "Assignment 4 verification passed."
