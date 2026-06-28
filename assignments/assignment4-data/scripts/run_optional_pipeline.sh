#!/usr/bin/env bash
# Optional Assignment 4 pipeline: process sample WET, tokenize, smoke train.
# For full Modal training: uv run modal run scripts/train.py --train-bin /path/to/data.bin
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/4 Local verification =="
bash scripts/verify_complete.sh

echo ""
echo "== 2/4 Process offline sample WET =="
uv run python scripts/process_sample_wet.py

CORPUS=artifacts/processed/sample_corpus.txt
if [[ ! -s "$CORPUS" ]]; then
  echo "No documents passed filters in sample WET (filters may be strict). Using fixture text fallback."
  mkdir -p artifacts/processed
  cat tests/fixtures/moby_extracted.txt tests/fixtures/high_quality_wiki_reference.txt > "$CORPUS"
fi

echo ""
echo "== 3/4 GPT-2 tokenize =="
uv run python scripts/tokenize_gpt2.py \
  --input "$CORPUS" \
  --output artifacts/tokens/smoke_train.bin

echo ""
echo "== 4/4 Smoke train (tiny model, few steps) =="
uv run python scripts/smoke_train.py \
  --train-bin artifacts/tokens/smoke_train.bin \
  --steps 10

echo ""
echo "Optional pipeline complete."
echo "For full English WET download (4 files): uv run python scripts/download_wet_smoke.py"
echo "For Modal 8xB200 training: uv run modal run scripts/train.py --train-bin /path/to/your_data.bin"
