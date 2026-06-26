#!/usr/bin/env bash
# Fast path A: OWT 1GB subset, serial BPE (stable on Mac), reduced training steps.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ARTIFACTS="$ROOT/artifacts"
DATA="$ROOT/data"
OWT_SUBSET="$DATA/owt_train_1gb.txt"
SUBSET_BYTES=$((1 * 1024 * 1024 * 1024))

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_tokenize() {
  local input="$1" output="$2" tokdir="$3"
  local meta="${output%.bin}.meta.json"
  if [[ -f "$meta" ]]; then
    log "Skip tokenize (done): $output"
    return
  fi
  rm -f "$output"
  log "Tokenizing $input -> $output"
  uv run python scripts/tokenize_corpus.py \
    --input "$input" --output "$output" --tokenizer-dir "$tokdir"
}

run_train() {
  local name="$1"
  shift
  local outdir="$ARTIFACTS/experiments/$name"
  if [[ -f "$outdir/run_summary.json" ]]; then
    log "Skip train (done): $name"
    return
  fi
  rm -rf "$outdir"
  mkdir -p "$outdir"
  log "Training: $name"
  uv run python train.py --output-dir "$outdir" --run-name "$name" "$@"
}

log "=== Fast full pipeline (OWT 1GB subset) ==="

if [[ ! -f "$OWT_SUBSET" ]]; then
  log "Creating 1GB OWT train subset..."
  head -c "$SUBSET_BYTES" "$DATA/owt_train.txt" > "$OWT_SUBSET"
fi

if [[ ! -f "$ARTIFACTS/bpe/owt_32k/vocab.json" ]]; then
  log "=== OWT BPE on 1GB subset (serial pretoken) ==="
  uv run python scripts/train_bpe.py \
    --input "$OWT_SUBSET" \
    --output-dir "$ARTIFACTS/bpe/owt_32k" \
    --vocab-size 32000 \
    --num-workers 1 \
    --name owt_32k_1gb_subset
fi

if [[ ! -f "$ARTIFACTS/experiments/tinystories_main/run_summary.json" ]]; then
  log "=== TinyStories fast training ==="
  run_train tinystories_main \
    --train-tokens "$ARTIFACTS/tokens/tinystories_train.bin" \
    --val-tokens "$ARTIFACTS/tokens/tinystories_valid.bin" \
    --tokenizer-dir "$ARTIFACTS/bpe/tinystories_10k" \
    --batch-size 32 --max-steps 2500 --eval-every 1000 --eval-batches 15 \
    --checkpoint-every 1250 --log-every 100
fi

if [[ ! -f "$ARTIFACTS/bpe_experiments.json" ]]; then
  log "=== BPE experiments ==="
  uv run python scripts/bpe_experiments.py
fi

log "=== OWT tokenization ==="
run_tokenize "$DATA/owt_valid.txt" "$ARTIFACTS/tokens/owt_valid.bin" "$ARTIFACTS/bpe/owt_32k"
run_tokenize "$OWT_SUBSET" "$ARTIFACTS/tokens/owt_train.bin" "$ARTIFACTS/bpe/owt_32k"

log "=== OWT fast training (2000 steps, batch 32) ==="
run_train owt_main \
  --train-tokens "$ARTIFACTS/tokens/owt_train.bin" \
  --val-tokens "$ARTIFACTS/tokens/owt_valid.bin" \
  --tokenizer-dir "$ARTIFACTS/bpe/owt_32k" \
  --vocab-size 32000 --batch-size 32 --max-steps 2000 \
  --eval-every 1000 --eval-batches 10 --checkpoint-every 1000 --log-every 100

log "=== finalize assignment (pytest + writeup + zip) ==="
bash "$ROOT/scripts/finalize_assignment.sh"

log "=== Fast full pipeline complete: $(date) ==="
rm -f "$ARTIFACTS/pipeline.pid"
