#!/usr/bin/env bash
# Fast path C: TinyStories only — no OWT. Uses fast training settings (path A knobs).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ARTIFACTS="$ROOT/artifacts"
DATA="$ROOT/data"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

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

log "=== Fast TS-only pipeline (skip OWT) ==="

if [[ ! -f "$ARTIFACTS/tokens/tinystories_train.meta.json" ]]; then
  log "ERROR: tinystories_train.meta.json missing — run tokenization first"
  exit 1
fi

log "=== TinyStories fast training (2500 steps, batch 32 — lower RAM) ==="
run_train tinystories_main \
  --train-tokens "$ARTIFACTS/tokens/tinystories_train.bin" \
  --val-tokens "$ARTIFACTS/tokens/tinystories_valid.bin" \
  --tokenizer-dir "$ARTIFACTS/bpe/tinystories_10k" \
  --batch-size 32 \
  --max-steps 2500 \
  --eval-every 1000 \
  --eval-batches 15 \
  --checkpoint-every 1250 \
  --log-every 100

log "=== finalize assignment (pytest + writeup + zip) ==="
bash "$ROOT/scripts/finalize_assignment.sh"

log "=== Fast TS-only pipeline complete: $(date) ==="
rm -f "$ARTIFACTS/pipeline.pid"
