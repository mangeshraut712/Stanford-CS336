#!/usr/bin/env bash
# Resume assignment pipeline from current artifacts. Safe to re-run; skips finished steps.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ARTIFACTS="$ROOT/artifacts"
DATA="$ROOT/data"
LOG="$ARTIFACTS/pipeline.log"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for() {
  local path="$1"
  local msg="$2"
  while [[ ! -f "$path" ]]; do
    log "Waiting for $msg..."
    sleep 60
  done
}

wait_for_process_or_file() {
  local pattern="$1"
  local done_path="$2"
  local msg="$3"
  while [[ ! -f "$done_path" ]]; do
    if pgrep -f "$pattern" >/dev/null 2>&1; then
      log "In progress: $msg (another process running)"
    else
      log "Expected $done_path but no process matching '$pattern'; will retry step"
      return 1
    fi
    sleep 60
  done
}

ensure_owt_bpe() {
  if [[ -f "$ARTIFACTS/bpe/owt_32k/vocab.json" ]]; then
    log "OWT BPE already done"
    return
  fi
  if pgrep -f "train_bpe.py.*owt_train" >/dev/null 2>&1; then
    log "OWT BPE already running"
    return
  fi
  log "Starting OWT BPE (32k, streaming pretokenization)..."
  uv run python scripts/train_bpe.py \
    --input "$DATA/owt_train.txt" \
    --output-dir "$ARTIFACTS/bpe/owt_32k" \
    --vocab-size 32000 \
    --num-workers 2 \
    --name owt_32k
}

run_tokenize() {
  local input="$1" output="$2" tokdir="$3"
  local meta="${output%.bin}.meta.json"
  if [[ -f "$meta" ]]; then
    log "Skip tokenize (done): $output"
    return
  fi
  if pgrep -f "tokenize_corpus.py.*${output##*/}" >/dev/null 2>&1; then
    wait_for "$meta" "tokenization of ${output##*/}"
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
  mkdir -p "$outdir"
  log "Training: $name"
  uv run python train.py --output-dir "$outdir" --run-name "$name" "$@"
}

log "=== Phase 0: ensure OWT BPE is running ==="
ensure_owt_bpe || true

log "=== Phase 1: TinyStories train tokenization ==="
if [[ ! -f "$ARTIFACTS/tokens/tinystories_train.meta.json" ]]; then
  if pgrep -f "tokenize_corpus.py.*tinystories_train" >/dev/null 2>&1; then
    wait_for "$ARTIFACTS/tokens/tinystories_train.meta.json" "tinystories_train tokenization"
  else
    run_tokenize "$DATA/TinyStoriesV2-GPT4-train.txt" \
      "$ARTIFACTS/tokens/tinystories_train.bin" \
      "$ARTIFACTS/bpe/tinystories_10k"
  fi
fi

run_tokenize "$DATA/TinyStoriesV2-GPT4-valid.txt" \
  "$ARTIFACTS/tokens/tinystories_valid.bin" \
  "$ARTIFACTS/bpe/tinystories_10k"

log "=== Phase 2: TinyStories main training (5000 steps) ==="
run_train tinystories_main \
  --train-tokens "$ARTIFACTS/tokens/tinystories_train.bin" \
  --val-tokens "$ARTIFACTS/tokens/tinystories_valid.bin" \
  --tokenizer-dir "$ARTIFACTS/bpe/tinystories_10k" \
  --batch-size 32 --max-steps 5000 --eval-every 500 --checkpoint-every 2500

log "=== Phase 3: OWT BPE ==="
if [[ ! -f "$ARTIFACTS/bpe/owt_32k/vocab.json" ]]; then
  if pgrep -f "train_bpe.py.*owt_train" >/dev/null 2>&1; then
    wait_for "$ARTIFACTS/bpe/owt_32k/vocab.json" "owt_32k tokenizer"
  else
    ensure_owt_bpe
  fi
fi

log "=== Phase 4: BPE experiments ==="
if [[ ! -f "$ARTIFACTS/bpe_experiments.json" ]]; then
  uv run python scripts/bpe_experiments.py
fi

log "=== Phase 5: OWT tokenization ==="
run_tokenize "$DATA/owt_valid.txt" "$ARTIFACTS/tokens/owt_valid.bin" "$ARTIFACTS/bpe/owt_32k"
run_tokenize "$DATA/owt_train.txt" "$ARTIFACTS/tokens/owt_train.bin" "$ARTIFACTS/bpe/owt_32k"

log "=== Phase 6: OWT main training ==="
run_train owt_main \
  --train-tokens "$ARTIFACTS/tokens/owt_train.bin" \
  --val-tokens "$ARTIFACTS/tokens/owt_valid.bin" \
  --tokenizer-dir "$ARTIFACTS/bpe/owt_32k" \
  --vocab-size 32000 --batch-size 16 --max-steps 5000 --eval-every 500 --checkpoint-every 2500

log "=== Phase 7: Leaderboard mod ==="
run_train leaderboard_mod \
  --train-tokens "$ARTIFACTS/tokens/owt_train.bin" \
  --val-tokens "$ARTIFACTS/tokens/owt_valid.bin" \
  --tokenizer-dir "$ARTIFACTS/bpe/owt_32k" \
  --vocab-size 32000 --batch-size 32 --max-steps 8000 --max-lr 6e-4 \
  --warmup-steps 400 --eval-every 400

log "=== Phase 8: pytest + writeup + submission ==="
uv run pytest -q
uv run python scripts/update_writeup.py
bash make_submission.sh

log "=== Pipeline complete: $(date) ==="
rm -f "$ARTIFACTS/pipeline.pid"
