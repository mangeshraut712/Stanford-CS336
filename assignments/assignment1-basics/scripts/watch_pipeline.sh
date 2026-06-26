#!/usr/bin/env bash
# Print pipeline phase + detect stalls/crashes. Exit 1 if pipeline died unexpectedly.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/artifacts/pipeline_fast.log"
PIDFILE="$ROOT/artifacts/pipeline.pid"

phase() {
  if [[ -f "$ROOT/artifacts/experiments/owt_main/run_summary.json" ]]; then
    echo "DONE (owt_main)"
  elif pgrep -f "train.py.*owt_main" >/dev/null 2>&1; then
    echo "TRAINING owt_main"
  elif [[ -f "$ROOT/artifacts/tokens/owt_train.meta.json" ]]; then
    echo "TOKENIZED (train done)"
  elif pgrep -f "tokenize_corpus.py" >/dev/null 2>&1; then
    echo "TOKENIZING"
  elif [[ -f "$ROOT/artifacts/bpe_experiments.json" ]]; then
    echo "BPE experiments done"
  elif [[ -f "$ROOT/artifacts/bpe/owt_32k/vocab.json" ]]; then
    echo "OWT BPE done"
  elif pgrep -f "train_bpe.py.*owt" >/dev/null 2>&1; then
    echo "OWT BPE running"
  else
    echo "IDLE or starting"
  fi
}

echo "=== $(date) ==="
echo "Phase: $(phase)"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Pipeline pid: $(cat "$PIDFILE") (alive)"
else
  echo "Pipeline pid: not running"
fi
echo "--- last log lines ---"
tail -8 "$LOG" 2>/dev/null || true

if grep -q "BrokenPipeError\|Traceback (most recent" "$LOG" 2>/dev/null; then
  if ! pgrep -f "train_bpe.py.*owt" >/dev/null 2>&1; then
    echo "WARNING: errors in log and BPE not running — may need restart"
    exit 1
  fi
fi

if [[ -f "$ROOT/artifacts/experiments/owt_main/run_summary.json" ]]; then
  echo "SUCCESS: pipeline complete"
  exit 0
fi
