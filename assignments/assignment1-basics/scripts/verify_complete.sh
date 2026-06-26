#!/usr/bin/env bash
# Exit 0 only when Assignment 1 fast pipeline is fully done (nothing in progress).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ART="$ROOT/artifacts"
PIDFILE="$ART/pipeline.pid"
LOG="$ART/pipeline_fast.log"

fail() { echo "INCOMPLETE: $*"; exit 1; }

# No pipeline worker should be running
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  fail "pipeline pid $(cat "$PIDFILE") still running ($(tail -1 "$LOG" 2>/dev/null || echo no log))"
fi
if pgrep -f "continue_fast.sh|train_bpe.py|tokenize_corpus.py|train.py" >/dev/null 2>&1; then
  fail "training/tokenize/BPE processes still active"
fi

required=(
  "$ART/bpe/tinystories_10k/vocab.json"
  "$ART/bpe/owt_32k/vocab.json"
  "$ART/bpe_experiments.json"
  "$ART/tokens/tinystories_train.meta.json"
  "$ART/tokens/owt_train.meta.json"
  "$ART/tokens/owt_valid.meta.json"
  "$ART/experiments/tinystories_main/run_summary.json"
  "$ART/experiments/owt_main/run_summary.json"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || fail "missing $path"
done

if ! grep -q "Fast full pipeline complete" "$LOG" 2>/dev/null; then
  fail "pipeline log missing completion marker"
fi

echo "ALL COMPLETE"
echo "  tinystories_main: $(python3 -c "import json; print(json.load(open('$ART/experiments/tinystories_main/run_summary.json'))['final_val_loss'])")"
echo "  owt_main:         $(python3 -c "import json; print(json.load(open('$ART/experiments/owt_main/run_summary.json'))['final_val_loss'])")"
exit 0
