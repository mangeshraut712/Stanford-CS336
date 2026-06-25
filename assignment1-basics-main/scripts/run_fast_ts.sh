#!/usr/bin/env bash
# Fast path C: TinyStories only. Keeps Mac awake via caffeinate.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/artifacts/pipeline_fast_ts.log"
PIDFILE="$ROOT/artifacts/pipeline.pid"
mkdir -p "$ROOT/artifacts"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Stopping existing pipeline pid $(cat "$PIDFILE")..."
  pkill -P "$(cat "$PIDFILE")" 2>/dev/null || true
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  rm -f "$PIDFILE"
fi

pkill -f "train_bpe.py.*owt_train" 2>/dev/null || true
pkill -f "train.py.*tinystories_main" 2>/dev/null || true
pkill -f "continue_pipeline.sh" 2>/dev/null || true

{
  echo "========================================"
  echo "Fast TS-only pipeline start: $(date)"
  echo "========================================"
} >>"$LOG"

nohup caffeinate -dims bash "$ROOT/scripts/continue_fast_ts.sh" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Started fast TS-only pipeline pid $(cat "$PIDFILE")"
echo "Log: $LOG"
echo "Tail: tail -f $LOG"
