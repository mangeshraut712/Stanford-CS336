#!/usr/bin/env bash
# Fast path A: full assignment with OWT 2GB subset + reduced steps.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/artifacts/pipeline_fast.log"
PIDFILE="$ROOT/artifacts/pipeline.pid"
mkdir -p "$ROOT/artifacts"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Stopping existing pipeline pid $(cat "$PIDFILE")..."
  pkill -P "$(cat "$PIDFILE")" 2>/dev/null || true
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  rm -f "$PIDFILE"
fi

pkill -f "train_bpe.py.*owt_train" 2>/dev/null || true
pkill -f "train.py" 2>/dev/null || true
pkill -f "continue_pipeline.sh" 2>/dev/null || true
pkill -f "continue_fast_ts.sh" 2>/dev/null || true

{
  echo "========================================"
  echo "Fast full pipeline start: $(date)"
  echo "========================================"
} >>"$LOG"

nohup caffeinate -dims bash "$ROOT/scripts/continue_fast.sh" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Started fast full pipeline pid $(cat "$PIDFILE")"
echo "Log: $LOG"
echo "Tail: tail -f $LOG"
