#!/usr/bin/env bash
# Run the full Assignment 1 pipeline unattended.
# Uses caffeinate so the Mac stays awake while plugged in (see README note on lid-closed sleep).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/artifacts/pipeline.log"
PIDFILE="$ROOT/artifacts/pipeline.pid"
mkdir -p "$ROOT/artifacts"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Pipeline already running (pid $(cat "$PIDFILE")). Tail: $LOG"
  exit 0
fi

{
  echo "========================================"
  echo "Assignment 1 pipeline start: $(date)"
  echo "========================================"
} >>"$LOG"

# -d prevent disk sleep, -i prevent idle sleep, -m prevent display sleep (optional), -s prevent system sleep
nohup caffeinate -dims bash "$ROOT/scripts/continue_pipeline.sh" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Started pipeline pid $(cat "$PIDFILE")"
echo "Log: $LOG"
echo "Tail with: tail -f $LOG"
