#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --no-install-package flash-attn -q
uv sync -q

echo "== pytest =="
uv run pytest -q tests/

REQUIRED=(
  cs336_alignment/grpo.py
  cs336_alignment/drgrpo_grader.py
  tests/adapters.py
  writeup.md
  SOLUTION.md
  GOALS.md
  LEARNINGS.md
)

echo ""
echo "== required files =="
for f in "${REQUIRED[@]}"; do
  [[ -f "$f" ]] || { echo "MISSING $f"; exit 1; }
  echo "  ok  $f"
done
echo "Assignment 5 verification passed."
