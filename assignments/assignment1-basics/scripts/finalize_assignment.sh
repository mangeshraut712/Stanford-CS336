#!/usr/bin/env bash
# Finalize Assignment 1: sync metrics, test, rebuild submission zip.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== pytest ==="
.venv/bin/python -m pytest -q

echo "=== sync writeup + checklist ==="
uv run python scripts/finalize_assignment.py

echo "=== submission zip ==="
bash make_submission.sh

echo "=== done ==="
cat ASSIGNMENT_CHECKLIST.md
