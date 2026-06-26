#!/usr/bin/env bash
# Sync docs and tests after experiments (learning project — no submission zip).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== pytest ==="
.venv/bin/python -m pytest -q

echo "=== sync writeup + project status ==="
uv run python scripts/finalize_assignment.py

echo "=== export writeup.pdf ==="
bash "$ROOT/scripts/export_writeup_pdf.sh"

echo "=== done ==="
cat PROJECT_STATUS.md
