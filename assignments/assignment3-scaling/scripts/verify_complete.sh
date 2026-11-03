#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync -q --extra server
export INTERNAL_API_KEY="${INTERNAL_API_KEY:-test-internal-key}"

echo "== pytest =="
uv run pytest -q

echo ""
echo "== analysis =="
uv run python scripts/run_analysis.py

REQUIRED=(
  artifacts/isoflops_compute_optimal.json
  artifacts/scaling_law_fits.json
  artifacts/isoflops_parabola_fits.json
  artifacts/isoflops_curves.svg
  artifacts/final_submission_prediction.json
  writeup.md
)

echo ""
echo "== artifacts =="
for f in "${REQUIRED[@]}"; do
  [[ -f "$f" ]] || { echo "MISSING $f"; exit 1; }
  echo "  ok  $f"
done
echo "Assignment 3 verification passed."
