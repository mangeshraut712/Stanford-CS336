#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/verify_complete.sh
echo ""
uv run python scripts/predict_final_submission.py
echo ""
echo "See writeup.md for full analysis."
