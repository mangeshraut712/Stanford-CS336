#!/usr/bin/env bash
# Verify Assignments 1–4 across the monorepo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "========== Assignment 1 =========="
cd assignments/assignment1-basics
uv sync -q
uv run pytest -q
echo "A1 OK"

echo ""
echo "========== Assignment 2 =========="
cd "$ROOT/assignments/assignment2-systems"
uv sync -q
uv run pytest -q
echo "A2 OK"

echo ""
echo "========== Assignment 3 =========="
cd "$ROOT/assignments/assignment3-scaling"
bash scripts/verify_complete.sh
echo "A3 OK"

echo ""
echo "========== Assignment 4 =========="
cd "$ROOT/assignments/assignment4-data"
bash scripts/verify_complete.sh
echo "A4 OK"

echo ""
echo "All assignments 1–4 verified."
