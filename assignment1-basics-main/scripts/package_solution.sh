#!/usr/bin/env bash
# Package our Assignment 1 solution: test → sync docs → submission zip.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== CS336 Assignment 1 — solution package ==="
echo "Repo: $ROOT"
echo ""

bash "$ROOT/scripts/finalize_assignment.sh"

echo ""
echo "Deliverables:"
echo "  SOLUTION.md     — solution overview (this repo)"
echo "  writeup.md      — export to PDF for Gradescope"
echo "  cs336-spring2025-assignment-1-submission.zip"
echo ""
echo "Next: open writeup.md → Print to PDF, or: brew install pandoc && pandoc writeup.md -o writeup.pdf"
