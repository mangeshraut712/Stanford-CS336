#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/verify_complete.sh
echo ""
echo "See writeup.md for GRPO and supplement overview."
