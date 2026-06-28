#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --no-install-package flash-attn -q
uv sync -q
uv run pytest -q tests/test_grpo.py
echo "A5 GRPO: all required tests passed."
