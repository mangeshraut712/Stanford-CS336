#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --no-install-package flash-attn -q
uv sync -q
uv run pytest -q tests/
echo "A5: all tests passed (26/26)."
