#!/usr/bin/env bash
# Export writeup.md → writeup.pdf (learning notes, not Gradescope).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

uv pip install markdown weasyprint -q 2>/dev/null || true

.venv/bin/python << 'PYEOF'
from pathlib import Path
import markdown
from weasyprint import HTML

root = Path(".")
md = (root / "writeup.md").read_text()
html_body = markdown.markdown(md, extensions=["tables", "fenced_code", "nl2br"])
css = """
@page { margin: 2cm; size: letter; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.45; }
h1 { font-size: 20pt; margin-top: 0; }
h2 { font-size: 14pt; margin-top: 1.2em; border-bottom: 1px solid #ccc; }
h3 { font-size: 12pt; }
code, pre { font-family: Menlo, monospace; font-size: 9pt; }
pre { background: #f5f5f5; padding: 8px; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }
"""
doc = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{html_body}</body></html>"
out = root / "writeup.pdf"
HTML(string=doc, base_url=str(root.resolve())).write_pdf(str(out))
print(f"Wrote {out} ({out.stat().st_size} bytes)")
PYEOF
