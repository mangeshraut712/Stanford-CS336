"""Extract plain text from HTML bytes."""

from __future__ import annotations

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.html import HTMLTree


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    html = html_bytes.decode("utf-8", errors="replace")
    tree = HTMLTree.parse(html)
    text = extract_plain_text(tree, main_content=False, links=False)
    if text is None:
        return None
    return text
