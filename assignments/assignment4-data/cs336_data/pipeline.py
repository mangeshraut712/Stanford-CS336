"""End-to-end document filtering pipeline for Assignment 4."""

from __future__ import annotations

from cs336_data.langid import identify_language
from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.quality import gopher_quality_filter
from cs336_data.toxicity import classify_nsfw, classify_toxic_speech


def filter_document(text: str, *, min_lang_score: float = 0.5) -> str | None:
    """Return cleaned text if it passes filters, else None."""
    text = text.strip()
    if not text:
        return None

    lang, lang_score = identify_language(text)
    if lang != "en" or lang_score < min_lang_score:
        return None

    if not gopher_quality_filter(text):
        return None

    nsfw_label, _ = classify_nsfw(text)
    if nsfw_label != "non-nsfw":
        return None

    toxic_label, _ = classify_toxic_speech(text)
    if toxic_label != "non-toxic":
        return None

    text, _ = mask_emails(text)
    text, _ = mask_phone_numbers(text)
    text, _ = mask_ips(text)
    return text
