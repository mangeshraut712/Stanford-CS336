"""PII masking utilities."""

from __future__ import annotations

import regex as re

EMAIL_PLACEHOLDER = "|||EMAIL_ADDRESS|||"
PHONE_PLACEHOLDER = "|||PHONE_NUMBER|||"
IP_PLACEHOLDER = "|||IP_ADDRESS|||"

EMAIL_PATTERN = re.compile(
    r"(?<![\w.])"
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    r"(?![\w.])"
)
PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\(\d{3}\)[\s\-]?\d{3}[\s\-]?\d{4}|\d{3}[\s\-]?\d{3}[\s\-]?\d{4}|\d{10})"
    r"(?!\d)"
)
IP_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\d{1,3}\.){3}\d{1,3}"
    r"(?!\d)"
)


def _mask_with_pattern(text: str, pattern: re.Pattern, placeholder: str) -> tuple[str, int]:
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return placeholder

    return pattern.sub(repl, text), count


def mask_emails(text: str) -> tuple[str, int]:
    return _mask_with_pattern(text, EMAIL_PATTERN, EMAIL_PLACEHOLDER)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return _mask_with_pattern(text, PHONE_PATTERN, PHONE_PLACEHOLDER)


def mask_ips(text: str) -> tuple[str, int]:
    return _mask_with_pattern(text, IP_PATTERN, IP_PLACEHOLDER)
