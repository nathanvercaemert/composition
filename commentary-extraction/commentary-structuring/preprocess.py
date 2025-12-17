from __future__ import annotations

"""
Preprocessing utilities for commentary structuring.

Steps:
1) Strip leading verse number markers like "12." at the start of the text.
2) Collapse consecutive newlines into a single newline.
3) Trim leading/trailing whitespace.
"""

import re
from typing import Final

LEADING_VERSE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\s*\d+\.\s*")
MULTIPLE_NEWLINES_PATTERN: Final[re.Pattern[str]] = re.compile(r"\n{2,}")


def remove_leading_verse_marker(text: str) -> str:
    """Remove a leading digit+dot marker (e.g., ``'12. '``) if present."""
    if not text:
        return text
    return LEADING_VERSE_PATTERN.sub("", text, count=1)


def normalize_newlines(text: str) -> str:
    """Collapse runs of two or more newlines down to a single newline."""
    if not text:
        return text
    return MULTIPLE_NEWLINES_PATTERN.sub("\n", text)


def preprocess_commentary(text: str) -> str:
    """
    Apply all preprocessing steps to a commentary string.

    Order:
    1. Strip leading verse marker.
    2. Collapse multiple newlines.
    3. Trim leading/trailing whitespace.
    """
    cleaned = remove_leading_verse_marker(text or "")
    cleaned = normalize_newlines(cleaned)
    return cleaned.strip()


__all__ = [
    "preprocess_commentary",
    "remove_leading_verse_marker",
    "normalize_newlines",
]

