from __future__ import annotations

"""
Structured commentary toolkit.

This module re-exports the primary entry points for convenient access.
"""

from preprocess import normalize_newlines, preprocess_commentary, remove_leading_verse_marker
from structure_all import run_structuring
from structure_chapter import structure_chapter
from structure_verse import structure_verse_commentary

__all__ = [
    "normalize_newlines",
    "preprocess_commentary",
    "remove_leading_verse_marker",
    "run_structuring",
    "structure_chapter",
    "structure_verse_commentary",
]

