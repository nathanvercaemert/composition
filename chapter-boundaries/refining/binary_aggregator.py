#!/usr/bin/env python3
"""Utilities for validating and aggregating chapter membership outputs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import natural_key

logger = logging.getLogger(__name__)


def _normalize_chapter_tags(raw_value: Any) -> List[str]:
    """Coerce various tag shapes (scalar, list) into a clean list of strings."""

    values: List[Any]
    if isinstance(raw_value, list):
        values = raw_value
    elif raw_value is None:
        return []
    else:
        values = [raw_value]

    normalized: List[str] = []
    for val in values:
        if val is None:
            continue
        text = str(val).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def validate_tagging_response(response: Any, expected_stems: Sequence[str]) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """Validate multi-label tagging response structure.

    Args:
        response: The model's JSON response (should be dict of page -> list of chapters)
        expected_stems: The page stems we expect in the response

    Returns:
        valid_mapping: dict of page stem -> list of chapter tags (only for valid entries)
        missing_pages: expected stems not present or not usable
        unexpected_pages: keys returned by the model that were not expected
    """

    expected_lookup = {stem: stem for stem in expected_stems}
    expected_keys: Set[str] = set(expected_lookup.keys())
    valid: Dict[str, List[str]] = {}
    unexpected: List[str] = []

    if not isinstance(response, dict):
        logger.warning("Model response is not a dict; received type: %s", type(response))
        return valid, list(expected_lookup.values()), unexpected

    for raw_page, raw_chapters in response.items():
        page_key = str(raw_page).strip()
        if page_key not in expected_keys:
            unexpected.append(page_key)
            continue

        normalized = _normalize_chapter_tags(raw_chapters)
        if normalized:
            valid[expected_lookup[page_key]] = normalized
        else:
            logger.warning("Page %s has empty or invalid chapter list %r; treating as missing", page_key, raw_chapters)

    missing = [stem for stem in expected_stems if stem not in valid]
    return valid, missing, unexpected


def validate_binary_response(response: Any, expected_stems: Sequence[str]) -> Tuple[Dict[str, bool], List[str], List[str]]:
    """Validate binary classification response structure.

    DEPRECATED: Kept for backward compatibility. New flows should use validate_tagging_response.

    Returns:
        valid_mapping: dict of page stem -> bool (only for valid keys/values)
        missing_pages: expected stems not present or not usable
        unexpected_pages: keys returned by the model that were not expected
    """

    def _coerce_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0", "none", "null"}:
                return False
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            if value == 0:
                return False
        return None

    expected_lookup = {stem: stem for stem in expected_stems}
    expected_keys: Set[str] = set(expected_lookup.keys())
    valid: Dict[str, bool] = {}
    unexpected: List[str] = []

    if not isinstance(response, dict):
        logger.warning("Model response is not a dict; received type: %s", type(response))
        return valid, list(expected_lookup.values()), unexpected

    for raw_page, raw_value in response.items():
        page_key = str(raw_page).strip()
        if page_key not in expected_keys:
            unexpected.append(page_key)
            continue

        coerced = _coerce_bool(raw_value)
        if coerced is None:
            logger.warning("Page %s has non-boolean value %r; treating as missing", page_key, raw_value)
            continue

        valid[expected_lookup[page_key]] = coerced

    missing = [stem for stem in expected_stems if stem not in valid]
    return valid, missing, unexpected


def extract_chapter_pages(tagging_response: Dict[str, List[str]], target_chapter: str) -> List[str]:
    """Extract pages that have the target chapter in their tag list."""

    matching_pages = [stem for stem, chapters in tagging_response.items() if target_chapter in chapters]
    return sorted(matching_pages, key=natural_key)


def merge_chapter_memberships(chapter_results: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """Convert per-chapter tagging results into final chapter->pages mapping."""

    merged: Dict[str, List[str]] = {}

    for chapter, page_tags in chapter_results.items():
        if not isinstance(page_tags, dict):
            merged[chapter] = []
            continue

        members = [
            stem
            for stem, tags in page_tags.items()
            if isinstance(tags, list) and chapter in tags
        ]
        merged[chapter] = sorted(members, key=natural_key)

    return merged


def detect_unassigned_pages(all_pages: Sequence[str], chapter_pages: Dict[str, List[str]]) -> List[str]:
    """Return pages that were not assigned to any chapter."""

    assigned = {stem for pages in chapter_pages.values() for stem in pages}
    return [stem for stem in all_pages if stem not in assigned]


def _group_contiguous(stems: List[str], index_map: Dict[str, int]) -> List[List[str]]:
    """Group stems into contiguous sequences based on their positions."""

    if not stems:
        return []

    ordered = sorted(stems, key=lambda s: index_map.get(s, float("inf")))
    groups: List[List[str]] = []
    current: List[str] = [ordered[0]]

    for prev, current_stem in zip(ordered, ordered[1:]):
        prev_idx = index_map.get(prev)
        curr_idx = index_map.get(current_stem)
        if prev_idx is None or curr_idx is None or curr_idx != prev_idx + 1:
            groups.append(current)
            current = [current_stem]
        else:
            current.append(current_stem)

    groups.append(current)
    return groups


def detect_boundary_conflicts(chapter_pages: Dict[str, List[str]], all_pages: Sequence[str]) -> List[Dict[str, Any]]:
    """Identify possible boundary issues (overlaps, gaps, discontinuities)."""

    conflicts: List[Dict[str, Any]] = []
    index_map = {stem: idx for idx, stem in enumerate(all_pages)}
    page_to_chapters: Dict[str, List[str]] = {}

    for chapter, pages in chapter_pages.items():
        missing = [stem for stem in pages if stem not in index_map]
        if missing:
            conflicts.append(
                {
                    "type": "unknown_page",
                    "chapter": chapter,
                    "pages": sorted(missing, key=natural_key),
                    "detail": "Pages not found in global ordering.",
                }
            )

        for stem in pages:
            page_to_chapters.setdefault(stem, []).append(chapter)

    for stem, chapters in page_to_chapters.items():
        if len(chapters) > 1:
            conflicts.append(
                {
                    "type": "overlap",
                    "page": stem,
                    "chapters": sorted(chapters, key=natural_key),
                    "detail": "Page assigned to multiple chapters.",
                }
            )

    unassigned = [stem for stem in all_pages if stem not in page_to_chapters]
    if unassigned:
        gaps = _group_contiguous(unassigned, index_map)
        for gap in gaps:
            conflicts.append(
                {
                    "type": "gap",
                    "pages": gap,
                    "detail": "Contiguous pages unassigned to any chapter.",
                }
            )

    for chapter, pages in chapter_pages.items():
        ordered = [stem for stem in sorted(set(pages), key=lambda s: index_map.get(s, float("inf"))) if stem in index_map]
        segments = _group_contiguous(ordered, index_map)
        if len(segments) > 1:
            conflicts.append(
                {
                    "type": "discontinuity",
                    "chapter": chapter,
                    "segments": segments,
                    "detail": "Chapter pages are not contiguous; possible boundary gap.",
                }
            )

    return conflicts


__all__ = [
    "validate_binary_response",
    "validate_tagging_response",
    "extract_chapter_pages",
    "merge_chapter_memberships",
    "detect_unassigned_pages",
    "detect_boundary_conflicts",
]

