#!/usr/bin/env python3
"""
Prompt templates and formatting helpers for refining chapter boundaries.

This module produces multi-label chapter tagging prompts that verify whether
pages near a target chapter are correctly assigned during the refinement pass
after the initial boundary detection.

Composes prompts from shared components defined in shared_prompt_components.py
to maintain consistency with the initial detection module.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from string import Template
from typing import Dict, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from instructions import (
    BatchContent,
    OCRFormattingOptions,
    _load_abbreviation_table_text,
    approx_token_count,
    read_ocr_outputs,
    truncate_ocr_text,
)
from shared_prompt_components import (
    CHAPTER_BOUNDARY_SIGNALS,
    DOCUMENT_STRUCTURE,
    OCR_SOURCE_NOTES,
    OUTPUT_RULES,
    OVERFLOW_DETECTION,
    TAGGING_GUIDELINES,
    TAGGING_PHILOSOPHY,
    TAGGING_TRIGGERS,
    VERIFICATION_CHECKLIST,
)

logger = logging.getLogger(__name__)

# Preserve imported helpers for tooling parity with the parent instructions module.
_ = _load_abbreviation_table_text


# =============================================================================
# PROMPT TEMPLATES - COMPOSED FROM SHARED + REFINEMENT-SPECIFIC COMPONENTS
# =============================================================================

USER_INSTRUCTIONS_PROMPT = Template(
    f"""You are refining chapter boundaries for pages near Chapter ${{chapter}} in ${{book_context}}. Initial detection identified approximate boundaries; your task is to verify and correct page-to-chapter assignments.

## Context
These pages span the region around Chapter ${{chapter}}. Some pages may belong to Chapter ${{chapter}}, adjacent chapters, or multiple chapters at boundaries.

## Classification Rules
{TAGGING_PHILOSOPHY}

{TAGGING_GUIDELINES}

{TAGGING_TRIGGERS}

## Document Structure

{DOCUMENT_STRUCTURE}

{CHAPTER_BOUNDARY_SIGNALS}

{OVERFLOW_DETECTION}

{OCR_SOURCE_NOTES}

## Output Format
Return a JSON object mapping each page stem to an array of chapter numbers (as strings).
Example: {{"page_045": ["${{chapter}}"], "page_046": ["${{chapter}}", "${{adjacent_chapter}}"], "page_047": ["${{adjacent_chapter}}"]}}

Tag liberally—when in doubt, add the tag.
"""
)


SYSTEM_OUTPUT_PROMPT = Template(
    f"""You are a chapter boundary refinement system. Output ONLY a JSON object mapping page stems to arrays of chapter tags.

## Pages in This Batch
You are analyzing the following pages (use these EXACT keys):
${{pages_section}}

## Expected Chapters
The primary chapter being refined is Chapter ${{chapter}}. Adjacent chapters may also appear on these pages.

## Expected Output Format
Return a single JSON object with ALL page stems as keys. Example structure:
${{example_section}}

## Output Requirements

{OUTPUT_RULES}

{VERIFICATION_CHECKLIST}
"""
)


def build_refinement_user_prompt(
    chapter: str,
    page_stems: Sequence[str],
    book_name: str | None = None,
    adjacent_chapters: Sequence[str] | None = None,
) -> str:
    """Construct the user-facing instructions for refining pages near a chapter."""

    book_context = f"the book '{book_name}'" if book_name else "this book"

    adjacent_chapter = "?"
    if adjacent_chapters:
        for adj in adjacent_chapters:
            if adj != chapter:
                adjacent_chapter = adj
                break
    else:
        try:
            ch_num = int(chapter)
            adjacent_chapter = str(ch_num + 1)
        except ValueError:
            adjacent_chapter = "2" if chapter == "1" else "1"

    return USER_INSTRUCTIONS_PROMPT.safe_substitute(
        chapter=chapter,
        book_context=book_context,
        adjacent_chapter=adjacent_chapter,
    )


def get_expected_tagging_schema(page_stems: Sequence[str], target_chapter: str) -> str:
    """Generate an example JSON payload mapping stems to arrays of chapter tags."""

    if not page_stems:
        example_payload = {
            "page_045": [target_chapter],
            "page_046": [target_chapter],
            "page_047": [target_chapter],
            "page_048": [str(int(target_chapter) + 1) if target_chapter.isdigit() else "2"],
        }
        return json.dumps(example_payload, indent=2, ensure_ascii=False)

    example_payload = {}
    midpoint = max(1, len(page_stems) // 2)

    try:
        next_chapter = str(int(target_chapter) + 1) if target_chapter.isdigit() else "2"
    except ValueError:
        next_chapter = "2"

    for idx, stem in enumerate(page_stems):
        if idx < midpoint - 1:
            example_payload[stem] = [target_chapter]
        elif idx < midpoint + 1:
            example_payload[stem] = [target_chapter, next_chapter]
        else:
            example_payload[stem] = [next_chapter]

    return json.dumps(example_payload, indent=2, ensure_ascii=False)


def build_refinement_system_prompt(chapter: str, page_stems: Sequence[str]) -> str:
    """Construct the system prompt describing the expected JSON output."""

    pages_section = "\n".join(f"- {stem}" for stem in page_stems) if page_stems else "- [no pages provided]"
    example_section = get_expected_tagging_schema(page_stems, target_chapter=chapter)

    return SYSTEM_OUTPUT_PROMPT.safe_substitute(
        chapter=chapter,
        pages_section=pages_section,
        example_section=example_section,
    )


def format_refinement_batch_content(
    book_path: Path,
    page_stems: Sequence[str],
    cache: dict | None = None,
) -> BatchContent:
    """Format OCR content for a refinement batch using the shared structure."""

    opts = OCRFormattingOptions()
    blocks = []
    truncation_map: Dict[str, Dict[str, bool]] = {}

    for stem in page_stems:
        ocr_outputs, trunc_flags = read_ocr_outputs(book_path, stem, cache=cache, options=opts)
        truncation_map[stem] = trunc_flags

        blocks.append(f"=== PAGE: {stem} ===")
        blocks.append(f"<PAGE id=\"{stem}\">")
        for label, text in ocr_outputs.items():
            blocks.append(f"<OCR source=\"{label}\" page=\"{stem}\">\n{text}\n</OCR>")
        blocks.append("</PAGE>")

    text = "\n".join(blocks)
    return BatchContent(
        text=text,
        estimated_tokens=approx_token_count(text),
        truncation_map=truncation_map,
    )


__all__ = [
    "build_refinement_user_prompt",
    "build_refinement_system_prompt",
    "format_refinement_batch_content",
    "get_expected_tagging_schema",
    "OCRFormattingOptions",
    "truncate_ocr_text",
    "USER_INSTRUCTIONS_PROMPT",
    "SYSTEM_OUTPUT_PROMPT",
]
