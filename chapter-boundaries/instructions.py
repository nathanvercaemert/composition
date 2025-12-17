#!/usr/bin/env python3
"""
instructions.py

Prompt construction, OCR formatting, and tagging validation utilities for
chapter boundary detection. Composes prompts from shared components to
eliminate redundancy with the refinement module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Sequence, Tuple

from shared_prompt_components import (
    BOUNDARY_RULES,
    CHAPTER_BOUNDARY_SIGNALS,
    DOCUMENT_STRUCTURE_EXTENDED,
    OCR_ERROR_PATTERNS,
    OCR_SOURCE_NOTES_EXTENDED,
    OUTPUT_RULES,
    ROMAN_NUMERAL_NOTES,
    ROMAN_NUMERAL_TABLE,
    TAGGING_GUIDELINES,
    TAGGING_PHILOSOPHY,
    TAGGING_TRIGGERS,
    VERIFICATION_CHECKLIST,
)

logger = logging.getLogger(__name__)

# Constants
MIN_EXPECTED_TAGS_PER_PAGE = 2
MAPPINGS_PATH = Path("mappings.json")
_ABBREVIATION_TABLE_TEXT: str | None = None


@dataclass
class OCRFormattingOptions:
    """Options for trimming and packaging OCR text per provider."""

    head_chars: int = 1500
    tail_chars: int = 1500
    max_chars_per_provider: int = 4000


@dataclass
class BatchContent:
    """Formatted batch content plus lightweight metadata."""

    text: str
    estimated_tokens: int
    truncation_map: Dict[str, Dict[str, bool]]


def approx_token_count(text: str) -> int:
    """Rough token estimate (chars/4) to stay within model limits."""

    return max(1, len(text) // 4)


def truncate_ocr_text(text: str, options: OCRFormattingOptions) -> tuple[str, bool]:
    """Return possibly truncated OCR text and whether truncation occurred."""

    if options.max_chars_per_provider > 0 and len(text) <= options.max_chars_per_provider:
        return text, False

    head = text[: max(options.head_chars, 0)].strip()
    tail = text[-max(options.tail_chars, 0) :].strip() if options.tail_chars else ""
    truncated = len(text) > len(head) + len(tail)

    if head and tail:
        combined = f"{head}\n...\n{tail}"
    else:
        combined = head or tail

    if options.max_chars_per_provider > 0 and len(combined) > options.max_chars_per_provider:
        combined = combined[: options.max_chars_per_provider]
        truncated = True

    return combined, truncated


# =============================================================================
# INPUT FORMAT (Unique to initial detection)
# =============================================================================

INPUT_FORMAT_SECTION = """## Input Format
Each page is structured as:
```
=== PAGE: {page_stem} ===
<PAGE id="{page_stem}">
  <OCR source="DeepSeek OCR" page="{page_stem}">[text]</OCR>
  <OCR source="PaddleOCR-VL" page="{page_stem}">[text]</OCR>
  <OCR source="Qwen-VL" page="{page_stem}">[text]</OCR>
</PAGE>
```
- Use the PAGE id exactly as your JSON key
- There are three OCR sources for each PAGE id; synthesize them
- Include every PAGE id in output, even if OCR is garbled
- Use only PAGE id for keys"""


# =============================================================================
# TASK SECTION (Unique to initial detection)
# =============================================================================

TASK_SECTION = """## Your Task

Analyze OCR from three systems to identify:
1. Explicit chapter headings ("Chapter 1", "CHAPTER ONE", the chapter defined by roman numerals, etc.)
2. Verse-specific commentary sequences indicating chapter boundaries
3. Chapter content and transition pages"""


# =============================================================================
# AMBIGUOUS PAGE RULES (Unique to initial detection)
# =============================================================================

AMBIGUOUS_PAGE_RULES = """## Ambiguous Pages

- **Between chapters**: Tag with both preceding and following chapters
- **Before first labelled chapter**: Assign to Chapter 1 (use only numeric labels)
- **After the last labelled chapter**: Assign to the final chapter"""


# =============================================================================
# REASONING PROCESS (Unique to initial detection)
# =============================================================================

REASONING_PROCESS = """## Reasoning Process

1. Inventory all pages in the batch
2. Find explicit chapter markers
3. Analyze verse commentary marker numbers for resets and continuity
4. Assign initial tags based on evidence
5. Expand boundaries: extend tags a page or two in each direction, depending on confidence level
6. Fill gaps within chapter ranges
7. Check neighbors: if adjacent pages share a tag, consider adding it
8. Check overflow: if verse-specific commentary overflows to a page with a different chapter (or no chapter at all) in the heading, tag that page with both chapters.
9. Verify: every page has ≥1 tag, no chapters skipped in sequences"""


# =============================================================================
# EXAMPLES (Unique to initial detection)
# =============================================================================

EXAMPLES_SECTION = """## Examples

**Boundary tagging**:
```json
{
  "page_003": ["1", "2"],
  "page_004": ["1", "2"],
  "page_007": ["2", "3"],
  "page_008": ["2", "3"]
}
```

**Ambiguous content** (multiple plausible chapters):
```json
{
  "page_022": ["4", "5", "6"],
  "page_023": ["5", "6"],
  "page_024": ["5", "6", "7"]
}
```"""


# =============================================================================
# PROMPT TEMPLATES - COMPOSED FROM SHARED + UNIQUE COMPONENTS
# =============================================================================

USER_INSTRUCTIONS_PROMPT = f"""You are a document analyst identifying chapter boundaries in OCR-processed religious/scriptural texts with verse-specific commentary marker numbers.

$book_context

{INPUT_FORMAT_SECTION}

## Document Structure

{DOCUMENT_STRUCTURE_EXTENDED}

{TASK_SECTION}

## Tagging Policy

{TAGGING_PHILOSOPHY}

{TAGGING_GUIDELINES}

{TAGGING_TRIGGERS}

{BOUNDARY_RULES}

## Verse Number Patterns

{CHAPTER_BOUNDARY_SIGNALS}

{OCR_ERROR_PATTERNS}

{OCR_SOURCE_NOTES_EXTENDED}

{AMBIGUOUS_PAGE_RULES}

{REASONING_PROCESS}

{EXAMPLES_SECTION}

## Book Abbreviations
$abbreviation_tables

{ROMAN_NUMERAL_NOTES}

```json
{ROMAN_NUMERAL_TABLE}
```"""


SYSTEM_OUTPUT_PROMPT = f"""You are a chapter boundary detection system. Output ONLY a JSON object mapping page stems to arrays of chapter tags.

## Pages in This Batch
You are analyzing the following pages (these are the EXACT keys you must use in your JSON output):
$pages_section

## Expected Output Format
Return a single JSON object that includes ALL AND ONLY the above page stems as keys. Example structure (use the real chapter numbers you infer, and include every page):
$json_template_section

## Output Requirements

{OUTPUT_RULES}

{VERIFICATION_CHECKLIST}"""


# Legacy aliases for backward compatibility
_LEGACY_SYSTEM_PROMPT = "Deprecated legacy prompt."
SYSTEM_PROMPT_BASE = "Deprecated combined prompt."


def _load_abbreviation_table_text(path: Path | str | None = None) -> str:
    """Return formatted abbreviation tables for inclusion in the prompt."""
    global _ABBREVIATION_TABLE_TEXT
    if _ABBREVIATION_TABLE_TEXT:
        return _ABBREVIATION_TABLE_TEXT

    table_path = Path(path) if path else MAPPINGS_PATH
    try:
        mapping_text = table_path.read_text(encoding="utf-8")
        data = json.loads(mapping_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falling back: unable to read abbreviation mappings: %s", exc)
        fallback = "Abbreviation table unavailable (failed to load mappings.json)."
        _ABBREVIATION_TABLE_TEXT = fallback
        return fallback

    lines = [
        "Full book abbreviation table (Old Testament, Deuterocanonical/Apocrypha, New Testament):",
    ]
    for book, aliases in data.items():
        alias_text = ", ".join(aliases)
        lines.append(f"- {book}: {alias_text}")
    _ABBREVIATION_TABLE_TEXT = "\n".join(lines)
    return _ABBREVIATION_TABLE_TEXT


def get_expected_json_schema(page_stems: Sequence[str]) -> str:
    """Generate a sample JSON payload showing expected output format."""
    example_payload = (
        {stem: ["1", "2"] for stem in page_stems} if page_stems else {"page_001": ["1", "2"]}
    )
    return json.dumps(example_payload, indent=2, ensure_ascii=False)


def build_user_prompt_preamble(page_stems: Sequence[str], book_name: str | None = None) -> str:
    """
    Construct the instructional preamble that now lives at the start of the user prompt.
    All guidance except output-format rules is provided here.
    """
    _ = page_stems  # retained for parity and possible future use
    book_context = (
        f"You are analyzing the book: {book_name}"
        if book_name
        else "You are analyzing a single book at a time; treat the pages as one continuous work."
    )
    template = Template(USER_INSTRUCTIONS_PROMPT)
    return template.safe_substitute(
        book_context=book_context,
        abbreviation_tables=_load_abbreviation_table_text(),
    )


def build_system_output_prompt(page_stems: Sequence[str]) -> str:
    """Construct the system prompt containing only output-format rules."""
    page_list = "\n".join(f"- {stem}" for stem in page_stems) if page_stems else "- [no pages provided]"
    example_payload = get_expected_json_schema(page_stems)
    template = Template(SYSTEM_OUTPUT_PROMPT)
    return template.safe_substitute(
        pages_section=page_list,
        json_template_section=example_payload,
    )


def build_system_prompt(page_stems: Sequence[str], book_name: str | None = None) -> str:
    """Backward-compatible alias for the output-only system prompt."""
    _ = book_name  # unused but retained for call compatibility
    return build_system_output_prompt(page_stems)


def read_ocr_outputs(
    book_path: Path,
    page_stem: str,
    *,
    cache: Dict[tuple[str, str], str] | None = None,
    options: OCRFormattingOptions | None = None,
) -> tuple[Dict[str, str], Dict[str, bool]]:
    """Read OCR outputs for a given page stem from all providers.

    Returns a tuple of (provider->text, provider->truncated_flag).
    """

    opts = options or OCRFormattingOptions()
    sources = {
        "DeepSeek OCR": book_path / "deepseek-ocr" / f"{page_stem}.txt",
        "PaddleOCR-VL": book_path / "paddleocr-vl" / f"{page_stem}.txt",
        "Qwen-VL": book_path / "qwen-vl" / f"{page_stem}.txt",
    }
    outputs: Dict[str, str] = {}
    truncated: Dict[str, bool] = {}

    for label, path in sources.items():
        cache_key = (page_stem, label)
        if cache is not None and cache_key in cache:
            outputs[label] = cache[cache_key]
            truncated[label] = False
            continue

        text = "[OCR output not available]"
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed reading %s: %s", path, exc)
                text = "[OCR output could not be read]"

        processed, was_truncated = truncate_ocr_text(text, opts) if text else ("", False)
        outputs[label] = processed
        truncated[label] = was_truncated
        if cache is not None:
            cache[cache_key] = processed

    return outputs, truncated


def format_batch_content(
    book_path: Path,
    page_stems: Sequence[str],
    *,
    cache: Dict[tuple[str, str], str] | None = None,
    options: OCRFormattingOptions | None = None,
) -> BatchContent:
    """Format a batch of pages into the prompt content with delimiters."""

    opts = options or OCRFormattingOptions()
    blocks: List[str] = []
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


def validate_tagging_density(
    page_tags: Dict[str, List[str]],
    min_tags_per_page: float = MIN_EXPECTED_TAGS_PER_PAGE,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate tagging density against liberal tagging requirements.

    Returns:
        (is_valid, stats_dict)
    """
    tag_counts = [len(tags) for tags in page_tags.values()] if page_tags else []
    total_pages = len(page_tags)
    total_tags = sum(tag_counts)
    avg_tags = total_tags / total_pages if total_pages else 0.0
    min_tags = min(tag_counts) if tag_counts else 0
    max_tags = max(tag_counts) if tag_counts else 0
    single_tag_pages = sum(1 for count in tag_counts if count == 1)
    zero_tag_pages = sum(1 for count in tag_counts if count == 0)

    stats = {
        "total_pages": total_pages,
        "total_tags": total_tags,
        "avg_tags_per_page": avg_tags,
        "min_tags": min_tags,
        "max_tags": max_tags,
        "single_tag_pages": single_tag_pages,
        "zero_tag_pages": zero_tag_pages,
        "threshold": min_tags_per_page,
    }

    is_valid = avg_tags >= min_tags_per_page and zero_tag_pages == 0
    return is_valid, stats


__all__ = [
    "OCRFormattingOptions",
    "BatchContent",
    "approx_token_count",
    "truncate_ocr_text",
    "build_user_prompt_preamble",
    "build_system_output_prompt",
    "build_system_prompt",
    "format_batch_content",
    "read_ocr_outputs",
    "get_expected_json_schema",
    "validate_tagging_density",
    "USER_INSTRUCTIONS_PROMPT",
    "SYSTEM_OUTPUT_PROMPT",
    "SYSTEM_PROMPT_BASE",
    "MIN_EXPECTED_TAGS_PER_PAGE",
    "MAPPINGS_PATH",
    "_load_abbreviation_table_text",
]
