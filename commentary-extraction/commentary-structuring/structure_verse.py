from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure local modules and shared utilities are importable
HERE = Path(__file__).resolve().parent
COMMENTARY_EXTRACTION = HERE.parent
PROJECT_ROOT = HERE.parent.parent
CHAPTER_BOUNDARIES = PROJECT_ROOT / "chapter-boundaries"

for path in (HERE, COMMENTARY_EXTRACTION, PROJECT_ROOT, CHAPTER_BOUNDARIES):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from model import query_structured_with_metadata  # noqa: E402
from preprocess import preprocess_commentary  # noqa: E402
from structuring_prompts import build_structuring_system_prompt, build_structuring_user_prompt  # noqa: E402
from structuring_schema import STRUCTURED_COMMENTARY_SCHEMA  # noqa: E402

logger = logging.getLogger(__name__)


def structure_verse_commentary(
    book_name: str,
    chapter_number: int,
    verse_number: int,
    commentary_text: str,
    *,
    reasoning_effort: str = "low",
    system_prompt: str | None = None,
    preprocessed_commentary: str | None = None,
    user_prompt: str | None = None,
    dry_run: bool = False,
) -> Tuple[List[dict], Optional[Dict[str, Any]]]:
    """
    Structure a single verse commentary into ordered parts using the model.

    Args:
        book_name: Book abbreviation.
        chapter_number: Chapter number (1-indexed).
        verse_number: Verse number (1-indexed).
        commentary_text: Raw commentary text for the verse.
        reasoning_effort: Reasoning effort for GPT-5.2 ("low", "medium", "high", "xhigh").
        system_prompt: Optional pre-built system prompt (reused for caching).
        preprocessed_commentary: Optional preprocessed commentary to avoid rework.
        user_prompt: Optional user prompt override (primarily for debugging).
        dry_run: When True, skip the model call and return an empty parts list.

    Returns:
        Tuple of (structured parts list, usage metadata dict). Usage is None in dry-run mode.

    Raises:
        ValueError: On malformed model responses.
        Exception: Bubble up model or validation errors for caller to handle.
    """
    preprocessed = preprocessed_commentary if preprocessed_commentary is not None else preprocess_commentary(commentary_text)
    sys_prompt = system_prompt or build_structuring_system_prompt(book_name, chapter_number)
    usr_prompt = user_prompt or build_structuring_user_prompt(book_name, chapter_number, verse_number, preprocessed)

    if dry_run:
        logger.info("Dry-run enabled; skipping model call for %s %s:%s", book_name, chapter_number, verse_number)
        return [], None

    result = query_structured_with_metadata(
        system_prompt=sys_prompt,
        user_message=usr_prompt,
        response_schema=STRUCTURED_COMMENTARY_SCHEMA,
        schema_name="structured_commentary_parts",
        use_reasoning=True,
        reasoning_effort=reasoning_effort,
    )

    structured_data = result.get("data")
    usage = result.get("usage")

    if isinstance(structured_data, dict):
        structured_parts = structured_data.get("parts")
        if structured_parts is None:
            raise ValueError(
                f"Model returned object without 'parts' for {book_name} {chapter_number}:{verse_number}: {structured_data!r}"
            )
    else:
        structured_parts = structured_data

    if not isinstance(structured_parts, list):
        raise ValueError(
            f"Model returned non-list parts for {book_name} {chapter_number}:{verse_number}: {structured_parts!r}"
        )

    return structured_parts, usage


__all__ = ["structure_verse_commentary"]

