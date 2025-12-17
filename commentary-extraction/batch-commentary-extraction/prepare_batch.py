#!/usr/bin/env python3
"""Prepare OpenAI Batch API requests for commentary extraction."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Ensure project root and batch tools dir are importable (verses.py, shared utils)
_BATCH_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BATCH_DIR.parent
for _path in (_PROJECT_ROOT, _BATCH_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from extraction_instructions import (  # noqa: E402
    build_extraction_system_prompt,
    build_extraction_user_prompt,
)
from extraction_schemas import VERSE_COMMENTARY_SCHEMA  # noqa: E402
from model import MODEL_NAME  # noqa: E402
from ocr_loader import load_chapter_boundaries, load_chapter_ocr  # noqa: E402
from verses import VersesData  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("temp/batch-commentary-extraction")
DEFAULT_REASONING_EFFORT = "high"
SCHEMA_NAME = "verse_commentary"


@dataclass(frozen=True)
class ReasoningConfig:
    """Capture reasoning toggle and effort."""

    use_reasoning: bool
    effort: str | None


def approx_token_count(text: str) -> int:
    """Rough token estimate (chars/4) for metadata."""
    return max(1, len(text) // 4)


def _select_reasoning_from_flags(
    no_reasoning: bool,
    low_reasoning: bool,
    med_reasoning: bool,
    high_reasoning: bool,
    xhigh_reasoning: bool,
) -> ReasoningConfig:
    """
    Determine reasoning configuration from CLI flags.

    Priority (first true wins): no_reasoning -> low -> medium -> high -> xhigh -> default high.
    """
    if no_reasoning:
        return ReasoningConfig(use_reasoning=False, effort=None)
    if low_reasoning:
        return ReasoningConfig(use_reasoning=True, effort="low")
    if med_reasoning:
        return ReasoningConfig(use_reasoning=True, effort="medium")
    if high_reasoning:
        return ReasoningConfig(use_reasoning=True, effort="high")
    if xhigh_reasoning:
        return ReasoningConfig(use_reasoning=True, effort="xhigh")
    return ReasoningConfig(use_reasoning=True, effort=DEFAULT_REASONING_EFFORT)


def _build_request_object(
    *,
    custom_id: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: Dict[str, object],
    schema_name: str,
    use_reasoning: bool,
    reasoning_effort: str | None,
) -> Dict[str, object]:
    """Construct a single JSONL request entry for the Responses API."""
    body: Dict[str, object] = {
        "model": MODEL_NAME,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": response_schema,
                "strict": True,
            }
        },
    }
    body["prompt_cache_retention"] = "24h"
    if use_reasoning and reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def prepare_batch_requests(
    book_name: str,
    chapter_number: int,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    no_reasoning: bool = False,
    low_reasoning: bool = False,
    med_reasoning: bool = False,
    high_reasoning: bool = False,
    xhigh_reasoning: bool = False,
    debug: bool = False,
) -> Tuple[Path, Path]:
    """
    Generate the batch JSONL file and metadata for a book/chapter.

    Returns:
        Tuple of (jsonl_path, meta_path).
    """
    verses_data = VersesData()
    if not verses_data.has_book(book_name):
        raise ValueError(f"Unknown book abbreviation: {book_name}")

    chapter_count = verses_data.get_chapter_count(book_name)
    if chapter_number < 1 or chapter_number > chapter_count:
        raise ValueError(f"Chapter {chapter_number} out of range for {book_name} (1-{chapter_count})")

    total_verses = verses_data.get_verse_count(book_name, chapter_number)
    book_path = Path("books") / book_name
    boundaries_path = book_path / "chapter_boundaries.json"
    boundaries = load_chapter_boundaries(boundaries_path)
    chapter_pages = boundaries.get(str(chapter_number), [])
    if not chapter_pages:
        raise FileNotFoundError(f"No pages found for chapter {chapter_number} in {boundaries_path}")

    chapter_ocr_content = load_chapter_ocr(book_path, chapter_pages)
    system_prompt = build_extraction_system_prompt(
        book_name=book_name,
        chapter_number=chapter_number,
        total_verses=total_verses,
        chapter_ocr_content=chapter_ocr_content,
    )

    reasoning = _select_reasoning_from_flags(
        no_reasoning=no_reasoning,
        low_reasoning=low_reasoning,
        med_reasoning=med_reasoning,
        high_reasoning=high_reasoning,
        xhigh_reasoning=xhigh_reasoning,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{book_name}_{chapter_number}_requests.jsonl"
    meta_path = output_dir / f"{book_name}_{chapter_number}_meta.json"

    logger.info("Writing batch requests to %s", jsonl_path)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for verse_num in range(1, total_verses + 1):
            custom_id = f"{book_name}-{chapter_number}-verse-{verse_num}"
            user_prompt = build_extraction_user_prompt(verse_num)
            request_obj = _build_request_object(
                custom_id=custom_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=VERSE_COMMENTARY_SCHEMA,
                schema_name=SCHEMA_NAME,
                use_reasoning=reasoning.use_reasoning,
                reasoning_effort=reasoning.effort,
            )
            fh.write(json.dumps(request_obj, ensure_ascii=False))
            fh.write("\n")

    meta = {
        "book": book_name,
        "chapter": chapter_number,
        "total_verses": total_verses,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "system_prompt_chars": len(system_prompt),
        "system_prompt_approx_tokens": approx_token_count(system_prompt),
        "reasoning_effort": reasoning.effort if reasoning.use_reasoning else None,
        "use_reasoning": reasoning.use_reasoning,
        "output_jsonl": str(jsonl_path),
        "boundaries_path": str(boundaries_path),
        "pages": chapter_pages,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if debug:
        debug_dir = output_dir / f"{book_name}_{chapter_number}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        (debug_dir / "chapter_ocr_content.txt").write_text(chapter_ocr_content, encoding="utf-8")
        (debug_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Prepared %s requests for %s %s", total_verses, book_name, chapter_number)
    return jsonl_path, meta_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OpenAI Batch API JSONL requests for a chapter.")
    parser.add_argument("book_name", help="Book abbreviation (e.g., Job, Gen, Ps)")
    parser.add_argument("chapter_number", type=int, help="Chapter number to extract (1-indexed)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSONL output (default: temp/batch-commentary-extraction)",
    )
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning in API calls")
    parser.add_argument("--low-reasoning", action="store_true", help="Use low reasoning effort")
    parser.add_argument("--med-reasoning", action="store_true", help="Use medium reasoning effort")
    parser.add_argument("--high-reasoning", action="store_true", help="Use high reasoning effort (default)")
    parser.add_argument("--xhigh-reasoning", action="store_true", help="Use extra-high reasoning effort")
    parser.add_argument("--debug", action="store_true", help="Save additional debug artifacts")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    prepare_batch_requests(
        book_name=args.book_name,
        chapter_number=args.chapter_number,
        output_dir=args.output_dir,
        no_reasoning=args.no_reasoning,
        low_reasoning=args.low_reasoning,
        med_reasoning=args.med_reasoning,
        high_reasoning=args.high_reasoning,
        xhigh_reasoning=args.xhigh_reasoning,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()

