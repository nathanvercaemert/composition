#!/usr/bin/env python3
"""
Commentary extraction orchestrator.

Extracts verse-specific commentary from OCR-processed religious texts,
processing one verse at a time for a specified book/chapter combination.

Usage:
    python commentary_extraction.py BOOK_NAME CHAPTER_NUMBER [options]

Examples:
    python commentary_extraction.py Job 1
    python commentary_extraction.py Job 1 --force --debug
    python commentary_extraction.py Gen 3 --delay 0.5
    python commentary_extraction.py Job 1 --no-reasoning
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Optional

# Ensure project root is on path for verses.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from verses import VersesData  # noqa: E402

from extraction_instructions import (  # noqa: E402
    build_extraction_system_prompt,
    build_extraction_user_prompt,
)
from extraction_schemas import VERSE_COMMENTARY_SCHEMA  # noqa: E402
from model import REASONING_EFFORT, query_structured, query_structured_with_metadata  # noqa: E402
from ocr_loader import load_chapter_boundaries, load_chapter_ocr  # noqa: E402

logger = logging.getLogger(__name__)


def approx_token_count(text: str) -> int:
    """Rough token estimate (chars/4) for logging purposes."""
    return max(1, len(text) // 4)


def extract_chapter_commentary(
    book_name: str,
    chapter_number: int,
    *,
    force: bool = False,
    debug: bool = False,
    delay: float = 0.5,
    temp_dir: Path = Path("temp/commentary-extraction"),
    output_dir: Path | None = None,
    tokens: bool = False,
    no_reasoning: bool = False,
    low_reasoning: bool = False,
    med_reasoning: bool = False,
    high_reasoning: bool = False,
) -> Optional[Dict[str, object]]:
    """
    Extract commentary for all verses in a chapter.

    Process:
    1. Load verse count from verses.py
    2. Load chapter boundaries from chapter_boundaries.json
    3. Load all OCR content for the chapter's pages
    4. Build the STATIC system prompt (includes all OCR content)
    5. For each verse 1 to N:
       a. Build the verse-specific user prompt
       b. Call model.query_structured() with the SAME system prompt
       c. Parse and store the result
       d. Save debug artifacts if enabled
       e. Wait delay seconds before next call
    6. Assemble final output JSON
    7. Write to books/BOOK_NAME/chapters/CHAPTER_NUMBER.json
    """

    verses_data = VersesData()

    if not verses_data.has_book(book_name):
        logger.error("Unknown book abbreviation: %s", book_name)
        return None

    try:
        chapter_count = verses_data.get_chapter_count(book_name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Unable to get chapter count for %s: %s", book_name, exc)
        return None

    if chapter_number < 1 or chapter_number > chapter_count:
        logger.error("Chapter %s out of range for %s (1-%s)", chapter_number, book_name, chapter_count)
        return None

    total_verses = verses_data.get_verse_count(book_name, chapter_number)

    book_path = Path("books") / book_name
    boundaries_path = book_path / "chapter_boundaries.json"
    boundaries = load_chapter_boundaries(boundaries_path)
    chapter_pages = boundaries.get(str(chapter_number), [])

    if not chapter_pages:
        logger.error("No pages found for chapter %s in %s", chapter_number, boundaries_path)
        return None

    chapter_ocr_content = load_chapter_ocr(book_path, chapter_pages)

    system_prompt = build_extraction_system_prompt(
        book_name=book_name,
        chapter_number=chapter_number,
        total_verses=total_verses,
        chapter_ocr_content=chapter_ocr_content,
    )

    logger.info("System prompt size: %s chars (~%s tokens)", len(system_prompt), approx_token_count(system_prompt))

    debug_dir = temp_dir / book_name / str(chapter_number)
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        (debug_dir / "chapter_ocr_content.txt").write_text(chapter_ocr_content, encoding="utf-8")
        extraction_meta = {
            "book": book_name,
            "chapter": chapter_number,
            "pages": chapter_pages,
            "total_verses": total_verses,
            "system_prompt_chars": len(system_prompt),
            "system_prompt_approx_tokens": approx_token_count(system_prompt),
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        (debug_dir / "extraction_meta.json").write_text(
            json.dumps(extraction_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    results: Dict[str, Dict[str, object]] = {}
    total_input_tokens = 0
    total_cached_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0

    use_reasoning = not no_reasoning
    if low_reasoning:
        reasoning_effort = "low"
    elif med_reasoning:
        reasoning_effort = "medium"
    elif high_reasoning:
        reasoning_effort = "high"
    else:
        reasoning_effort = REASONING_EFFORT

    for verse_num in range(1, total_verses + 1):
        logger.info("Extracting verse %s/%s", verse_num, total_verses)
        user_prompt = build_extraction_user_prompt(verse_num)
        verse_key = str(verse_num)
        response_data: Dict[str, object]

        try:
            if tokens:
                result = query_structured_with_metadata(
                    system_prompt=system_prompt,
                    user_message=user_prompt,
                    response_schema=VERSE_COMMENTARY_SCHEMA,
                    schema_name="verse_commentary",
                    use_reasoning=use_reasoning,
                    reasoning_effort=reasoning_effort,
                )
                response_data = result["data"]
                usage = result.get("usage") or {}

                input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0

                input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
                output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}

                cached_tokens = input_details.get("cached_tokens") or 0
                reasoning_tokens = output_details.get("reasoning_tokens") or 0

                total_input_tokens += input_tokens
                total_cached_tokens += cached_tokens
                total_output_tokens += output_tokens
                total_reasoning_tokens += reasoning_tokens
            else:
                response_data = query_structured(
                    system_prompt=system_prompt,
                    user_message=user_prompt,
                    response_schema=VERSE_COMMENTARY_SCHEMA,
                    schema_name="verse_commentary",
                    use_reasoning=use_reasoning,
                    reasoning_effort=reasoning_effort,
                )
        except Exception as exc:  # pragma: no cover - runtime error path
            logger.error("Failed to extract verse %s: %s", verse_num, exc)
            response_data = {"has_commentary": False, "commentary": None, "error": str(exc)}

        results[verse_key] = response_data

        if debug:
            verse_dir = debug_dir / "verses" / f"verse_{verse_num:03d}"
            verse_dir.mkdir(parents=True, exist_ok=True)
            (verse_dir / "user_prompt.txt").write_text(user_prompt, encoding="utf-8")
            (verse_dir / "response.json").write_text(
                json.dumps(response_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (verse_dir / "extraction_result.json").write_text(
                json.dumps(response_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            reasoning_path = verse_dir / "reasoning.txt"
            reasoning_path.write_text("Structured call; reasoning not available.", encoding="utf-8")

        if verse_num < total_verses and delay > 0:
            time.sleep(delay)

    output = {
        "book": book_name,
        "chapter": chapter_number,
        "verses": results,
        "metadata": {
            "total_verses": total_verses,
            "verses_with_commentary": sum(1 for v in results.values() if v.get("has_commentary")),
            "verses_without_commentary": sum(1 for v in results.values() if not v.get("has_commentary")),
            "extraction_timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    }

    final_output_dir = output_dir or (book_path / "chapters")
    final_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = final_output_dir / f"{chapter_number}.json"

    if output_path.exists() and not force:
        reply = input(f"{output_path} exists. Overwrite? [y/N]: ").strip().lower()
        if reply not in {"y", "yes"}:
            logger.info("Aborted.")
            return None

    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote commentary to %s", output_path)

    if tokens:
        print("\n" + "=" * 50)
        print("TOKEN USAGE SUMMARY")
        print("=" * 50)
        print(f"Total Input Tokens:     {total_input_tokens:,}")
        print(f"  └─ Cached Tokens:     {total_cached_tokens:,}")
        print(f"Total Output Tokens:    {total_output_tokens:,}")
        print(f"  └─ Reasoning Tokens:  {total_reasoning_tokens:,}")
        print(f"Grand Total:            {total_input_tokens + total_output_tokens:,}")
        if total_input_tokens > 0:
            cache_rate = (total_cached_tokens / total_input_tokens) * 100
            print(f"Cache Hit Rate:         {cache_rate:.1f}%")
        print("=" * 50)

    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract verse-specific commentary for a chapter.")
    parser.add_argument("book_name", help="Book abbreviation (e.g., Job, Gen, Ps)")
    parser.add_argument("chapter_number", type=int, help="Chapter number to extract (1-indexed)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    parser.add_argument("--debug", action="store_true", help="Save intermediate artifacts to temp directory")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds)")
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("temp/commentary-extraction"),
        help="Directory for intermediate data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for final output (default: books/BOOK_NAME/chapters/)",
    )
    parser.add_argument(
        "--tokens",
        action="store_true",
        help="Display cumulative token usage statistics after processing",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable reasoning in API calls (omit the reasoning configuration)",
    )
    parser.add_argument(
        "--low-reasoning",
        action="store_true",
        help="Use low reasoning effort instead of extra-high (faster, cheaper, less thorough)",
    )
    parser.add_argument(
        "--med-reasoning",
        action="store_true",
        help="Use medium reasoning effort instead of extra-high (balanced speed/cost vs thoroughness)",
    )
    parser.add_argument(
        "--high-reasoning",
        action="store_true",
        help="Use high reasoning effort instead of extra-high (more thorough than medium, cheaper than extra-high)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    extract_chapter_commentary(
        book_name=args.book_name,
        chapter_number=args.chapter_number,
        force=args.force,
        debug=args.debug,
        delay=args.delay,
        temp_dir=args.temp_dir,
        output_dir=args.output_dir,
        tokens=args.tokens,
        no_reasoning=args.no_reasoning,
        low_reasoning=args.low_reasoning,
        med_reasoning=args.med_reasoning,
        high_reasoning=args.high_reasoning,
    )


if __name__ == "__main__":
    main()

