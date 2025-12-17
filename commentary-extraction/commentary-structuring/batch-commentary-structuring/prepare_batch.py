#!/usr/bin/env python3
"""Prepare OpenAI Batch API requests for commentary structuring."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Tuple

# Ensure project modules are importable (structuring + shared utilities)
HERE = Path(__file__).resolve().parent
STRUCTURING_DIR = HERE.parent
EXTRACTION_DIR = STRUCTURING_DIR.parent
PROJECT_ROOT = EXTRACTION_DIR.parent

for _path in (HERE, STRUCTURING_DIR, EXTRACTION_DIR, PROJECT_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from structuring_prompts import (  # noqa: E402
    build_structuring_system_prompt,
    build_structuring_user_prompt,
)
from structuring_schema import STRUCTURED_COMMENTARY_SCHEMA  # noqa: E402
from preprocess import preprocess_commentary  # noqa: E402
from model import MODEL_NAME  # noqa: E402
from verses import VersesData  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("temp/batch-commentary-structuring")
DEFAULT_REASONING_EFFORT = "high"
SCHEMA_NAME = "structured_commentary_parts"


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
                "name": SCHEMA_NAME,
                "schema": STRUCTURED_COMMENTARY_SCHEMA,
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


def _load_chapter_payload(path: Path) -> Dict[str, object]:
    """Load the extracted commentary JSON for the chapter."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from {path}: {exc}") from exc


def prepare_batch_requests(
    book_name: str,
    chapter_number: int,
    *,
    input_file: Path | None = None,
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
    source_path = input_file or (Path("books") / book_name / "chapters" / f"{chapter_number}.json")
    chapter_payload = _load_chapter_payload(source_path)
    verses_payload: Dict[str, Dict[str, object]] = chapter_payload.get("verses", {}) if isinstance(chapter_payload, dict) else {}

    system_prompt = build_structuring_system_prompt(book_name, chapter_number)

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
    verses_with_commentary = 0
    debug_dir: Path | None = None
    if debug:
        debug_dir = output_dir / f"{book_name}_{chapter_number}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for verse_num in range(1, total_verses + 1):
            verse_key = str(verse_num)
            verse_entry = verses_payload.get(verse_key) or {}
            raw_commentary = verse_entry.get("commentary") or ""
            has_commentary = bool(verse_entry.get("has_commentary")) and bool(raw_commentary.strip())
            if not has_commentary:
                continue

            verses_with_commentary += 1
            preprocessed = preprocess_commentary(str(raw_commentary))
            user_prompt = build_structuring_user_prompt(book_name, chapter_number, verse_num, preprocessed)
            custom_id = f"{book_name}-{chapter_number}-structuring-verse-{verse_num}"

            request_obj = _build_request_object(
                custom_id=custom_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                use_reasoning=reasoning.use_reasoning,
                reasoning_effort=reasoning.effort,
            )
            fh.write(json.dumps(request_obj, ensure_ascii=False))
            fh.write("\n")

            if debug and debug_dir:
                verse_dir = debug_dir / f"verse_{verse_num:03d}"
                verse_dir.mkdir(parents=True, exist_ok=True)
                (verse_dir / "preprocessed.txt").write_text(preprocessed, encoding="utf-8")
                (verse_dir / "user_prompt.txt").write_text(user_prompt, encoding="utf-8")

    meta = {
        "book": book_name,
        "chapter": chapter_number,
        "total_verses": total_verses,
        "verses_with_commentary": verses_with_commentary,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "system_prompt_chars": len(system_prompt),
        "system_prompt_approx_tokens": approx_token_count(system_prompt),
        "reasoning_effort": reasoning.effort if reasoning.use_reasoning else None,
        "use_reasoning": reasoning.use_reasoning,
        "input_file": str(source_path),
        "output_jsonl": str(jsonl_path),
        "schema_name": SCHEMA_NAME,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if debug and debug_dir:
        (debug_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Prepared %s requests for %s %s (verses with commentary: %s)",
        verses_with_commentary,
        book_name,
        chapter_number,
        verses_with_commentary,
    )
    return jsonl_path, meta_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OpenAI Batch API JSONL requests for structuring.")
    parser.add_argument("book_name", help="Book abbreviation (e.g., Job, Gen, Ps)")
    parser.add_argument("chapter_number", type=int, help="Chapter number to structure (1-indexed)")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Override input JSON path (default: books/{BOOK}/chapters/{CHAPTER}.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSONL output (default: temp/batch-commentary-structuring)",
    )
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning in API calls")
    parser.add_argument("--low-reasoning", action="store_true", help="Use low reasoning effort")
    parser.add_argument("--med-reasoning", action="store_true", help="Use medium reasoning effort")
    parser.add_argument("--high-reasoning", action="store_true", help="Use high reasoning effort (default)")
    parser.add_argument("--xhigh-reasoning", action="store_true", help="Use extra-high reasoning effort")
    parser.add_argument("--debug", action="store_true", help="Save system prompt and preprocessed text")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    prepare_batch_requests(
        book_name=args.book_name,
        chapter_number=args.chapter_number,
        input_file=args.input_file,
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


__all__ = ["prepare_batch_requests", "main"]

