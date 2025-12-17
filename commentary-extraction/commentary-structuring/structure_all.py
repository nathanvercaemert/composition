from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

# Make sure sibling modules and shared utilities are importable
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
CHAPTER_BOUNDARIES = PROJECT_ROOT / "chapter-boundaries"

for path in (HERE, PROJECT_ROOT, CHAPTER_BOUNDARIES):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from structure_chapter import structure_chapter  # noqa: E402


def run_structuring(
    book_name: str,
    chapter_number: int,
    *,
    force: bool = False,
    input_file: Path | None = None,
    output_dir: Path | None = None,
    temp_dir: Path | None = None,
    debug: bool = False,
    dry_run: bool = False,
    reasoning_effort: str = "low",
    track_tokens: bool = False,
) -> Tuple[Path, Path, Optional[dict]]:
    """
    Run the full structuring pipeline for a single chapter.

    Args:
        book_name: Book abbreviation (e.g., Job).
        chapter_number: Chapter number (1-indexed).
        force: Overwrite existing outputs when True.
        input_file: Optional override for the input JSON path.
        output_dir: Optional override for output directory.
        temp_dir: Directory for intermediate artifacts.
        debug: Save prompts and structured parts per verse.
        dry_run: Preprocess only; skip model calls.
        reasoning_effort: GPT-5.2 reasoning level ("low", "medium", "high", "xhigh").
        track_tokens: Aggregate and return token usage when True.

    Returns:
        Tuple of (structured_commentary.json path, structured_commentary.md path, token usage stats or None).
    """
    return structure_chapter(
        book_name=book_name,
        chapter_number=chapter_number,
        input_path=input_file,
        output_dir=output_dir,
        temp_dir=temp_dir,
        force=force,
        debug=debug,
        dry_run=dry_run,
        reasoning_effort=reasoning_effort,
        track_tokens=track_tokens,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structure commentary for a chapter and render JSON/Markdown.")
    parser.add_argument("book_name", help="Book abbreviation (e.g., Job, Gen, Ps)")
    parser.add_argument("chapter_number", type=int, help="Chapter number (1-indexed)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Override input JSON path (default: books/{BOOK}/chapters/{CHAPTER}.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: input file directory with chapter subdirectory)",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Directory for intermediate artifacts (default: temp/commentary-structuring/{BOOK}_{CHAPTER})",
    )
    parser.add_argument("--debug", action="store_true", help="Save intermediate prompts and structured parts")
    parser.add_argument("--dry-run", action="store_true", help="Preprocess only; do not call the model")
    parser.add_argument(
        "--tokens",
        action="store_true",
        help="Display token usage statistics after processing",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["low", "medium", "high", "xhigh"],
        help="Reasoning effort level for GPT-5.2 (default: low)",
    )
    return parser.parse_args()


def _print_token_summary(token_stats: Optional[dict]) -> None:
    if not token_stats:
        print("No token usage data was returned.")
        return

    input_tokens = int(token_stats.get("input_tokens") or 0)
    output_tokens = int(token_stats.get("output_tokens") or 0)
    cached_tokens = int(token_stats.get("cached_tokens") or 0)
    total_tokens = int(token_stats.get("total_tokens") or (input_tokens + output_tokens))
    verses_processed = int(token_stats.get("verses_with_usage") or 0)
    avg_tokens = round(total_tokens / verses_processed) if verses_processed else 0
    cache_pct = (cached_tokens / input_tokens * 100) if input_tokens else 0.0

    print("Token Usage Summary")
    print("-------------------")
    print(f"Input tokens:   {input_tokens:,}")
    print(f"Cached tokens:  {cached_tokens:,}" + (f" ({cache_pct:.1f}%)" if cached_tokens else ""))
    print(f"Output tokens:  {output_tokens:,}")
    print(f"Total tokens:   {total_tokens:,}")
    print()
    print(f"Verses processed: {verses_processed:,}")
    print(f"Avg tokens/verse: {avg_tokens:,}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    try:
        _, _, token_stats = run_structuring(
            book_name=args.book_name,
            chapter_number=args.chapter_number,
            force=args.force,
            input_file=args.input_file,
            output_dir=args.output_dir,
            temp_dir=args.temp_dir,
            debug=args.debug,
            dry_run=args.dry_run,
            reasoning_effort=args.reasoning_effort,
            track_tokens=args.tokens,
        )
        if args.tokens:
            _print_token_summary(token_stats)
    except Exception as exc:  # noqa: BLE001 - runtime guard
        logging.getLogger(__name__).error("Structuring failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ["run_structuring", "main"]

