#!/usr/bin/env python3
"""
Finalize chapter boundaries by padding each chapter with adjacent pages.

Reads refined boundaries from temp/BOOK/chapter_boundaries_refined.json (or a
provided path), appends the next sequential page to every chapter except the
last, and prepends the previous sequential page to every chapter except the
first (never before the book's first page). Writes the final mapping to
books/BOOK/chapter_boundaries.json (or a provided output path). This is the
final step in the chapter boundary pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import get_page_stems, natural_key

logger = logging.getLogger(__name__)


def load_refined_boundaries(path: Path) -> Dict[str, List[str]]:
    """Load refined chapter boundaries from JSON."""

    if not path.exists():
        logger.error("Refined boundaries not found at %s", path)
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read refined boundaries from %s: %s", path, exc)
        return {}

    if not isinstance(data, dict):
        logger.error("Refined boundaries must be a JSON object mapping chapter->pages.")
        return {}

    normalized: Dict[str, List[str]] = {}
    for chapter, pages in data.items():
        if isinstance(pages, list):
            normalized[str(chapter)] = [str(p) for p in pages]
        else:
            logger.warning("Chapter %s has invalid pages list; using empty list.", chapter)
            normalized[str(chapter)] = []
    return normalized


def apply_end_padding(boundaries: Dict[str, List[str]], page_stems: List[str]) -> Dict[str, List[str]]:
    """Pad chapters with previous/next pages where applicable."""

    if not boundaries or not page_stems:
        return {}

    ordered_chapters = sorted(boundaries.keys(), key=natural_key)
    page_indices = {stem: idx for idx, stem in enumerate(page_stems)}
    finalized: Dict[str, List[str]] = {}

    for idx, chapter in enumerate(ordered_chapters):
        pages = [str(p) for p in boundaries.get(chapter, [])]
        is_first_chapter = idx == 0
        is_last_chapter = idx == len(ordered_chapters) - 1

        if pages:
            # Prepend previous page for non-first chapters when available.
            if not is_first_chapter:
                first_page = pages[0]
                first_idx = page_indices.get(first_page)
                if first_idx is None:
                    logger.warning(
                        "First page %s for chapter %s not found in raw page list; skipping leading padding.",
                        first_page,
                        chapter,
                    )
                elif first_idx > 0:
                    prev_page = page_stems[first_idx - 1]
                    if prev_page not in pages:
                        pages.insert(0, prev_page)
                        logger.info("Prepended padding page %s to chapter %s", prev_page, chapter)

        if not is_last_chapter and pages:
            last_page = pages[-1]
            last_idx = page_indices.get(last_page)
            if last_idx is None:
                logger.warning(
                    "Last page %s for chapter %s not found in raw page list; skipping padding.",
                    last_page,
                    chapter,
                )
            elif last_idx + 1 < len(page_stems):
                next_page = page_stems[last_idx + 1]
                if next_page not in pages:
                    pages.append(next_page)
                    logger.info("Appended padding page %s to chapter %s", next_page, chapter)

        finalized[chapter] = pages

    return finalized


def write_final_boundaries(finalized: Dict[str, List[str]], output_path: Path, *, force: bool) -> bool:
    """Persist finalized boundaries to disk, prompting before overwrite unless forced."""

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to create output directory %s: %s", output_path.parent, exc)
        return False

    if output_path.exists() and not force:
        try:
            reply = input(f"{output_path} exists. Overwrite? [y/N]: ").strip().lower()
        except EOFError:
            reply = "n"
        if reply not in {"y", "yes"}:
            logger.info("Aborting; existing output preserved.")
            return False

    try:
        output_path.write_text(json.dumps(finalized, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write finalized boundaries to %s: %s", output_path, exc)
        return False

    logger.info("Wrote finalized chapter boundaries to %s", output_path)
    return True


def finalize_book(book_name: str, refined_path: Path, output_path: Path, *, force: bool) -> Dict[str, List[str]] | None:
    """Finalize chapter boundaries for a single book."""

    book_path = Path("books") / book_name
    if not book_path.exists():
        logger.error("Book directory %s does not exist", book_path)
        return None

    refined = load_refined_boundaries(refined_path)
    if not refined:
        logger.error("No refined boundaries loaded; aborting.")
        return None

    page_stems = get_page_stems(book_path)
    if not page_stems:
        logger.error("No page stems found under %s/raw-images", book_path)
        return None

    finalized = apply_end_padding(refined, page_stems)
    if not finalized:
        logger.warning("No chapters to finalize for %s", book_name)

    if not write_final_boundaries(finalized, output_path, force=force):
        return None

    return finalized


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Finalize chapter boundaries by padding end pages.")
    parser.add_argument("book_name", help="Name of the book directory under books/")
    parser.add_argument(
        "--refined-path",
        type=Path,
        help="Path to refined boundaries JSON (default: temp/BOOK/chapter_boundaries_refined.json).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Path to write finalized boundaries (default: books/BOOK/chapter_boundaries.json).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output without prompting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    book_name = args.book_name
    refined_path = args.refined_path or Path("temp") / book_name / "chapter_boundaries_refined.json"
    output_path = args.output_path or Path("books") / book_name / "chapter_boundaries.json"

    result = finalize_book(book_name, refined_path, output_path, force=args.force)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()

