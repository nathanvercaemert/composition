#!/usr/bin/env python3
"""
Coordinate chapter boundary refinement using per-chapter multi-label tagging.

Reads unrefined chapter boundaries, pads the candidate pages, calls the model
once per chapter, validates the tagging response, and writes the refined
chapter boundaries to temp/BOOK/chapter_boundaries_refined.json by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_client import ModelCallError, ModelClient, ModelClientError, ResponseParseError
from orchestrator import get_page_stems, natural_key
from refiner_instructions import (
    build_refinement_system_prompt,
    build_refinement_user_prompt,
    format_refinement_batch_content,
)
from binary_aggregator import (
    extract_chapter_pages,
    merge_chapter_memberships,
    validate_tagging_response,
)

logger = logging.getLogger(__name__)


@dataclass
class RefinementOptions:
    """Configuration options for the refinement run."""

    padding: int = 3
    force: bool = False
    debug: bool = False
    delay_between_chapters: float = 1.0


@dataclass
class ChapterRefinementResult:
    """Captures the outcome of a single chapter refinement call."""

    chapter: str
    original_pages: List[str]
    padded_pages: List[str]
    refined_pages: List[str]
    page_tags: Dict[str, List[str]] = field(default_factory=dict)
    duration_seconds: float = 0.0
    missing_pages: List[str] = field(default_factory=list)
    unexpected_pages: List[str] = field(default_factory=list)
    error: str | None = None
    raw_response: Any | None = None


def load_unrefined_boundaries(path: Path) -> Dict[str, List[str]]:
    """Load initial chapter boundaries from JSON."""

    if not path.exists():
        logger.error("Unrefined boundaries not found at %s", path)
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read %s: %s", path, exc)
        return {}

    if not isinstance(data, dict):
        logger.error("Unrefined boundaries must be a JSON object mapping chapter->pages.")
        return {}

    result: Dict[str, List[str]] = {}
    for chapter, pages in data.items():
        if isinstance(pages, list):
            result[str(chapter)] = [str(p) for p in pages]
        else:
            logger.warning("Chapter %s has invalid pages list; skipping entries.", chapter)
            result[str(chapter)] = []
    return result


def get_all_page_stems(book_path: Path) -> List[str]:
    """Return all page stems for a book, ordered naturally."""

    stems = get_page_stems(book_path)
    if not stems:
        logger.error("No page stems found under %s/raw-images", book_path)
    return stems


def compute_padded_range(chapter_pages: List[str], all_pages: List[str], padding: int) -> List[str]:
    """Return the padded set of page stems for a chapter, respecting book bounds."""

    if not chapter_pages or not all_pages:
        return []

    index_map = {stem: idx for idx, stem in enumerate(all_pages)}
    indices = [index_map[p] for p in chapter_pages if p in index_map]
    if not indices:
        logger.warning("Chapter pages %s not found in page list; skipping padding.", chapter_pages)
        return []

    start_idx = max(0, min(indices) - max(0, padding))
    end_idx = min(len(all_pages) - 1, max(indices) + max(0, padding))

    if start_idx == 0 and min(indices) - padding < 0:
        logger.debug("Padding truncated at start of book.")
    if end_idx == len(all_pages) - 1 and max(indices) + padding >= len(all_pages):
        logger.debug("Padding truncated at end of book.")

    return all_pages[start_idx : end_idx + 1]


def refine_chapter(
    client: ModelClient,
    book_path: Path,
    chapter: str,
    padded_stems: Sequence[str],
    book_name: str,
    debug_dir: Path | None,
    adjacent_chapters: Sequence[str] | None = None,
) -> ChapterRefinementResult:
    """Run a multi-label tagging refinement call for a single chapter's region."""

    start = time.time()
    user_prompt = build_refinement_user_prompt(
        chapter,
        padded_stems,
        book_name=book_name,
        adjacent_chapters=adjacent_chapters,
    )
    system_prompt = build_refinement_system_prompt(chapter, padded_stems)
    batch_content = format_refinement_batch_content(book_path, padded_stems)
    user_content = f"{user_prompt}\n\n{batch_content.text}"

    raw_response: Any = None
    validated: Dict[str, List[str]] = {}
    missing: List[str] = []
    unexpected: List[str] = []
    error: str | None = None

    if not padded_stems:
        logger.warning("Chapter %s has no pages to refine; skipping model call.", chapter)
    else:
        try:
            raw_response = client.complete_json(
                system_prompt=system_prompt,
                user_content=user_content,
            )
            validated, missing, unexpected = validate_tagging_response(raw_response, padded_stems)
        except (ModelCallError, ResponseParseError, ModelClientError) as exc:
            error = str(exc)
            logger.warning("Model call failed for chapter %s: %s", chapter, exc)
        except Exception as exc:  # noqa: BLE001
            error = f"Unexpected error: {exc}"
            logger.error("Unexpected failure refining chapter %s: %s", chapter, exc)

    final_page_tags: Dict[str, List[str]] = {}
    for stem in padded_stems:
        final_page_tags[stem] = validated.get(stem, [])
    refined_pages = extract_chapter_pages(final_page_tags, chapter)

    if missing:
        logger.warning(
            "Chapter %s missing %s page(s) in response; they will have no tags: %s",
            chapter,
            len(missing),
            missing,
        )
    if unexpected:
        logger.warning("Chapter %s returned unexpected page keys: %s", chapter, unexpected)
    if not refined_pages:
        logger.warning("Chapter %s produced no pages after refinement.", chapter)

    duration = time.time() - start

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            (debug_dir / "input_content.txt").write_text(user_content, encoding="utf-8")
            (debug_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
            if raw_response is not None:
                (debug_dir / "model_response.json").write_text(
                    json.dumps(raw_response, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            meta = {
                "chapter": chapter,
                "book": book_name,
                "padded_pages": list(padded_stems),
                "refined_pages": refined_pages,
                "all_page_tags": final_page_tags,
                "missing_pages": missing,
                "unexpected_pages": unexpected,
                "duration_seconds": duration,
                "error": error,
                "truncation_map": batch_content.truncation_map,
                "estimated_tokens": batch_content.estimated_tokens,
            }
            (debug_dir / "refinement_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write debug artifacts for chapter %s: %s", chapter, exc)

    return ChapterRefinementResult(
        chapter=chapter,
        original_pages=list(padded_stems),
        padded_pages=list(padded_stems),
        refined_pages=refined_pages,
        page_tags=final_page_tags,
        duration_seconds=duration,
        missing_pages=missing,
        unexpected_pages=unexpected,
        error=error,
        raw_response=raw_response,
    )


def assemble_refined_boundaries(results: List[ChapterRefinementResult]) -> Dict[str, List[str]]:
    """Assemble final chapter boundaries from chapter-level results."""

    chapter_maps = {res.chapter: res.page_tags for res in results}
    merged = merge_chapter_memberships(chapter_maps)

    for res in results:
        if not merged.get(res.chapter):
            logger.warning("Chapter %s has zero pages in the final mapping.", res.chapter)

    return {chapter: merged.get(chapter, []) for chapter in sorted(merged.keys(), key=natural_key)}


def process_book_refinement(
    book_name: str,
    options: RefinementOptions,
    unrefined_path: Path,
    output_path: Path,
) -> Dict[str, List[str]] | None:
    """Refine chapter boundaries for a single book and write the result."""

    book_path = Path("books") / book_name
    if not book_path.exists():
        logger.error("Book directory %s does not exist", book_path)
        return None

    unrefined = load_unrefined_boundaries(unrefined_path)
    if not unrefined:
        logger.error("No unrefined boundaries loaded; aborting.")
        return None

    if output_path.exists() and not options.force:
        answer = input(f"Output {output_path} exists. Overwrite? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            logger.info("Aborting; output file left untouched.")
            return None

    all_pages = get_all_page_stems(book_path)
    if not all_pages:
        return None

    client = ModelClient()
    debug_base = Path("temp") / book_name / "refining" if options.debug else None
    if debug_base:
        debug_base.mkdir(parents=True, exist_ok=True)

    results: List[ChapterRefinementResult] = []
    sorted_chapters = sorted(unrefined.items(), key=lambda kv: natural_key(kv[0]))
    all_chapter_keys = [ch for ch, _ in sorted_chapters]

    for idx, (chapter, pages) in enumerate(sorted_chapters):
        padded = compute_padded_range(pages, all_pages, options.padding)
        chapter_debug_dir = None
        if debug_base:
            safe_chapter = "".join(c if c.isalnum() else "_" for c in str(chapter))
            chapter_label = safe_chapter or "chapter"
            chapter_debug_dir = debug_base / f"chapter_{chapter_label}"

        adjacent_chapters: List[str] = []
        if idx > 0:
            adjacent_chapters.append(all_chapter_keys[idx - 1])
        if idx < len(all_chapter_keys) - 1:
            adjacent_chapters.append(all_chapter_keys[idx + 1])

        logger.info("Refining chapter %s (%s pages, padded to %s)", chapter, len(pages), len(padded))
        result = refine_chapter(
            client,
            book_path,
            chapter,
            padded,
            book_name,
            chapter_debug_dir,
            adjacent_chapters=adjacent_chapters,
        )
        result.original_pages = list(pages)
        result.padded_pages = list(padded)
        results.append(result)

        if options.delay_between_chapters > 0 and idx < len(sorted_chapters) - 1:
            time.sleep(options.delay_between_chapters)

    refined = assemble_refined_boundaries(results)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(refined, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote refined boundaries to %s", output_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write refined boundaries: %s", exc)

    return refined


def main() -> None:
    """CLI entry point for chapter boundary refinement."""

    parser = argparse.ArgumentParser(description="Refine chapter boundaries using multi-label chapter tagging.")
    parser.add_argument("book_name", help="Name of the book directory under books/")
    parser.add_argument(
        "--unrefined-path",
        type=Path,
        help="Path to unrefined boundaries JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Path to write refined boundaries JSON (default: temp/BOOK_NAME/chapter_boundaries_refined.json).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=3,
        help="Pages of padding to include on each side (default: 3).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output without prompting.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate refinement data to temp/BOOK/refining/.",
    )
    parser.add_argument(
        "--delay-between-chapters",
        type=float,
        default=1.0,
        help="Seconds to pause between chapter refinements (default: 1.0).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    book_name = args.book_name
    unrefined_path = args.unrefined_path or Path("temp") / book_name / "chapter_boundaries_unrefined.json"
    output_path = args.output_path or Path("temp") / book_name / "chapter_boundaries_refined.json"

    options = RefinementOptions(
        padding=args.padding,
        force=args.force,
        debug=args.debug,
        delay_between_chapters=args.delay_between_chapters,
    )

    process_book_refinement(book_name, options, unrefined_path, output_path)


if __name__ == "__main__":
    main()

