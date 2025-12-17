from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure shared modules are importable when run as a script
HERE = Path(__file__).resolve().parent
COMMENTARY_EXTRACTION = HERE.parent
PROJECT_ROOT = HERE.parent.parent
CHAPTER_BOUNDARIES = PROJECT_ROOT / "chapter-boundaries"

for path in (HERE, COMMENTARY_EXTRACTION, PROJECT_ROOT, CHAPTER_BOUNDARIES):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from verses import VersesData  # type: ignore  # noqa: E402

from markdown_renderer import render_chapter_markdown  # noqa: E402
from preprocess import preprocess_commentary  # noqa: E402
from structuring_prompts import build_structuring_system_prompt, build_structuring_user_prompt  # noqa: E402
from structure_verse import structure_verse_commentary  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_TEMP_ROOT = Path("temp/commentary-structuring")


def _load_input(input_path: Path) -> Dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def _ensure_output_paths(json_path: Path, md_path: Path, force: bool) -> None:
    if force:
        return
    collisions = [p for p in (json_path, md_path) if p.exists()]
    if collisions:
        names = ", ".join(str(p) for p in collisions)
        raise FileExistsError(f"Output already exists: {names}. Use --force to overwrite.")


def structure_chapter(
    book_name: str,
    chapter_number: int,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    temp_dir: Path | None = None,
    *,
    force: bool = False,
    debug: bool = False,
    dry_run: bool = False,
    reasoning_effort: str = "low",
    track_tokens: bool = False,
) -> Tuple[Path, Path, Optional[Dict[str, Any]]]:
    """
    Structure all verse commentaries in a chapter and write JSON + Markdown outputs.

    Args:
        book_name: Book abbreviation (e.g., Job, Gen).
        chapter_number: Chapter number (1-indexed).
        input_path: Optional override for the source JSON.
        output_dir: Optional override for the output directory.
        temp_dir: Where to place debug artifacts when enabled.
        force: Overwrite existing outputs when True.
        debug: Persist prompts and structured parts for inspection.
        dry_run: Skip model calls; still preprocess and write debug files.
        reasoning_effort: GPT-5.2 reasoning level ("low", "medium", "high", "xhigh").
        track_tokens: Aggregate token usage across verse calls when True.

    Returns:
        Tuple of (structured_commentary.json path, structured_commentary.md path, token usage stats).
        Token stats is a dict with aggregated counts when track_tokens is True; otherwise None.
    """
    verses_data = VersesData()
    if not verses_data.has_book(book_name):
        raise ValueError(f"Unknown book abbreviation: {book_name}")

    total_verses = verses_data.get_verse_count(book_name, chapter_number)

    source_path = input_path or (Path("books") / book_name / "chapters" / f"{chapter_number}.json")
    chapter_payload = _load_input(source_path)
    verses_payload: Dict[str, Dict[str, object]] = chapter_payload.get("verses", {}) if isinstance(chapter_payload, dict) else {}

    out_dir = output_dir or (source_path.parent / str(chapter_number))
    out_dir.mkdir(parents=True, exist_ok=True)
    json_output_path = out_dir / "structured_commentary.json"
    md_output_path = out_dir / "structured_commentary.md"
    _ensure_output_paths(json_output_path, md_output_path, force=force)

    working_dir = temp_dir or (DEFAULT_TEMP_ROOT / f"{book_name}_{chapter_number}")
    if debug or dry_run:
        working_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = build_structuring_system_prompt(book_name, chapter_number)
    if debug:
        (working_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")

    results: Dict[str, Dict[str, object]] = {}
    error_count = 0
    verses_with_commentary = 0
    token_totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "total_tokens": 0} if track_tokens else None
    verses_with_usage = 0

    for verse_num in range(1, total_verses + 1):
        verse_key = str(verse_num)
        verse_entry = verses_payload.get(verse_key) or {}
        has_commentary = bool(verse_entry.get("has_commentary"))
        raw_commentary = verse_entry.get("commentary") or ""
        preprocessed = preprocess_commentary(raw_commentary) if raw_commentary else ""
        user_prompt = build_structuring_user_prompt(book_name, chapter_number, verse_num, preprocessed)

        verse_dir: Optional[Path] = None
        if debug or dry_run:
            verse_dir = working_dir / f"verse_{verse_num:03d}"
            verse_dir.mkdir(parents=True, exist_ok=True)
            (verse_dir / "preprocessed.txt").write_text(preprocessed, encoding="utf-8")
            (verse_dir / "user_prompt.txt").write_text(user_prompt, encoding="utf-8")

        if not has_commentary or not raw_commentary.strip():
            results[verse_key] = {
                "has_commentary": False,
                "structured_parts": [],
            }
            if verse_dir:
                (verse_dir / "status.txt").write_text("Skipped: no commentary", encoding="utf-8")
            continue

        verses_with_commentary += 1

        try:
            parts, usage = structure_verse_commentary(
                book_name=book_name,
                chapter_number=chapter_number,
                verse_number=verse_num,
                commentary_text=raw_commentary,
                system_prompt=system_prompt,
                preprocessed_commentary=preprocessed,
                user_prompt=user_prompt,
                dry_run=dry_run,
                reasoning_effort=reasoning_effort,
            )
            verse_result: Dict[str, object] = {
                "has_commentary": True,
                "structured_parts": parts,
            }
            if debug or dry_run:
                verse_result["preprocessed_commentary"] = preprocessed
            if track_tokens and token_totals is not None and usage:
                verses_with_usage += 1
                # Extract tokens matching the pattern from commentary_extraction.py
                input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
                cached_tokens = input_details.get("cached_tokens") or 0
                token_totals["input_tokens"] += int(input_tokens)
                token_totals["output_tokens"] += int(output_tokens)
                token_totals["cached_tokens"] += int(cached_tokens)
        except Exception as exc:  # noqa: BLE001 - runtime safety
            error_count += 1
            verse_result = {
                "has_commentary": True,
                "structured_parts": [],
                "error": str(exc),
            }
            if verse_dir:
                (verse_dir / "error.txt").write_text(str(exc), encoding="utf-8")
            logger.error("Failed to structure %s %s:%s: %s", book_name, chapter_number, verse_num, exc)
        else:
            if verse_dir:
                (verse_dir / "structured_parts.json").write_text(
                    json.dumps(verse_result["structured_parts"], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        results[verse_key] = verse_result

    token_stats: Optional[Dict[str, Any]] = None
    metadata = {
        "source_file": str(source_path),
        "structuring_timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model_used": "gpt-5.2",
        "verses_processed": total_verses,
        "verses_with_commentary": verses_with_commentary,
        "verses_with_errors": error_count,
    }
    if track_tokens and token_totals is not None:
        # Derive total_tokens if the API omits it; prefer provided total when present.
        if token_totals["total_tokens"] == 0:
            token_totals["total_tokens"] = token_totals["input_tokens"] + token_totals["output_tokens"]
        token_stats = {
            **token_totals,
            "verses_with_usage": verses_with_usage,
        }
        if token_stats["input_tokens"] > 0 and token_stats["cached_tokens"] > 0:
            token_stats["cache_hit_rate"] = token_stats["cached_tokens"] / token_stats["input_tokens"]
        metadata["token_usage"] = token_stats

    output_payload = {
        "book": book_name,
        "chapter": chapter_number,
        "verses": results,
        "metadata": metadata,
    }

    json_output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_text = render_chapter_markdown(book_name, chapter_number, results)
    md_output_path.write_text(markdown_text, encoding="utf-8")

    logger.info("Wrote structured commentary JSON to %s", json_output_path)
    logger.info("Wrote structured commentary Markdown to %s", md_output_path)

    return json_output_path, md_output_path, token_stats


__all__ = ["structure_chapter"]

