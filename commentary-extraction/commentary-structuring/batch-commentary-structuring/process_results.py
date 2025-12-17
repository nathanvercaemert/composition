#!/usr/bin/env python3
"""Process batch structuring output JSONL into final JSON + Markdown."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Ensure project modules are importable (structuring + shared utilities)
HERE = Path(__file__).resolve().parent
STRUCTURING_DIR = HERE.parent
EXTRACTION_DIR = STRUCTURING_DIR.parent
PROJECT_ROOT = EXTRACTION_DIR.parent

for _path in (HERE, STRUCTURING_DIR, EXTRACTION_DIR, PROJECT_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from markdown_renderer import render_chapter_markdown  # noqa: E402
from model import MODEL_NAME, _extract_output_text  # noqa: E402
from preprocess import preprocess_commentary  # noqa: E402
from verses import VersesData  # noqa: E402

logger = logging.getLogger(__name__)

CUSTOM_ID_PATTERN = re.compile(r"^(?P<book>[^-]+)-(?P<chapter>\d+)-structuring-verse-(?P<verse>\d+)$")


def _parse_custom_id(custom_id: str) -> Tuple[str, int, int] | None:
    """Extract (book, chapter, verse) from custom_id."""
    match = CUSTOM_ID_PATTERN.match(custom_id.strip())
    if not match:
        return None
    return match.group("book"), int(match.group("chapter")), int(match.group("verse"))


def _coerce_int(value: object) -> int | None:
    """Attempt to coerce a value to int, returning None on failure."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _load_jsonl_lines(path: Path) -> Iterable[Dict[str, object]]:
    """Yield parsed JSON objects for each line in a JSONL file."""
    with path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error("Invalid JSON at line %s in %s: %s", line_num, path, exc)


def _load_meta(meta_path: Path | None) -> Dict[str, object]:
    """Load metadata file if present."""
    if not meta_path or not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to read meta file %s: %s", meta_path, exc)
        return {}


def _auto_meta_path(output_jsonl: Path) -> Path | None:
    """Find a related meta or batch file beside the output JSONL."""
    base = output_jsonl.stem.replace("_output", "")
    meta_candidate = output_jsonl.with_name(f"{base}_meta.json")
    if meta_candidate.exists():
        return meta_candidate
    batch_candidate = output_jsonl.with_name(f"{base}_batch.json")
    if batch_candidate.exists():
        return batch_candidate
    return None


def _parse_usage(usage: Dict[str, object], totals: Dict[str, int]) -> None:
    """Accumulate token usage into totals dict."""
    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    cached_tokens = input_details.get("cached_tokens") or 0
    reasoning_tokens = output_details.get("reasoning_tokens") or 0

    totals["input_tokens"] += int(input_tokens or 0)
    totals["output_tokens"] += int(output_tokens or 0)
    totals["cached_tokens"] += int(cached_tokens or 0)
    totals["reasoning_tokens"] += int(reasoning_tokens or 0)


def _extract_structured_parts(body: Dict[str, object]) -> list[object]:
    """Extract structured parts list from a Responses API body."""
    text_output = _extract_output_text(body)
    parsed = json.loads(text_output)
    if isinstance(parsed, dict) and "parts" in parsed:
        parts = parsed.get("parts")
    else:
        parts = parsed

    if not isinstance(parts, list):
        raise ValueError(f"Expected parts list, got {type(parts).__name__}")
    return parts


def _ensure_output_paths(json_path: Path, md_path: Path, force: bool) -> None:
    """Guard against overwriting unless force is enabled."""
    if force:
        return
    collisions = [p for p in (json_path, md_path) if p.exists()]
    if collisions:
        names = ", ".join(str(p) for p in collisions)
        raise FileExistsError(f"Output already exists: {names}. Use --force to overwrite.")


def process_results(
    output_jsonl: Path,
    *,
    meta_file: Path | None = None,
    input_file: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
    show_tokens: bool = False,
    debug: bool = False,
) -> Tuple[Path, Path]:
    """Convert batch JSONL responses into final chapter JSON + Markdown."""
    if not output_jsonl.exists():
        raise FileNotFoundError(f"Output JSONL not found: {output_jsonl}")

    meta_path = meta_file or _auto_meta_path(output_jsonl)
    meta = _load_meta(meta_path)
    batch_id = meta.get("batch_id")

    verses_data = VersesData()

    results: Dict[str, Dict[str, object]] = {}
    errors: Dict[str, str] = {}
    book_name: str | None = meta.get("book")
    chapter_number: int | None = _coerce_int(meta.get("chapter"))

    totals = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    for entry in _load_jsonl_lines(output_jsonl):
        custom_id = entry.get("custom_id")
        parsed_id = _parse_custom_id(custom_id) if isinstance(custom_id, str) else None
        if not parsed_id:
            logger.error("Skipping entry without valid custom_id: %s", custom_id)
            continue
        book_from_id, chapter_from_id, verse_from_id = parsed_id
        book_name = book_name or book_from_id
        chapter_number = chapter_number or chapter_from_id
        verse_key = str(verse_from_id)

        error_obj = entry.get("error")
        response_obj = entry.get("response") or {}
        status_code = response_obj.get("status_code")
        body = response_obj.get("body") or {}

        if error_obj:
            errors[verse_key] = str(error_obj)
            continue
        if status_code and status_code != 200:
            errors[verse_key] = f"Non-200 status: {status_code}"
            continue

        try:
            structured_parts = _extract_structured_parts(body)
            results[verse_key] = {
                "has_commentary": True,
                "structured_parts": structured_parts,
            }
        except Exception as exc:  # noqa: BLE001 - runtime safety
            errors[verse_key] = f"Failed to parse response: {exc}"
            continue

        if show_tokens:
            usage = body.get("usage") or {}
            if isinstance(usage, dict):
                _parse_usage(usage, totals)

    if not book_name or chapter_number is None:
        raise RuntimeError("Unable to determine book/chapter from custom_id or metadata.")

    source_path = input_file or (Path(meta.get("input_file")) if meta.get("input_file") else None)
    if not source_path:
        source_path = Path("books") / book_name / "chapters" / f"{chapter_number}.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Original input file not found: {source_path}")

    try:
        input_payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse input JSON at {source_path}: {exc}") from exc

    verses_payload: Dict[str, Dict[str, object]] = input_payload.get("verses", {}) if isinstance(input_payload, dict) else {}
    total_verses = (
        _coerce_int(meta.get("total_verses"))
        or verses_data.get_verse_count(book_name, chapter_number)
    )

    has_commentary_flags: Dict[str, bool] = {}
    preprocessed_by_verse: Dict[str, str] = {}
    for verse_num in range(1, total_verses + 1):
        verse_key = str(verse_num)
        verse_entry = verses_payload.get(verse_key) or {}
        raw_commentary = verse_entry.get("commentary") or ""
        has_commentary = bool(verse_entry.get("has_commentary")) and bool(str(raw_commentary).strip())
        has_commentary_flags[verse_key] = has_commentary
        if has_commentary or debug:
            preprocessed_by_verse[verse_key] = preprocess_commentary(str(raw_commentary))

    final_verses: Dict[str, Dict[str, object]] = {}
    for verse_num in range(1, total_verses + 1):
        verse_key = str(verse_num)
        if verse_key in results:
            verse_result = results[verse_key]
        elif verse_key in errors:
            verse_result = {
                "has_commentary": has_commentary_flags.get(verse_key, True),
                "structured_parts": [],
                "error": errors[verse_key],
            }
        elif has_commentary_flags.get(verse_key):
            verse_result = {
                "has_commentary": True,
                "structured_parts": [],
                "error": "No batch response",
            }
        else:
            verse_result = {"has_commentary": False, "structured_parts": []}

        if debug:
            verse_result["preprocessed_commentary"] = preprocessed_by_verse.get(verse_key, "")
        final_verses[verse_key] = verse_result

    verses_with_commentary = sum(1 for v in final_verses.values() if v.get("has_commentary"))
    verses_with_errors = sum(1 for v in final_verses.values() if v.get("error"))

    metadata = {
        "source_file": str(source_path),
        "structuring_timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model_used": MODEL_NAME,
        "processing_method": "batch",
        "batch_id": batch_id,
        "verses_processed": int(total_verses),
        "verses_with_commentary": verses_with_commentary,
        "verses_with_errors": verses_with_errors,
    }
    if show_tokens:
        metadata["token_usage"] = totals

    target_dir = output_dir or (Path("books") / book_name / "chapters" / str(chapter_number))
    target_dir.mkdir(parents=True, exist_ok=True)
    json_output_path = target_dir / "structured_commentary.json"
    md_output_path = target_dir / "structured_commentary.md"

    _ensure_output_paths(json_output_path, md_output_path, force=force)

    output_payload = {
        "book": book_name,
        "chapter": chapter_number,
        "verses": final_verses,
        "metadata": metadata,
    }

    json_output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_text = render_chapter_markdown(book_name, chapter_number, final_verses)
    md_output_path.write_text(markdown_text, encoding="utf-8")

    logger.info("Wrote structured commentary JSON to %s", json_output_path)
    logger.info("Wrote structured commentary Markdown to %s", md_output_path)

    if show_tokens:
        total_input = totals["input_tokens"]
        total_output = totals["output_tokens"]
        print("\n" + "=" * 47)
        print("TOKEN USAGE SUMMARY")
        print("=" * 47)
        print(f"Total Input Tokens:     {total_input:,}")
        print(f"  └─ Cached Tokens:     {totals['cached_tokens']:,}")
        print(f"Total Output Tokens:    {total_output:,}")
        print(f"  └─ Reasoning Tokens:  {totals['reasoning_tokens']:,}")
        print(f"Grand Total:            {total_input + total_output:,}")
        if total_input:
            cache_rate = (totals["cached_tokens"] / total_input) * 100
            print(f"Cache Hit Rate:         {cache_rate:.1f}%")
        print("=" * 47)

    return json_output_path, md_output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process batch output JSONL into structured commentary JSON + Markdown.")
    parser.add_argument("output_jsonl", type=Path, help="Path to the batch output JSONL file")
    parser.add_argument("--meta-file", type=Path, default=None, help="Path to the metadata JSON file")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Path to the original commentary JSON (default: derived from book/chapter)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for final outputs (default: books/{BOOK}/chapters/{CHAPTER}/)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--tokens", action="store_true", help="Display token usage statistics from responses")
    parser.add_argument("--debug", action="store_true", help="Include preprocessed commentary in output JSON")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    try:
        process_results(
            args.output_jsonl,
            meta_file=args.meta_file,
            input_file=args.input_file,
            output_dir=args.output_dir,
            force=args.force,
            show_tokens=args.tokens,
            debug=args.debug,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.error("Failed to process results: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ["process_results", "main"]

