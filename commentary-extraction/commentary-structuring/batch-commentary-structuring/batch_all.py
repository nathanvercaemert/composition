#!/usr/bin/env python3
"""End-to-end orchestrator for batch commentary structuring."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

from poll_batch import poll_batch  # noqa: E402
from prepare_batch import prepare_batch_requests  # noqa: E402
from process_results import process_results  # noqa: E402
from retrieve_batch import retrieve_batch_outputs  # noqa: E402
from submit_batch import submit_batch  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_TEMP_DIR = Path("temp/batch-commentary-structuring")


def _reasoning_flag_kwargs(reasoning_effort: str, use_reasoning: bool) -> dict:
    """Translate reasoning settings into flag kwargs for prepare_batch_requests."""
    if not use_reasoning:
        return {
            "no_reasoning": True,
            "low_reasoning": False,
            "med_reasoning": False,
            "high_reasoning": False,
            "xhigh_reasoning": False,
        }

    effort = reasoning_effort.lower().strip()
    flags = {
        "no_reasoning": False,
        "low_reasoning": effort == "low",
        "med_reasoning": effort == "medium",
        "high_reasoning": effort == "high",
        "xhigh_reasoning": effort == "xhigh",
    }
    # Default to high when an unexpected value is provided
    if not any(flags.values()):
        flags["high_reasoning"] = True
    return flags


def run_all(
    book_name: str,
    chapter_number: int,
    *,
    force: bool = False,
    tokens: bool = False,
    debug: bool = False,
    poll_interval: int = 60,
    input_file: Path | None = None,
    output_dir: Path | None = None,
    temp_dir: Path = DEFAULT_TEMP_DIR,
    reasoning_effort: str = "high",
    use_reasoning: bool = True,
    cleanup: bool = False,
) -> Tuple[Path, Path]:
    """Run the entire batch structuring workflow. Returns (json_path, md_path)."""
    temp_dir.mkdir(parents=True, exist_ok=True)

    reasoning_kwargs = _reasoning_flag_kwargs(reasoning_effort, use_reasoning=use_reasoning)

    jsonl_path, meta_path = prepare_batch_requests(
        book_name=book_name,
        chapter_number=chapter_number,
        input_file=input_file,
        output_dir=temp_dir,
        debug=debug,
        **reasoning_kwargs,
    )

    batch_info_path = submit_batch(jsonl_path, meta_file=meta_path)
    poll_batch(batch_info_path, watch=True, interval=poll_interval)
    output_path, error_path = retrieve_batch_outputs(batch_info_path, output_dir=temp_dir, include_errors=True)

    json_output_path, md_output_path = process_results(
        output_path,
        meta_file=meta_path,
        input_file=input_file,
        output_dir=output_dir,
        force=force,
        show_tokens=tokens,
        debug=debug,
    )

    if cleanup:
        for path in [jsonl_path, meta_path, batch_info_path, output_path, error_path]:
            if path and Path(path).exists():
                try:
                    Path(path).unlink()
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning("Failed to delete %s: %s", path, exc)

    return json_output_path, md_output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full batch commentary structuring workflow.")
    parser.add_argument("book_name", help="Book abbreviation")
    parser.add_argument("chapter_number", type=int, help="Chapter number (1-indexed)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--tokens", action="store_true", help="Display token usage statistics")
    parser.add_argument("--debug", action="store_true", help="Persist debug artifacts (prompts, preprocessed text)")
    parser.add_argument("--poll-interval", type=int, default=60, help="Polling interval in seconds")
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
        help="Final output directory (default: books/{BOOK}/chapters/{CHAPTER}/)",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=DEFAULT_TEMP_DIR,
        help="Temporary directory for batch files",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high", "xhigh"],
        help="Reasoning effort level for GPT-5.2 (default: high)",
    )
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning in API calls")
    parser.add_argument("--cleanup", action="store_true", help="Delete intermediate batch files after completion")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    try:
        run_all(
            book_name=args.book_name,
            chapter_number=args.chapter_number,
            force=args.force,
            tokens=args.tokens,
            debug=args.debug,
            poll_interval=args.poll_interval,
            input_file=args.input_file,
            output_dir=args.output_dir,
            temp_dir=args.temp_dir,
            reasoning_effort=args.reasoning_effort,
            use_reasoning=not args.no_reasoning,
            cleanup=args.cleanup,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.error("batch_all failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ["run_all", "main"]

