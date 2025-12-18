#!/usr/bin/env python3
"""End-to-end orchestrator for batch commentary extraction."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from prepare_batch import prepare_batch_requests  # noqa: E402
from process_results import process_results  # noqa: E402
from poll_batch import poll_batch  # noqa: E402
from retrieve_batch import retrieve_batch_outputs  # noqa: E402
from submit_batch import submit_batch  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_TEMP_DIR = Path("temp/batch-commentary-extraction")


def _collect_token_usage(output_jsonl: Path) -> Dict[str, int]:
    """Parse a batch output JSONL file and aggregate token usage."""
    totals = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    if not output_jsonl.exists():
        logger.warning("Cannot collect token usage; output file missing: %s", output_jsonl)
        return totals

    try:
        with output_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                body = (entry.get("response") or {}).get("body") or {}
                usage = body.get("usage") or {}
                if not isinstance(usage, dict):
                    continue

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
    except Exception as exc:  # pragma: no cover - best effort aggregation
        logger.warning("Failed to collect token usage from %s: %s", output_jsonl, exc)

    return totals


def run_all(
    book_name: str,
    chapter_number: int,
    *,
    force: bool = False,
    tokens: bool = False,
    poll_interval: int = 60,
    output_dir: Path | None = None,
    temp_dir: Path = DEFAULT_TEMP_DIR,
    no_reasoning: bool = False,
    low_reasoning: bool = False,
    med_reasoning: bool = False,
    high_reasoning: bool = False,
    xhigh_reasoning: bool = False,
    cleanup: bool = False,
    return_usage: bool = False,
) -> Path | Tuple[Path, Dict[str, int]]:
    """Run the entire batch workflow and return the final output path (and tokens when requested)."""
    temp_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path, meta_path = prepare_batch_requests(
        book_name=book_name,
        chapter_number=chapter_number,
        output_dir=temp_dir,
        no_reasoning=no_reasoning,
        low_reasoning=low_reasoning,
        med_reasoning=med_reasoning,
        high_reasoning=high_reasoning,
        xhigh_reasoning=xhigh_reasoning,
    )

    batch_info_path = submit_batch(jsonl_path, meta_file=meta_path)
    poll_batch(batch_info_path, watch=True, interval=poll_interval)
    output_path, error_path = retrieve_batch_outputs(batch_info_path, output_dir=temp_dir, include_errors=True)

    token_usage: Dict[str, int] | None = None
    if return_usage:
        token_usage = _collect_token_usage(output_path)

    final_output = process_results(
        output_path,
        meta_file=meta_path,
        output_dir=output_dir,
        force=force,
        show_tokens=tokens,
    )

    if cleanup:
        for path in [jsonl_path, meta_path, batch_info_path, output_path, error_path]:
            if path and Path(path).exists():
                try:
                    Path(path).unlink()
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning("Failed to delete %s: %s", path, exc)

    if return_usage and token_usage is not None:
        return final_output, token_usage
    return final_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full batch commentary extraction workflow.")
    parser.add_argument("book_name", help="Book abbreviation")
    parser.add_argument("chapter_number", type=int, help="Chapter number (1-indexed)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    parser.add_argument("--tokens", action="store_true", help="Display token usage statistics")
    parser.add_argument("--poll-interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Final output directory (default: books/{BOOK}/chapters/)",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=DEFAULT_TEMP_DIR,
        help="Temporary directory for batch files",
    )
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning")
    parser.add_argument("--low-reasoning", action="store_true", help="Use low reasoning effort")
    parser.add_argument("--med-reasoning", action="store_true", help="Use medium reasoning effort")
    parser.add_argument("--high-reasoning", action="store_true", help="Use high reasoning effort (default)")
    parser.add_argument("--xhigh-reasoning", action="store_true", help="Use extra-high reasoning effort")
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
            poll_interval=args.poll_interval,
            output_dir=args.output_dir,
            temp_dir=args.temp_dir,
            no_reasoning=args.no_reasoning,
            low_reasoning=args.low_reasoning,
            med_reasoning=args.med_reasoning,
            high_reasoning=args.high_reasoning,
            xhigh_reasoning=args.xhigh_reasoning,
            cleanup=args.cleanup,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.error("batch_all failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

