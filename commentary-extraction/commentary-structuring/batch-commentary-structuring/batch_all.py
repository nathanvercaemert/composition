#!/usr/bin/env python3
"""End-to-end orchestrator for batch commentary structuring."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

# Load local modules with unique aliases to avoid collisions with extraction modules
MODULE_DIR = Path(__file__).resolve().parent


def _load_local_module(module_name: str):
    path = MODULE_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"commentary_structuring_{module_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load structuring module: {path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure the module is registered before execution so decorators (dataclass) can resolve __module__
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_prepare_batch = _load_local_module("prepare_batch")
prepare_batch_requests = _prepare_batch.prepare_batch_requests

_poll_batch = _load_local_module("poll_batch")
poll_batch = _poll_batch.poll_batch

_process_results = _load_local_module("process_results")
process_results = _process_results.process_results

_retrieve_batch = _load_local_module("retrieve_batch")
retrieve_batch_outputs = _retrieve_batch.retrieve_batch_outputs

_submit_batch = _load_local_module("submit_batch")
submit_batch = _submit_batch.submit_batch

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
    debug: bool = False,
    poll_interval: int = 60,
    input_file: Path | None = None,
    output_dir: Path | None = None,
    temp_dir: Path = DEFAULT_TEMP_DIR,
    reasoning_effort: str = "high",
    use_reasoning: bool = True,
    cleanup: bool = False,
    return_usage: bool = False,
) -> Tuple[Path, Path] | Tuple[Tuple[Path, Path], Dict[str, int]]:
    """Run the entire batch structuring workflow. Returns (json_path, md_path) and optionally token usage."""
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

    token_usage: Dict[str, int] | None = None
    if return_usage:
        token_usage = _collect_token_usage(output_path)

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

    if return_usage and token_usage is not None:
        return (json_output_path, md_output_path), token_usage
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

