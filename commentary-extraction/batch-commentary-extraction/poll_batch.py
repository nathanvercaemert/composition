#!/usr/bin/env python3
"""Poll OpenAI batch job status."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import requests

# Ensure project root is importable for shared modules (verses.py, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import _read_api_key  # noqa: E402

logger = logging.getLogger(__name__)

BATCHES_URL = "https://api.openai.com/v1/batches"
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def _load_batch_ref(batch_arg: str) -> Tuple[str, Path | None]:
    """Return (batch_id, batch_file_path)."""
    candidate = Path(batch_arg)
    if candidate.exists():
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return str(data["batch_id"]), candidate
        except Exception as exc:
            raise ValueError(f"Failed to read batch id from {candidate}: {exc}") from exc
    return batch_arg, None


def _fetch_batch(api_key: str, batch_id: str) -> Dict[str, object]:
    """Fetch batch status from the API."""
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(f"{BATCHES_URL}/{batch_id}", headers=headers, timeout=300)
    resp.raise_for_status()
    return resp.json()


def _save_batch_updates(batch_file: Path, payload: Dict[str, object]) -> None:
    """Merge latest batch info into existing batch file."""
    try:
        existing = json.loads(batch_file.read_text(encoding="utf-8"))
    except Exception:
        existing = {}
    existing.update(payload)
    batch_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_status(batch_id: str, batch: Dict[str, object]) -> None:
    """Print concise status/progress to stdout."""
    status = batch.get("status", "unknown")
    counts = batch.get("request_counts", {}) or {}
    total = counts.get("total", 0)
    completed = counts.get("completed", 0)
    failed = counts.get("failed", 0)
    print(f"Batch: {batch_id}")
    print(f"Status: {status}")
    print(f"Progress: {completed}/{total} completed, {failed} failed")


def poll_batch(
    batch_arg: str,
    *,
    watch: bool = False,
    interval: int = 60,
    max_interval: int = 300,
    timeout_minutes: int = 1440,
) -> Dict[str, object]:
    """Poll batch status, optionally watching until a terminal state."""
    api_key = _read_api_key()
    batch_id, batch_file = _load_batch_ref(batch_arg)

    start_time = time.monotonic()
    current_interval = max(1, interval)
    timeout_seconds = timeout_minutes * 60

    while True:
        batch = _fetch_batch(api_key, batch_id)
        _print_status(batch_id, batch)

        if batch_file:
            updates = {
                "status": batch.get("status"),
                "output_file_id": batch.get("output_file_id"),
                "error_file_id": batch.get("error_file_id"),
                "request_counts": batch.get("request_counts"),
            }
            _save_batch_updates(batch_file, updates)

        status = batch.get("status", "")
        if not watch or status in TERMINAL_STATUSES:
            return batch

        elapsed = time.monotonic() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(f"Timeout waiting for batch {batch_id} to complete")

        sleep_for = min(current_interval, max_interval)
        logger.info("Waiting %ss before next poll...", sleep_for)
        time.sleep(sleep_for)
        current_interval = min(current_interval * 2, max_interval)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll an OpenAI batch job.")
    parser.add_argument("batch_id_or_file", help="Batch ID (batch_xxx) or path to batch.json")
    parser.add_argument("--watch", action="store_true", help="Continuously poll until completion")
    parser.add_argument("--interval", type=int, default=60, help="Initial polling interval in seconds")
    parser.add_argument("--max-interval", type=int, default=300, help="Maximum polling interval in seconds")
    parser.add_argument(
        "--timeout",
        type=int,
        default=1440,
        help="Maximum time to wait in minutes (default: 24h)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    try:
        poll_batch(
            args.batch_id_or_file,
            watch=args.watch,
            interval=args.interval,
            max_interval=args.max_interval,
            timeout_minutes=args.timeout,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.error("Polling failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

