#!/usr/bin/env python3
"""Upload batch requests and create an OpenAI Batch job."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Tuple

import requests

# Ensure project root is importable for shared modules (verses.py, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import _read_api_key  # noqa: E402

logger = logging.getLogger(__name__)

BATCHES_URL = "https://api.openai.com/v1/batches"
FILES_URL = "https://api.openai.com/v1/files"
DEFAULT_COMPLETION_WINDOW = "24h"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _parse_metadata_args(pairs: list[str]) -> Dict[str, str]:
    """Parse repeated KEY=VALUE metadata arguments into a dict."""
    metadata: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Metadata must be KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _normalize_completion_window(value: str) -> str:
    """Ensure completion_window string ends with 'h' when given hours."""
    if value.lower().endswith("h"):
        return value
    if value.isdigit():
        return f"{value}h"
    return value


def _infer_book_chapter(jsonl_path: Path, meta_path: Path | None) -> Tuple[str | None, int | None]:
    """Try to determine book and chapter from meta file or filename."""
    if meta_path and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return meta.get("book"), int(meta.get("chapter"))
        except Exception:
            logger.debug("Failed to read meta file at %s", meta_path)

    stem = jsonl_path.stem.replace("_requests", "")
    if "_" in stem:
        book, chapter_str = stem.split("_", 1)
        try:
            return book, int(chapter_str)
        except ValueError:
            return None, None
    return None, None


def _backoff_delay(
    attempt: int,
    *,
    base: float = 1.5,
    max_delay: float = 60.0,
    jitter_ratio: float = 0.25,
) -> float:
    """Compute an exponential backoff delay with optional jitter."""
    delay = min(base * (2 ** (attempt - 1)), max_delay)
    if jitter_ratio > 0:
        jitter = delay * jitter_ratio
        delay = max(0.0, delay + random.uniform(-jitter, jitter))
    return delay


def _upload_file(
    api_key: str,
    jsonl_path: Path,
    *,
    max_retries: int = 6,
    backoff_base: float = 1.5,
    max_backoff: float = 60.0,
) -> Dict[str, object]:
    """Upload the JSONL file to OpenAI Files API with resilient retry/backoff for 5xx/429 errors."""
    headers = {"Authorization": f"Bearer {api_key}"}
    attempt = 0

    while True:
        attempt += 1
        try:
            with jsonl_path.open("rb") as fh:
                files = {"file": (jsonl_path.name, fh, "application/jsonl")}
                data = {"purpose": "batch"}
                resp = requests.post(FILES_URL, headers=headers, files=files, data=data, timeout=600)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else None
            if status in RETRYABLE_STATUS_CODES and attempt <= max_retries:
                delay = _backoff_delay(attempt, base=backoff_base, max_delay=max_backoff)
                logger.warning(
                    "Upload to %s failed with status %s (attempt %s/%s). Retrying in %.1fs.",
                    FILES_URL,
                    status,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            raise
        except requests.RequestException:
            if attempt <= max_retries:
                delay = _backoff_delay(attempt, base=backoff_base, max_delay=max_backoff)
                logger.warning(
                    "Upload to %s failed (attempt %s/%s). Retrying in %.1fs.",
                    FILES_URL,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            raise


def _create_batch(
    api_key: str,
    *,
    input_file_id: str,
    completion_window: str,
    metadata: Dict[str, str] | None,
    max_retries: int = 6,
    backoff_base: float = 1.5,
    max_backoff: float = 60.0,
) -> Dict[str, object]:
    """Create a batch job via the OpenAI Batches API with retry/backoff on transient errors."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/responses",
        "completion_window": completion_window,
    }
    if metadata:
        payload["metadata"] = metadata

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(BATCHES_URL, headers=headers, data=json.dumps(payload), timeout=600)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else None
            if status in RETRYABLE_STATUS_CODES and attempt <= max_retries:
                delay = _backoff_delay(attempt, base=backoff_base, max_delay=max_backoff)
                logger.warning(
                    "Batch creation failed with status %s (attempt %s/%s). Retrying in %.1fs.",
                    status,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            raise
        except requests.RequestException:
            if attempt <= max_retries:
                delay = _backoff_delay(attempt, base=backoff_base, max_delay=max_backoff)
                logger.warning(
                    "Batch creation request failed (attempt %s/%s). Retrying in %.1fs.",
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            raise


def submit_batch(
    jsonl_file: Path,
    *,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    metadata: Dict[str, str] | None = None,
    meta_file: Path | None = None,
) -> Path:
    """Upload a JSONL file and create a batch job. Returns the batch info path."""
    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")

    api_key = _read_api_key()
    completion_window = _normalize_completion_window(completion_window)
    meta_file = meta_file if meta_file and meta_file.exists() else jsonl_file.with_name(jsonl_file.stem.replace("_requests", "") + "_meta.json")

    book, chapter = _infer_book_chapter(jsonl_file, meta_file if meta_file.exists() else None)

    upload_resp = _upload_file(api_key, jsonl_file)
    input_file_id = upload_resp.get("id")
    if not input_file_id:
        raise RuntimeError(f"Upload did not return file id: {upload_resp}")

    batch_resp = _create_batch(
        api_key,
        input_file_id=input_file_id,
        completion_window=completion_window,
        metadata=metadata,
    )

    batch_id = batch_resp.get("id")
    status = batch_resp.get("status")

    created_at = batch_resp.get("created_at")
    created_iso = datetime.fromtimestamp(created_at, tz=UTC).isoformat().replace("+00:00", "Z") if isinstance(created_at, (int, float)) else None

    if book and chapter:
        batch_info_path = jsonl_file.parent / f"{book}_{chapter}_batch.json"
    else:
        batch_info_path = jsonl_file.with_name(jsonl_file.stem + "_batch.json")

    batch_info = {
        "batch_id": batch_id,
        "input_file_id": input_file_id,
        "status": status,
        "created_at": created_iso,
        "jsonl_file_path": str(jsonl_file),
        "meta_file_path": str(meta_file) if meta_file and meta_file.exists() else None,
        "completion_window": completion_window,
        "metadata": metadata or {},
    }
    batch_info_path.write_text(json.dumps(batch_info, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Batch created: {batch_id} (status: {status})")
    logger.info("Saved batch info to %s", batch_info_path)
    return batch_info_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload JSONL and create an OpenAI batch job.")
    parser.add_argument("jsonl_file", type=Path, help="Path to the JSONL requests file")
    parser.add_argument(
        "--completion-window",
        default=DEFAULT_COMPLETION_WINDOW,
        help='Completion window (hours or string, default: "24h")',
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Add metadata key=value (may be repeated)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    metadata = _parse_metadata_args(args.metadata) if args.metadata else {}
    try:
        submit_batch(
            args.jsonl_file,
            completion_window=args.completion_window,
            metadata=metadata or None,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.error("Failed to submit batch: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

