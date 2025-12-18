#!/usr/bin/env python3
"""Thin wrapper around batch-commentary-extraction's retrieve_batch without sys.path side effects."""

from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path
from types import ModuleType

import requests

_EXTRACTION_BATCH = Path(__file__).resolve().parent.parent.parent / "batch-commentary-extraction"
_EXTRACTION_RETRIEVE_BATCH = _EXTRACTION_BATCH / "retrieve_batch.py"

logger = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _load_extraction_retrieve_batch() -> ModuleType:
    """Load the extraction retrieve_batch module without mutating sys.path."""
    spec = importlib.util.spec_from_file_location(
        "batch_commentary_extraction.retrieve_batch", _EXTRACTION_RETRIEVE_BATCH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load retrieve_batch from {_EXTRACTION_RETRIEVE_BATCH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _download_file_with_retry(
    api_key: str,
    file_id: str,
    dest: Path,
    *,
    max_retries: int = 3,
    backoff_base: float = 1.5,
) -> None:
    """Download a file's content to disk with simple retry/backoff for transient errors."""
    headers = {"Authorization": f"Bearer {api_key}"}
    attempt = 0

    while True:
        try:
            with requests.get(
                f"https://api.openai.com/v1/files/{file_id}/content",
                headers=headers,
                timeout=600,
                stream=True,
            ) as resp:
                status = resp.status_code
                resp.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
                return
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else None
            attempt += 1
            if status in RETRYABLE_STATUS_CODES and attempt <= max_retries:
                delay = min(backoff_base * (2 ** (attempt - 1)), 30)
                logger.warning(
                    "Download for file %s failed with status %s (attempt %s/%s). Retrying in %.1fs.",
                    file_id,
                    status,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            raise
        except requests.RequestException:
            attempt += 1
            if attempt <= max_retries:
                delay = min(backoff_base * (2 ** (attempt - 1)), 30)
                logger.warning(
                    "Download for file %s failed (attempt %s/%s). Retrying in %.1fs.",
                    file_id,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            raise


_extraction_retrieve_batch = _load_extraction_retrieve_batch()

# Monkey-patch the extraction retrieve_batch to use retrying downloads.
_extraction_retrieve_batch._download_file = _download_file_with_retry

retrieve_batch_outputs = _extraction_retrieve_batch.retrieve_batch_outputs
main = _extraction_retrieve_batch.main

__all__ = ["retrieve_batch_outputs", "main"]


if __name__ == "__main__":
    main()

