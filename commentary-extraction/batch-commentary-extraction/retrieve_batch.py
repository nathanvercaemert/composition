#!/usr/bin/env python3
"""Download completed batch outputs from OpenAI."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import requests

# Ensure project root is importable for shared modules (verses.py, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import _read_api_key  # noqa: E402

logger = logging.getLogger(__name__)

BATCHES_URL = "https://api.openai.com/v1/batches"
FILES_URL = "https://api.openai.com/v1/files"
DEFAULT_OUTPUT_DIR = Path("temp/batch-commentary-extraction")
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _load_batch_ref(batch_arg: str) -> Tuple[str, Path | None, Dict[str, object]]:
    """Return (batch_id, batch_file_path, batch_file_data)."""
    candidate = Path(batch_arg)
    if candidate.exists():
        data = json.loads(candidate.read_text(encoding="utf-8"))
        return str(data["batch_id"]), candidate, data
    return batch_arg, None, {}


def _fetch_batch(api_key: str, batch_id: str) -> Dict[str, object]:
    """Fetch batch status from the API."""
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(f"{BATCHES_URL}/{batch_id}", headers=headers, timeout=300)
    resp.raise_for_status()
    return resp.json()


def _download_file(
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
                f"{FILES_URL}/{file_id}/content", headers=headers, timeout=600, stream=True
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
            # Network/connection errors without HTTP status
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


def _infer_book_chapter(batch_file_data: Dict[str, object]) -> Tuple[str | None, int | None]:
    """Attempt to infer book/chapter from batch file or linked meta."""
    meta_path = batch_file_data.get("meta_file_path")
    if meta_path:
        meta_candidate = Path(meta_path)
        if meta_candidate.exists():
            try:
                meta = json.loads(meta_candidate.read_text(encoding="utf-8"))
                return meta.get("book"), int(meta.get("chapter"))
            except Exception:
                logger.debug("Unable to parse meta file at %s", meta_candidate)
    jsonl_path = batch_file_data.get("jsonl_file_path")
    if jsonl_path:
        stem = Path(jsonl_path).stem.replace("_requests", "")
        if "_" in stem:
            book, chapter_str = stem.split("_", 1)
            try:
                return book, int(chapter_str)
            except ValueError:
                return None, None
    return None, None


def retrieve_batch_outputs(
    batch_arg: str,
    *,
    output_dir: Path | None = None,
    include_errors: bool = False,
) -> Tuple[Path, Path | None]:
    """Retrieve batch outputs (and optional errors). Returns (output_path, error_path)."""
    api_key = _read_api_key()
    batch_id, batch_file, batch_data = _load_batch_ref(batch_arg)
    batch_status = _fetch_batch(api_key, batch_id)

    output_file_id = batch_status.get("output_file_id")
    if not output_file_id:
        raise RuntimeError(f"Batch {batch_id} is not completed or has no output_file_id yet (status={batch_status.get('status')})")

    error_file_id = batch_status.get("error_file_id")

    book, chapter = _infer_book_chapter(batch_data)

    target_dir = output_dir or (batch_file.parent if batch_file else DEFAULT_OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    if book and chapter:
        output_path = target_dir / f"{book}_{chapter}_output.jsonl"
        error_path = target_dir / f"{book}_{chapter}_errors.jsonl"
    else:
        output_path = target_dir / f"{batch_id}_output.jsonl"
        error_path = target_dir / f"{batch_id}_errors.jsonl"

    _download_file(api_key, output_file_id, output_path)
    logger.info("Downloaded output to %s", output_path)

    downloaded_error_path: Path | None = None
    if include_errors and error_file_id:
        _download_file(api_key, error_file_id, error_path)
        downloaded_error_path = error_path
        logger.info("Downloaded errors to %s", error_path)

    if batch_file:
        updates = {
            "output_file_id": output_file_id,
            "error_file_id": error_file_id,
            "output_file_path": str(output_path),
        }
        if downloaded_error_path:
            updates["error_file_path"] = str(downloaded_error_path)
        batch_data.update(updates)
        batch_file.write_text(json.dumps(batch_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Downloaded responses to: {output_path}")
    if downloaded_error_path:
        print(f"Errors: {downloaded_error_path}")

    return output_path, downloaded_error_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download batch output files.")
    parser.add_argument("batch_id_or_file", help="Batch ID or path to batch.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for downloaded files (default: alongside batch.json or temp/batch-commentary-extraction)",
    )
    parser.add_argument("--include-errors", action="store_true", help="Also download error file if present")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    try:
        retrieve_batch_outputs(
            args.batch_id_or_file,
            output_dir=args.output_dir,
            include_errors=args.include_errors,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.error("Failed to retrieve batch outputs: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

