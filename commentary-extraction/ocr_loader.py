"""
Utilities for loading OCR content for commentary extraction.

This module loads chapter boundary definitions and formats OCR text from all
providers into a single string suitable for inclusion in the system prompt.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

OCR_PROVIDERS = [
    ("DeepSeek OCR", "deepseek-ocr"),
    ("PaddleOCR-VL", "paddleocr-vl"),
    ("Qwen-VL", "qwen-vl"),
]


def load_chapter_boundaries(path: Path) -> Dict[str, List[str]]:
    """
    Load the chapter boundaries JSON file.

    Returns an empty dict if the file is missing or malformed.
    """
    if not path.exists():
        logger.error("chapter_boundaries.json not found at %s", path)
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s: %s", path, exc)
        return {}
    except Exception as exc:
        logger.error("Unable to read %s: %s", path, exc)
        return {}

    if not isinstance(raw, dict):
        logger.error("chapter_boundaries.json should be an object mapping chapter -> pages")
        return {}

    cleaned: Dict[str, List[str]] = {}
    for chapter, pages in raw.items():
        if not isinstance(chapter, str):
            logger.warning("Skipping non-string chapter key: %r", chapter)
            continue
        if not isinstance(pages, list):
            logger.warning("Skipping chapter %s: pages should be a list", chapter)
            continue
        cleaned[chapter] = [str(p).strip() for p in pages if str(p).strip()]
    return cleaned


def _load_provider_text(book_path: Path, provider_name: str, provider_dir: str, page_stem: str) -> str:
    """
    Load OCR text for a single provider/page. Returns a placeholder if missing.
    """
    ocr_path = book_path / provider_dir / f"{page_stem}.txt"
    if not ocr_path.exists():
        logger.warning("%s missing OCR for %s (%s)", provider_name, page_stem, ocr_path)
        return "[OCR not available]"
    try:
        return ocr_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("Failed reading %s for %s: %s", provider_name, page_stem, exc)
        return "[OCR not available]"


def _format_page(book_path: Path, page_stem: str) -> str:
    """
    Format OCR content for a single page across all providers.
    """
    lines: List[str] = [f"=== PAGE: {page_stem} ===", f'<PAGE id="{page_stem}">']
    for provider_name, provider_dir in OCR_PROVIDERS:
        content = _load_provider_text(book_path, provider_name, provider_dir, page_stem)
        lines.append(f'  <OCR source="{provider_name}" page="{page_stem}">')
        lines.append(content)
        lines.append("  </OCR>")
    lines.append("</PAGE>")
    return "\n".join(lines)


def load_chapter_ocr(book_path: Path, page_stems: List[str]) -> str:
    """
    Load and concatenate OCR content from all providers for all pages in a chapter.

    Format:
    === PAGE: {page_stem} ===
    <PAGE id="{page_stem}">
      <OCR source="DeepSeek OCR" page="{page_stem}">
      [OCR text content]
      </OCR>
      <OCR source="PaddleOCR-VL" page="{page_stem}">
      [OCR text content]
      </OCR>
      <OCR source="Qwen-VL" page="{page_stem}">
      [OCR text content]
      </OCR>
    </PAGE>
    """
    if not page_stems:
        logger.warning("No page stems provided for chapter OCR load.")
        return ""

    formatted_pages = [_format_page(book_path, stem) for stem in page_stems]
    return "\n\n".join(formatted_pages)


__all__ = ["load_chapter_boundaries", "load_chapter_ocr", "OCR_PROVIDERS"]

