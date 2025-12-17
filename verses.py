#!/usr/bin/env python3
"""Utility module for working with ``verses.json`` data.

This module provides a reusable, process-agnostic API for loading and querying
Bible book, chapter, and verse metadata from ``verses.json``. It is designed to
be imported by any script or application that needs verse metadata without
assuming a specific workflow.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

DEFAULT_VERSES_FILENAME = "verses.json"


# Exceptions -----------------------------------------------------------------
class VersesDataError(Exception):
    """Base exception for verses data issues."""


class VersesFileNotFoundError(VersesDataError):
    """Raised when the verses.json file cannot be located."""


class VersesParseError(VersesDataError):
    """Raised when verses.json is malformed or cannot be parsed."""


class BookNotFoundError(VersesDataError):
    """Raised when a requested book abbreviation is missing."""


class ChapterOutOfRangeError(VersesDataError):
    """Raised when a requested chapter number is invalid for a book."""


# Data model -----------------------------------------------------------------
@dataclass(frozen=True)
class BookInfo:
    """Immutable representation of a single Bible book."""

    abbreviation: str
    chapter_count: int
    verse_counts: tuple[int, ...]

    def verse_count(self, chapter: int) -> int:
        """Return the number of verses in the given 1-indexed chapter."""
        if chapter < 1 or chapter > self.chapter_count:
            raise ChapterOutOfRangeError(
                f"Chapter {chapter} is out of range for {self.abbreviation} "
                f"(1-{self.chapter_count})"
            )
        return self.verse_counts[chapter - 1]

    @property
    def total_verses(self) -> int:
        """Total number of verses in the book."""
        return sum(self.verse_counts)

    def __len__(self) -> int:
        return self.chapter_count


# Loader and access API ------------------------------------------------------
class VersesData:
    """Accessor for verses metadata with caching and validation."""

    def __init__(
        self,
        path: str | Path | None = None,
        data: Iterable[BookInfo | Sequence[object]] | None = None,
    ) -> None:
        """
        Initialize the verses data accessor.

        Args:
            path: Optional path to ``verses.json``. If omitted, the loader will
                try to locate the file beside this module, then one directory up,
                and finally in the current working directory.
            data: Optional iterable of ``BookInfo`` instances or raw entries in
                the ``verses.json`` format for testing without file I/O.
        """
        self._path: Path | None = Path(path).expanduser() if path else None
        self._provided_data = data
        self._books_by_abbrev: dict[str, BookInfo] = {}
        self._book_order: list[str] = []
        self._loaded = False
        self._lock = RLock()

    # Core loading -----------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Load verses data on first access."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            if self._provided_data is not None:
                books = self._build_books_from_iterable(self._provided_data)
                self._path = self._path or Path(DEFAULT_VERSES_FILENAME)
            else:
                path = self._resolve_path()
                books = self._load_from_file(path)
                self._path = path

            self._books_by_abbrev = {book.abbreviation: book for book in books}
            self._book_order = [book.abbreviation for book in books]
            self._loaded = True
            logger.debug("Loaded %s books from %s", len(books), self._path)

    def _resolve_path(self) -> Path:
        """Determine the verses.json path using supplied or default locations."""
        if self._path is not None:
            path = self._path
            if not path.exists():
                raise VersesFileNotFoundError(f"verses.json not found at {path}")
            return path

        module_dir = Path(__file__).resolve().parent
        candidates = [
            module_dir / DEFAULT_VERSES_FILENAME,
            module_dir.parent / DEFAULT_VERSES_FILENAME,
            Path.cwd() / DEFAULT_VERSES_FILENAME,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise VersesFileNotFoundError(
            f"verses.json not found. Tried: {', '.join(str(p) for p in candidates)}"
        )

    def _load_from_file(self, path: Path) -> list[BookInfo]:
        """Read and validate verses.json from disk."""
        if not path.exists():
            raise VersesFileNotFoundError(f"verses.json not found at {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VersesParseError(f"Invalid JSON in {path}: {exc}") from exc
        except Exception as exc:
            raise VersesDataError(f"Unable to read {path}: {exc}") from exc

        if not isinstance(raw, list):
            raise VersesParseError(f"Expected a list at top level, found {type(raw).__name__}")
        return self._build_books_from_iterable(raw)

    def _build_books_from_iterable(
        self, entries: Iterable[BookInfo | Sequence[object]]
    ) -> list[BookInfo]:
        """Normalize and validate raw entries into ``BookInfo`` instances."""
        books: list[BookInfo] = []
        seen: set[str] = set()
        for idx, entry in enumerate(entries):
            if isinstance(entry, BookInfo):
                book = entry
            else:
                book = self._normalize_entry(entry, idx)

            if book is None:
                continue

            if book.abbreviation in seen:
                logger.warning("Duplicate book abbreviation %s; keeping first occurrence", book.abbreviation)
                continue

            seen.add(book.abbreviation)
            books.append(book)

        if not books:
            raise VersesParseError("No valid entries found in verses data.")

        return books

    def _normalize_entry(self, entry: Sequence[object], index: int) -> BookInfo | None:
        """Validate a single raw entry from verses.json."""
        if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)):
            logger.warning("Skipping malformed entry at index %s: %s", index, entry)
            return None

        if len(entry) != 3:
            logger.warning(
                "Skipping entry at index %s: expected 3 elements, found %s (%s)",
                index,
                len(entry),
                entry,
            )
            return None

        abbr, chapter_count, verse_counts_raw = entry

        if not isinstance(abbr, str) or not abbr.strip():
            logger.warning("Skipping entry at index %s: invalid abbreviation %r", index, abbr)
            return None
        abbr = abbr.strip()

        if not isinstance(chapter_count, int) or chapter_count <= 0:
            logger.warning(
                "Skipping entry for %s: invalid chapter count %r", abbr, chapter_count
            )
            return None

        if not isinstance(verse_counts_raw, Sequence) or isinstance(verse_counts_raw, (str, bytes)):
            logger.warning("Skipping entry for %s: verse counts must be a list/tuple", abbr)
            return None

        verse_counts: list[int] = []
        for chapter_idx, value in enumerate(verse_counts_raw, start=1):
            if not isinstance(value, int) or value <= 0:
                logger.warning(
                    "Skipping entry for %s: invalid verse count at chapter %s: %r",
                    abbr,
                    chapter_idx,
                    value,
                )
                return None
            verse_counts.append(int(value))

        if len(verse_counts) != chapter_count:
            logger.warning(
                "Chapter count mismatch for %s: header says %s chapters, verse_counts has %s; "
                "using %s from verse_counts.",
                abbr,
                chapter_count,
                len(verse_counts),
                len(verse_counts),
            )
            chapter_count = len(verse_counts)

        return BookInfo(abbreviation=abbr, chapter_count=chapter_count, verse_counts=tuple(verse_counts))

    # Public API -------------------------------------------------------------
    @property
    def path(self) -> Path | None:
        """Resolved path to verses.json, if available."""
        return self._path

    def reload(self) -> None:
        """Force a reload from disk or provided data."""
        with self._lock:
            self._loaded = False
            self._books_by_abbrev.clear()
            self._book_order.clear()
        self._ensure_loaded()

    def get_book(self, abbreviation: str) -> BookInfo:
        """Return ``BookInfo`` for the given abbreviation."""
        self._ensure_loaded()
        if abbreviation not in self._books_by_abbrev:
            raise BookNotFoundError(f"Book abbreviation {abbreviation!r} not found")
        return self._books_by_abbrev[abbreviation]

    def has_book(self, abbreviation: str) -> bool:
        """Return True if the abbreviation exists."""
        self._ensure_loaded()
        return abbreviation in self._books_by_abbrev

    def list_books(self) -> list[str]:
        """Return all book abbreviations in canonical order."""
        self._ensure_loaded()
        return list(self._book_order)

    def get_book_index(self, abbreviation: str) -> int:
        """Return the canonical 0-based index for the book."""
        self._ensure_loaded()
        if abbreviation not in self._books_by_abbrev:
            raise BookNotFoundError(f"Book abbreviation {abbreviation!r} not found")
        return self._book_order.index(abbreviation)

    def get_chapter_count(self, abbreviation: str) -> int:
        """Return the number of chapters for the book."""
        return self.get_book(abbreviation).chapter_count

    def get_verse_count(self, abbreviation: str, chapter: int) -> int:
        """Return the number of verses in the given 1-indexed chapter."""
        return self.get_book(abbreviation).verse_count(chapter)

    def get_all_verse_counts(self, abbreviation: str) -> tuple[int, ...]:
        """Return all verse counts for the book as an immutable tuple."""
        return self.get_book(abbreviation).verse_counts

    def get_total_verses(self, abbreviation: str) -> int:
        """Return the total number of verses for a book."""
        return self.get_book(abbreviation).total_verses

    def get_total_chapters_all_books(self) -> int:
        """Return the total chapters across all loaded books."""
        self._ensure_loaded()
        return sum(self._books_by_abbrev[abbr].chapter_count for abbr in self._book_order)

    def get_total_verses_all_books(self) -> int:
        """Return the total verses across all loaded books."""
        self._ensure_loaded()
        return sum(self._books_by_abbrev[abbr].total_verses for abbr in self._book_order)

    def is_valid_reference(self, abbreviation: str, chapter: int, verse: int) -> bool:
        """Return True if the reference exists in the data."""
        if chapter < 1 or verse < 1:
            return False
        try:
            book = self.get_book(abbreviation)
        except BookNotFoundError:
            return False
        if chapter > book.chapter_count:
            return False
        if verse > book.verse_counts[chapter - 1]:
            return False
        return True

    def get_reference_range(self, abbreviation: str) -> str:
        """Return a string describing the full verse range for the book."""
        book = self.get_book(abbreviation)
        last_chapter = book.chapter_count
        last_verse = book.verse_counts[-1]
        return f"{book.abbreviation} 1:1 - {book.abbreviation} {last_chapter}:{last_verse}"

    # Dunder helpers ---------------------------------------------------------
    def __contains__(self, abbreviation: object) -> bool:
        return isinstance(abbreviation, str) and self.has_book(abbreviation)

    def __iter__(self) -> Iterator[BookInfo]:
        self._ensure_loaded()
        return (self._books_by_abbrev[abbr] for abbr in self._book_order)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._book_order)

    def __repr__(self) -> str:
        path_display = str(self._path) if self._path else "<auto>"
        return f"VersesData(books={len(self)}, path={path_display})"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    verses = VersesData()
    try:
        first_book = verses.list_books()[0]
        print(f"Loaded {len(verses)} books from {verses.path or 'auto-detected path'}")
        print(f"{first_book} has {verses.get_chapter_count(first_book)} chapters.")
        print(f"{first_book} 1:1 - first chapter has {verses.get_verse_count(first_book, 1)} verses.")
        print(f"Is {first_book} 1:1 valid? {verses.is_valid_reference(first_book, 1, 1)}")
        print(f"Total verses in Bible: {verses.get_total_verses_all_books():,}")
    except VersesDataError as exc:  # pragma: no cover - convenience demo
        logger.error("Failed to load verses data: %s", exc)

