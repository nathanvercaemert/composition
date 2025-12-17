from __future__ import annotations

from typing import Dict, List


def _to_str_or_empty(value: object) -> str:
    """Convert value to string, treating None as empty."""
    return "" if value is None else str(value)


def _format_xref(part: Dict[str, object]) -> str:
    book = _to_str_or_empty(part.get("book"))
    chapter = _to_str_or_empty(part.get("chapter"))
    verse = part.get("verse")
    rng = part.get("range")

    if isinstance(rng, dict) and rng.get("start") is not None and rng.get("end") is not None:
        verse_str = f"{_to_str_or_empty(rng.get('start'))}-{_to_str_or_empty(rng.get('end'))}"
    else:
        verse_str = _to_str_or_empty(verse)

    ref = f"{book} {chapter}:{verse_str}"
    return f"[{ref}]({ref})"


def render_verse_markdown(verse_number: int, parts: List[Dict[str, object]]) -> str:
    """Render a single verse's structured parts as Markdown."""
    if not parts:
        body = "No commentary available for this verse."
    else:
        segments: List[str] = []
        for part in parts:
            kind = part.get("kind")
            if kind == "text":
                segments.append(_to_str_or_empty(part.get("text")))
            elif kind == "original":
                segments.append(f"**{_to_str_or_empty(part.get('ref'))}**")
            elif kind == "xref":
                segments.append(_format_xref(part))
            else:
                segments.append(str(part))
        body = "".join(segments)

    lines = [
        f"## Verse {verse_number}",
        "",
        body,
        "",
    ]
    return "\n".join(lines)


def render_chapter_markdown(book_name: str, chapter_number: int, verses: Dict[str, Dict[str, object]]) -> str:
    """Render the entire chapter as a Markdown document."""
    verse_numbers = sorted(int(k) for k in verses.keys())
    rendered = [
        f"# {book_name} Chapter {chapter_number} — Structured Commentary",
        "",
    ]

    for idx, verse_num in enumerate(verse_numbers):
        parts = verses[str(verse_num)].get("structured_parts") or []
        rendered.append(render_verse_markdown(verse_num, parts))
        if idx < len(verse_numbers) - 1:
            rendered.append("\n---\n")

    return "\n".join(rendered).strip() + "\n"


__all__ = ["render_verse_markdown", "render_chapter_markdown"]

