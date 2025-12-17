"""
JSON schemas for commentary extraction outputs.

The schemas are intentionally minimal to keep Responses API outputs tight and
cache-friendly.
"""

from __future__ import annotations

VERSE_COMMENTARY_SCHEMA = {
    "type": "object",
    "properties": {
        "has_commentary": {
            "type": "boolean",
            "description": "True if verse-specific commentary exists for this verse",
        },
        "commentary": {
            "type": ["string", "null"],
            "description": "The extracted verse-specific commentary text, or null if none exists",
        },
    },
    "required": ["has_commentary", "commentary"],
    "additionalProperties": False,
}

__all__ = ["VERSE_COMMENTARY_SCHEMA"]

