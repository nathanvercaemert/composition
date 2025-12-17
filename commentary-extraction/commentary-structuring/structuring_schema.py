from __future__ import annotations

"""
JSON schema for structured commentary parts.

Conceptually, a structured commentary is a discriminated union of three part
types:
- text: plain commentary content with whitespace preserved
- original: a quotation of the verse being commented on
- xref: a cross-reference to another passage, identified by book/chapter and
  either a single verse or a verse range

In an ideal schema this would be expressed with a `oneOf` per part type, for
example:
    {
      "items": {"$ref": "#/$defs/Part"},
      "$defs": {
        "Part": {
          "oneOf": ["TextPart", "OriginalPart", "CrossReferencePart"]
        }
      }
    }

However, the Responses API currently enforces several constraints:
- The top-level schema must be an object (not an array), so we wrap parts in a
  `parts` property even though consumers work with a plain list in code.
- Every property must appear in the `required` list when strict validation is
  enabled.
- Combinators such as `oneOf`/`anyOf`/`if`/`then`/`else` are rejected.

To stay compliant, the implementation uses a single flat part object where all
possible fields are present and unused fields are set to null. The prompt
instructs the model to populate only the fields that are semantically active for
the given `kind`. The intended semantics for each kind are:
- text: `text` is populated; `ref`, `book`, `chapter`, `verse`, and `range` are
  null.
- original: `ref` is populated; `text`, `book`, `chapter`, `verse`, and `range`
  are null.
- xref: `book` and `chapter` are populated and either `verse` or `range` (but
  not both) is populated; `text` and `ref` are null. Exactly one of `verse` or
  `range` must be non-null.

Important: The structuring model operates under a strict fidelity constraint.
All text content in the output must appear verbatim in the input commentary.
The model functions as a parser/classifier, not a content generator.
"""

PART_TYPE_DEFINITIONS = {
    "text": {
        "description": "Plain commentary text with whitespace preserved exactly.",
        "active_fields": ["text"],
        "null_fields": ["ref", "book", "chapter", "verse", "range"],
    },
    "original": {
        "description": "Quotation of the verse being commented on (original text).",
        "active_fields": ["ref"],
        "null_fields": ["text", "book", "chapter", "verse", "range"],
    },
    "xref": {
        "description": (
            "Cross-reference to another passage. Requires book and chapter and "
            "either a single verse or a verse range."
        ),
        "active_fields": ["book", "chapter"],
        "conditional_fields": {
            "verse": "Single verse reference (mutually exclusive with range).",
            "range": "Verse range reference (mutually exclusive with verse).",
        },
        "null_fields": ["text", "ref"],
        "notes": ["Exactly one of `verse` or `range` must be non-null."],
    },
}

STRUCTURED_COMMENTARY_SCHEMA = {
    "title": "Structured Commentary Parts",
    "type": "object",
    "description": (
        "Object containing the ordered parts representing a parsed verse commentary. "
        "The flat shape below approximates a discriminated union required by the "
        "Responses API (all properties required, unused fields set to null)."
    ),
    "required": ["parts"],
    "additionalProperties": False,
    "properties": {
        "parts": {
            "type": "array",
            "description": (
                "Ordered sequence of parts representing a parsed verse commentary. "
                "Conceptually each item is one of three kinds (text, original, xref) "
                "but represented here as a flat object with nullable unused fields."
            ),
            "items": {
                "type": "object",
                "description": (
                    "A single part in the structured commentary. The `kind` field "
                    "discriminates between three logical types: `text` (plain "
                    "commentary), `original` (verse quotation), and `xref` "
                    "(cross-reference). Fields not applicable to the current kind "
                    "must be null. This flat structure is required by the API; "
                    "conceptually these are distinct types."
                ),
                "additionalProperties": False,
                "required": ["kind", "text", "ref", "book", "chapter", "verse", "range"],
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["text", "original", "xref"],
                        "description": (
                            "Discriminator field indicating the part type. Determines "
                            "which other fields are semantically active: `text` uses "
                            "only the text field; `original` uses only the ref field; "
                            "`xref` uses book, chapter, and either verse or range."
                        ),
                    },
                    "text": {
                        "type": ["string", "null"],
                        "description": (
                            "Plain text with whitespace preserved exactly. Active when "
                            "kind='text'; null otherwise."
                        ),
                    },
                    "ref": {
                        "type": ["string", "null"],
                        "description": (
                            "Original verse quotation (bracket and surrounding whitespace "
                            "removed). Active when kind='original'; null otherwise."
                        ),
                    },
                    "book": {
                        "type": ["string", "null"],
                        "description": (
                            "Book abbreviation (e.g., Gen, Matt, Rom). Active when "
                            "kind='xref'; null otherwise."
                        ),
                    },
                    "chapter": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": "Chapter number. Active when kind='xref'; null otherwise.",
                    },
                    "verse": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": (
                            "Single verse number. Active when kind='xref' and range is "
                            "null; null otherwise. Mutually exclusive with range."
                        ),
                    },
                    "range": {
                        "type": ["object", "null"],
                        "additionalProperties": False,
                        "required": ["start", "end"],
                        "description": (
                            "Inclusive verse range for cross references. Active when "
                            "kind='xref' and verse is null; null otherwise. Mutually "
                            "exclusive with verse."
                        ),
                        "properties": {
                            "start": {"type": "integer", "minimum": 1},
                            "end": {"type": "integer", "minimum": 1},
                        },
                    },
                },
            },
        }
    },
}


def validate_part_structure(part: dict[str, object]) -> tuple[bool, str | None]:
    """
    Validate a single part against the conceptual discriminated-union rules that
    the flat JSON Schema cannot enforce (e.g., which fields must be null for a
    given kind and verse/range exclusivity).

    Returns:
        (is_valid, error_message). error_message is None when valid.
    """

    required_keys = {"kind", "text", "ref", "book", "chapter", "verse", "range"}
    missing = required_keys - set(part.keys())
    if missing:
        return False, f"Missing keys: {sorted(missing)}"

    kind = part.get("kind")
    if kind not in PART_TYPE_DEFINITIONS:
        return False, f"Invalid kind: {kind!r}"

    if kind == "text":
        if not isinstance(part.get("text"), str):
            return False, "kind='text' requires a string text field"
        for key in ("ref", "book", "chapter", "verse", "range"):
            if part.get(key) is not None:
                return False, f"kind='text' requires {key} to be null"
        return True, None

    if kind == "original":
        if not isinstance(part.get("ref"), str):
            return False, "kind='original' requires a string ref field"
        for key in ("text", "book", "chapter", "verse", "range"):
            if part.get(key) is not None:
                return False, f"kind='original' requires {key} to be null"
        return True, None

    # kind == "xref"
    if not isinstance(part.get("book"), str):
        return False, "kind='xref' requires a string book field"
    chapter = part.get("chapter")
    if not isinstance(chapter, int) or chapter < 1:
        return False, "kind='xref' requires a positive integer chapter field"
    for key in ("text", "ref"):
        if part.get(key) is not None:
            return False, f"kind='xref' requires {key} to be null"

    verse = part.get("verse")
    range_value = part.get("range")
    if verse is None and range_value is None:
        return False, "kind='xref' requires either verse or range"
    if verse is not None and range_value is not None:
        return False, "kind='xref' must not have both verse and range"

    if verse is not None:
        if not isinstance(verse, int) or verse < 1:
            return False, "verse must be a positive integer when provided"
        return True, None

    if not isinstance(range_value, dict):
        return False, "range must be an object with start/end when provided"
    start = range_value.get("start")
    end = range_value.get("end")
    if not isinstance(start, int) or start < 1:
        return False, "range.start must be a positive integer"
    if not isinstance(end, int) or end < 1:
        return False, "range.end must be a positive integer"
    if end < start:
        return False, "range.end must be greater than or equal to range.start"
    return True, None


__all__ = ["STRUCTURED_COMMENTARY_SCHEMA", "PART_TYPE_DEFINITIONS", "validate_part_structure"]
