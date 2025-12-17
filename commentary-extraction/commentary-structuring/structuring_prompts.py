from __future__ import annotations

from textwrap import dedent

# Book abbreviations as defined in the specification
BOOK_ABBREVIATIONS = {
    "Old Testament": [
        "Gen", "Exod", "Lev", "Num", "Deut", "Josh", "Judg", "Ruth",
        "1Sam", "2Sam", "1Kgs", "2Kgs", "1Chr", "2Chr", "Ezra", "Neh",
        "Esth", "Job", "Ps", "Prov", "Eccl", "Song", "Isa", "Jer", "Lam",
        "Ezek", "Dan", "Hos", "Joel", "Amos", "Obad", "Jonah", "Mic",
        "Nah", "Hab", "Zeph", "Hag", "Zech", "Mal",
    ],
    "Deuterocanonical/Apocrypha": [
        "Tob", "Jdt", "AddEsth", "Wis", "Sir", "Bar", "EpJer", "PrAzar",
        "Sus", "Bel", "1Macc", "2Macc", "3Macc", "4Macc", "1Esd", "2Esd", "PrMan",
    ],
    "New Testament": [
        "Matt", "Mark", "Luke", "John", "Acts", "Rom", "1Cor", "2Cor",
        "Gal", "Eph", "Phil", "Col", "1Thess", "2Thess", "1Tim", "2Tim",
        "Titus", "Phlm", "Heb", "Jas", "1Pet", "2Pet", "1John", "2John",
        "3John", "Jude", "Rev",
    ],
}


def _format_book_abbreviations() -> str:
    lines: list[str] = []
    for section, books in BOOK_ABBREVIATIONS.items():
        lines.append(f"{section}: " + ", ".join(books))
    return "\n".join(lines)


# NOTE: Keep literal braces escaped ({{ }}) in the prompt template because we
# format it with .format() below.
SYSTEM_PROMPT_TEMPLATE = dedent("""\
    ## Task & Output
    Parse Bible commentary into a JSON object {{"parts": [...]}}. Return only the JSON, no markdown or explanation.

    ## Fidelity
    You are a parser: copy every character verbatim from the input. Do not paraphrase, add, reorder, summarize, or insert punctuation/words not present. When unsure, preserve the original text unchanged.

    ## Schema Rules
    - Each part in `parts` must include all fields: `kind`, `text`, `ref`, `book`, `chapter`, `verse`, `range`.
    - Fields that do not apply are `null` (not empty strings). `range` is either `null` or {{"start": int, "end": int}}.
    - `kind` is required on every part.
    - For xrefs, use `verse` for a single verse or `range` for spans—never both (set the unused one to `null`).

    ## Part Types
    **text** — populate `text`; set `ref`, `book`, `chapter`, `verse`, `range` to `null`. Preserve whitespace exactly (spaces, tabs, newlines).

    **original** — quotation of the verse being commented on. Detected when the segment ends with `]`; store the text before `]`, trimmed of surrounding whitespace, in `ref`. Set `text`, `book`, `chapter`, `verse`, `range` to `null`.

    **xref** — cross-reference. Populate `book`, `chapter`, and either `verse` or `range`. Set `text`, `ref`, and the unused `verse`/`range` to `null`. Split comma/semicolon lists into separate xref parts.

    ## Cross-Reference Parsing
    | Pattern | Output |
    |---------|--------|
    | `Gen 1:2` | book="Gen", chapter=1, verse=2 |
    | `1:5` (no book) | book="{book_name}", chapter=1, verse=5 |
    | `v. 7` or `vv. 7` | book="{book_name}", chapter={chapter_number}, verse=7 |
    | `vv. 3-7` | book="{book_name}", chapter={chapter_number}, range={{"start":3, "end":7}} |
    | `cf. Rom 8:28` | book="Rom", chapter=8, verse=28 |
    | `ch. 5:3` | book="{book_name}", chapter=5, verse=3 |
    | `Gen 1:2, 4, 6` | Three xrefs: Gen 1:2, Gen 1:4, Gen 1:6 |

    ## Processing
    Emit parts in the exact order found. Anything not recognized as `original` or `xref` becomes a `text` part.

    ## Examples

    ### Example 1: Original + single xrefs
    Input: `Who is this that darkeneth counsel] The verb is used in Ezek. 32:7, 8 of darkening the sun.`
    Output:
    {{"parts": [
      {{"kind": "original", "ref": "Who is this that darkeneth counsel", "text": null, "book": null, "chapter": null, "verse": null, "range": null}},
      {{"kind": "text", "text": " The verb is used in ", "ref": null, "book": null, "chapter": null, "verse": null, "range": null}},
      {{"kind": "xref", "book": "Ezek", "chapter": 32, "verse": 7, "text": null, "ref": null, "range": null}},
      {{"kind": "text", "text": ", ", "ref": null, "book": null, "chapter": null, "verse": null, "range": null}},
      {{"kind": "xref", "book": "Ezek", "chapter": 32, "verse": 8, "text": null, "ref": null, "range": null}},
      {{"kind": "text", "text": " of darkening the sun.", "ref": null, "book": null, "chapter": null, "verse": null, "range": null}}
    ]}}

    ### Example 2: Verse range xref
    Input: `Compare the similar imagery in Ps. 104:1-4 where God is described as clothed with light.`
    Output:
    {{"parts": [
      {{"kind": "text", "text": "Compare the similar imagery in ", "ref": null, "book": null, "chapter": null, "verse": null, "range": null}},
      {{"kind": "xref", "book": "Ps", "chapter": 104, "range": {{"start": 1, "end": 4}}, "text": null, "ref": null, "verse": null}},
      {{"kind": "text", "text": " where God is described as clothed with light.", "ref": null, "book": null, "chapter": null, "verse": null, "range": null}}
    ]}}

    ### Example 3: Internal chapter range
    Input: `This theme continues from vv. 4-7 where the foundations of the earth are described.`
    Output:
    {{"parts": [
      {{"kind": "text", "text": "This theme continues from ", "ref": null, "book": null, "chapter": null, "verse": null, "range": null}},
      {{"kind": "xref", "book": "{book_name}", "chapter": {chapter_number}, "range": {{"start": 4, "end": 7}}, "text": null, "ref": null, "verse": null}},
      {{"kind": "text", "text": " where the foundations of the earth are described.", "ref": null, "book": null, "chapter": null, "verse": null, "range": null}}
    ]}}

    ### Example 4: Plain text only
    Input: `The Hebrew word here is very rare and occurs only in late texts.`
    Output:
    {{"parts": [
      {{"kind": "text", "text": "The Hebrew word here is very rare and occurs only in late texts.", "ref": null, "book": null, "chapter": null, "verse": null, "range": null}}
    ]}}

    ### Example 5: Comma-separated xrefs
    Input: `See also Gen. 1:2, 6, 9 for the creation narrative structure.`
    Output:
    {{"parts": [
      {{"kind": "text", "text": "See also ", "ref": null, "book": null, "chapter": null, "verse": null, "range": null}},
      {{"kind": "xref", "book": "Gen", "chapter": 1, "verse": 2, "text": null, "ref": null, "range": null}},
      {{"kind": "text", "text": ", ", "ref": null, "book": null, "chapter": null, "verse": null, "range": null}},
      {{"kind": "xref", "book": "Gen", "chapter": 1, "verse": 6, "text": null, "ref": null, "range": null}},
      {{"kind": "text", "text": ", ", "ref": null, "book": null, "chapter": null, "verse": null, "range": null}},
      {{"kind": "xref", "book": "Gen", "chapter": 1, "verse": 9, "text": null, "ref": null, "range": null}},
      {{"kind": "text", "text": " for the creation narrative structure.", "ref": null, "book": null, "chapter": null, "verse": null, "range": null}}
    ]}}

    ## Reference Data
    Book abbreviations: {_book_abbreviations}

    Context: current book = {book_name}, current chapter = {chapter_number}""")


def build_structuring_system_prompt(book_name: str, chapter_number: int) -> str:
    """Construct the system prompt for the structuring model."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        _book_abbreviations=_format_book_abbreviations(),
        book_name=book_name,
        chapter_number=chapter_number,
    )


def build_structuring_user_prompt(
    book_name: str,
    chapter_number: int,
    verse_number: int,
    commentary_text: str,
) -> str:
    """Construct the user prompt containing only the commentary text to parse.

    The verse context is already provided in the system prompt, so we pass only
    the raw commentary text here to avoid the model parsing template metadata
    as commentary content.
    """
    return commentary_text


__all__ = [
    "build_structuring_system_prompt",
    "build_structuring_user_prompt",
    "BOOK_ABBREVIATIONS",
]
