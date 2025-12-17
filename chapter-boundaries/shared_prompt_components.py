#!/usr/bin/env python3
"""
shared_prompt_components.py

Reusable prompt components for chapter boundary detection and refinement.
Define each concept exactly once and reference from both instructions.py
and refiner_instructions.py.

This module follows the Single Definition Rule: every tagging rule, document
structure description, and OCR guidance appears here once, ensuring consistency
across the initial detection and refinement pipelines.
"""

from __future__ import annotations

# =============================================================================
# DOCUMENT STRUCTURE (Section 2.2 of Metaprompt)
# =============================================================================

DOCUMENT_STRUCTURE = """**Page Layout**:
- **Top**: Scripture text with verse numbers in margins
- **Body**: Verse-specific commentary with marker numbers ("**N.**" with period)
- **Headers**: Book/chapter info (may have low integrity)"""

DOCUMENT_STRUCTURE_EXTENDED = """These are text, verse-numbered religious/scriptural commentary volumes where chapters contain sequential verse-specific commentary marker numbers (**1.**, **2.**, **3.**...) within the body text of the page. Verse commentary marker numbers are the most reliable structural markers.

{document_structure}

Commentary for a single verse may span multiple pages without repeating verse marker numbers.
This is critical: **verse-specific commentary may overflow to a page with a different chapter (or no chapter at all) in the heading.**

Note that the page may contain Roman numerals that group the following chapters in a confusing way. Use the aggregate of multiple page headers to determine the "current" chapter, considering chapter boundaries as edge cases.

Ignore post-print markings (stamps, pen notes, underlines, etc.).""".format(
    document_structure=DOCUMENT_STRUCTURE
)


# =============================================================================
# OCR SOURCE CHARACTERISTICS (Section 2.3 of Metaprompt)
# =============================================================================

OCR_SOURCE_NOTES = """## OCR Source Notes
- **DeepSeek**: Best for prose content; deduce chapters from context and marker numbers
- **PaddleOCR-VL / Qwen-VL**: Better for structural elements (chapter headings)

Triangulate across sources; cross-reference verse-specific commentary marker numbers."""

OCR_SOURCE_NOTES_EXTENDED = """## OCR Source Notes

- **DeepSeek**: Best for prose content and narrative flow context; rarely shows chapter headings; you should deduce from the context (verse-specific commentary marker numbers) and other pages' chapter numbers that a given page is part of a given chapter
- **PaddleOCR-VL / Qwen-VL**: Better for structural elements (headings with chapter numbers)

Triangulate across sources: trust clear readings; cross-reference verse-specific commentary marker numbers."""


# =============================================================================
# TAGGING PHILOSOPHY (Section 2.4 of Metaprompt)
# =============================================================================

TAGGING_PHILOSOPHY = """This is **multi-label tagging**. Each page should be tagged with ALL chapters whose content appears on it.
Prioritize recall over precision. The downstream system resolves overlaps—liberal tagging is preferred."""


# =============================================================================
# TAGGING TRIGGER CONDITIONS (Section 3.1 of Metaprompt)
# =============================================================================

TAGGING_TRIGGERS = """**When to Add a Chapter Tag**:
- The page explicitly mentions that chapter (header, chapter number, Roman numeral)
- The page contains verse-specific commentary marker numbers for that chapter
- Commentary from that chapter overflows onto the page
- The page falls between pages confirmed for that chapter
- The page is near/adjacent to confirmed pages for that chapter
- The OCR is unclear but the chapter is plausible
- Verse-specific commentary for that chapter is present in any form"""


# =============================================================================
# TAGGING GUIDELINES (Header for tagging sections)
# =============================================================================

TAGGING_GUIDELINES = """**Tagging Guidelines**:
- Every page needs at least one tag
- Boundary/ambiguous pages should have multiple tags
- When uncertain, include additional plausible chapters (favor recall over precision)"""


# =============================================================================
# BOUNDARY HANDLING RULES (Section 3.2 of Metaprompt)
# =============================================================================

BOUNDARY_RULES = """**Boundary Handling**:
- Tag transition pages with both the ending and beginning chapters
- Fill gaps: if pages before and after have a tag, intermediate pages likely need it too
- Extend boundaries 1-2 pages in each direction when uncertain"""


# =============================================================================
# OVERFLOW DETECTION (Section 4 of Metaprompt)
# =============================================================================

OVERFLOW_DETECTION = """## Overflow Detection
Commentary from Chapter N may overflow onto a page whose header shows Chapter N+1. Tag such pages with BOTH chapters.

**Overflow Signs**:
- No verse marker at body text start (continuation from previous page)
- Incomplete first sentence or lowercase start
- Scripture references before any marker
- Verse discussion without introducing a new marker"""


# =============================================================================
# CHAPTER BOUNDARY SIGNALS (Section 5 of Metaprompt)
# =============================================================================

CHAPTER_BOUNDARY_SIGNALS = """**Chapter Boundary Indicators**:
- Verse-specific commentary marker numbers resetting (high→low, e.g., "30." → "1.") indicates new chapter
- Continuous sequence of marker numbers across pages = same chapter
- Explicit chapter headings ("Chapter 1", "CHAPTER ONE", Roman numerals)"""

OCR_ERROR_PATTERNS = """**OCR Errors**: "1" may appear as "l", "I", "|"; "8" as "B" or "0". Look for patterns in corrupted text."""


# =============================================================================
# OUTPUT FORMAT RULES (Section 7.3 of Metaprompt)
# =============================================================================

OUTPUT_RULES = """**Critical Rules:**
1. Output ONLY a JSON object; no prose, no markdown fences
2. Every page key must have at least one chapter tag
3. Every page in the batch MUST appear as a key (use exactly the stems provided)
4. Chapter tags are strings representing chapter numbers (e.g., "1", "2", "3")
5. Many pages will have MULTIPLE chapter tags at boundaries
6. Tags should be liberally applied—when in doubt, add the tag"""


# =============================================================================
# VERIFICATION CHECKLIST (Section 7.3 of Metaprompt)
# =============================================================================

VERIFICATION_CHECKLIST = """## Verification Checklist
Before outputting, verify:
□ Every PAGE id present as a key
□ Only the keys listed above
□ Stems match exactly (case-sensitive)
□ Each page has ≥1 tag
□ JSON only — no prose or markdown"""


# =============================================================================
# ROMAN NUMERAL TABLE (Initial detection only)
# =============================================================================

ROMAN_NUMERAL_NOTES = """## Roman Numeral Conversions

Note that some pages may partially belong to a chapter but lack that chapter's roman numeral in the page header. This is common at chapter boundaries. Tag these pages with both chapters.

If there is a string of consecutive pages with chapter A in the heading, only tag one or two of those pages with B, and do so at the border of A and B.

Remember to consider exceptions around chapter boundaries where verse-specific commentary text has overflowed from the previous chapter (so you add an additional tag for the previous chapter to the overflow page)."""

ROMAN_NUMERAL_TABLE = """[{"1":["i","I"],"2":["ii","II"],"3":["iii","III"],"4":["iv","IV"],"5":["v","V"],"6":["vi","VI"],"7":["vii","VII"],"8":["viii","VIII"],"9":["ix","IX"],"10":["x","X"],"11":["xi","XI"],"12":["xii","XII"],"13":["xiii","XIII"],"14":["xiv","XIV"],"15":["xv","XV"],"16":["xvi","XVI"],"17":["xvii","XVII"],"18":["xviii","XVIII"],"19":["xix","XIX"],"20":["xx","XX"],"21":["xxi","XXI"],"22":["xxii","XXII"],"23":["xxiii","XXIII"],"24":["xxiv","XXIV"],"25":["xxv","XXV"],"26":["xxvi","XXVI"],"27":["xxvii","XXVII"],"28":["xxviii","XXVIII"],"29":["xxix","XXIX"],"30":["xxx","XXX"],"31":["xxxi","XXXI"],"32":["xxxii","XXXII"],"33":["xxxiii","XXXIII"],"34":["xxxiv","XXXIV"],"35":["xxxv","XXXV"],"36":["xxxvi","XXXVI"],"37":["xxxvii","XXXVII"],"38":["xxxviii","XXXVIII"],"39":["xxxix","XXXIX"],"40":["xl","XL"],"41":["xli","XLI"],"42":["xlii","XLII"],"43":["xliii","XLIII"],"44":["xliv","XLIV"],"45":["xlv","XLV"],"46":["xlvi","XLVI"],"47":["xlvii","XLVII"],"48":["xlviii","XLVIII"],"49":["xlix","XLIX"],"50":["l","L"],"51":["li","LI"],"52":["lii","LII"],"53":["liii","LIII"],"54":["liv","LIV"],"55":["lv","LV"],"56":["lvi","LVI"],"57":["lvii","LVII"],"58":["lviii","LVIII"],"59":["lix","LIX"],"60":["lx","LX"],"61":["lxi","LXI"],"62":["lxii","LXII"],"63":["lxiii","LXIII"],"64":["lxiv","LXIV"],"65":["lxv","LXV"],"66":["lxvi","LXVI"],"67":["lxvii","LXVII"],"68":["lxviii","LXVIII"],"69":["lxix","LXIX"],"70":["lxx","LXX"],"71":["lxxi","LXXI"],"72":["lxxii","LXXII"],"73":["lxxiii","LXXIII"],"74":["lxxiv","LXXIV"],"75":["lxxv","LXXV"],"76":["lxxvi","LXXVI"],"77":["lxxvii","LXXVII"],"78":["lxxviii","LXXVIII"],"79":["lxxix","LXXIX"],"80":["lxxx","LXXX"],"81":["lxxxi","LXXXI"],"82":["lxxxii","LXXXII"],"83":["lxxxiii","LXXXIII"],"84":["lxxxiv","LXXXIV"],"85":["lxxxv","LXXXV"],"86":["lxxxvi","LXXXVI"],"87":["lxxxvii","LXXXVII"],"88":["lxxxviii","LXXXVIII"],"89":["lxxxix","LXXXIX"],"90":["xc","XC"],"91":["xci","XCI"],"92":["xcii","XCII"],"93":["xciii","XCIII"],"94":["xciv","XCIV"],"95":["xcv","XCV"],"96":["xcvi","XCVI"],"97":["xcvii","XCVII"],"98":["xcviii","XCVIII"],"99":["xcix","XCIX"],"100":["c","C"],"101":["ci","CI"],"102":["cii","CII"],"103":["ciii","CIII"],"104":["civ","CIV"],"105":["cv","CV"],"106":["cvi","CVI"],"107":["cvii","CVII"],"108":["cviii","CVIII"],"109":["cix","CIX"],"110":["cx","CX"],"111":["cxi","CXI"],"112":["cxii","CXII"],"113":["cxiii","CXIII"],"114":["cxiv","CXIV"],"115":["cxv","CXV"],"116":["cxvi","CXVI"],"117":["cxvii","CXVII"],"118":["cxviii","CXVIII"],"119":["cxix","CXIX"],"120":["cxx","CXX"],"121":["cxxi","CXXI"],"122":["cxxii","CXXII"],"123":["cxxiii","CXXIII"],"124":["cxxiv","CXXIV"],"125":["cxxv","CXXV"],"126":["cxxvi","CXXVI"],"127":["cxxvii","CXXVII"],"128":["cxxviii","CXXVIII"],"129":["cxxix","CXXIX"],"130":["cxxx","CXXX"],"131":["cxxxi","CXXXI"],"132":["cxxxii","CXXXII"],"133":["cxxxiii","CXXXIII"],"134":["cxxxiv","CXXXIV"],"135":["cxxxv","CXXXV"],"136":["cxxxvi","CXXXVI"],"137":["cxxxvii","CXXXVII"],"138":["cxxxviii","CXXXVIII"],"139":["cxxxix","CXXXIX"],"140":["cxl","CXL"],"141":["cxli","CXLI"],"142":["cxlii","CXLII"],"143":["cxliii","CXLIII"],"144":["cxliv","CXLIV"],"145":["cxlv","CXLV"],"146":["cxlvi","CXLVI"],"147":["cxlvii","CXLVII"],"148":["cxlviii","CXLVIII"],"149":["cxlix","CXLIX"],"150":["cl","CL"],"151":["cli","CLI"],"152":["clii","CLII"],"153":["cliii","CLIII"],"154":["cliv","CLIV"],"155":["clv","CLV"],"156":["clvi","CLVI"],"157":["clvii","CLVII"],"158":["clviii","CLVIII"],"159":["clix","CLIX"],"160":["clx","CLX"],"161":["clxi","CLXI"],"162":["clxii","CLXII"],"163":["clxiii","CLXIII"],"164":["clxiv","CLXIV"],"165":["clxv","CLXV"],"166":["clxvi","CLXVI"],"167":["clxvii","CLXVII"],"168":["clxviii","CLXVIII"],"169":["clxix","CLXIX"],"170":["clxx","CLXX"],"171":["clxxi","CLXXI"],"172":["clxxii","CLXXII"],"173":["clxxiii","CLXXIII"],"174":["clxxiv","CLXXIV"],"175":["clxxv","CLXXV"],"176":["clxxvi","CLXXVI"],"177":["clxxvii","CLXXVII"],"178":["clxxviii","CLXXVIII"],"179":["clxxix","CLXXIX"],"180":["clxxx","CLXXX"],"181":["clxxxi","CLXXXI"],"182":["clxxxii","CLXXXII"],"183":["clxxxiii","CLXXXIII"],"184":["clxxxiv","CLXXXIV"],"185":["clxxxv","CLXXXV"],"186":["clxxxvi","CLXXXVI"],"187":["clxxxvii","CLXXXVII"],"188":["clxxxviii","CLXXXVIII"],"189":["clxxxix","CLXXXIX"],"190":["cxc","CXC"],"191":["cxci","CXCI"],"192":["cxcii","CXCII"],"193":["cxciii","CXCIII"],"194":["cxciv","CXCIV"],"195":["cxcv","CXCV"],"196":["cxcvi","CXCVI"],"197":["cxcvii","CXCVII"],"198":["cxcviii","CXCVIII"],"199":["cxcix","CXCIX"],"200":["cc","CC"]}]"""


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Document structure
    "DOCUMENT_STRUCTURE",
    "DOCUMENT_STRUCTURE_EXTENDED",
    # OCR notes
    "OCR_SOURCE_NOTES",
    "OCR_SOURCE_NOTES_EXTENDED",
    # Tagging
    "TAGGING_PHILOSOPHY",
    "TAGGING_TRIGGERS",
    "TAGGING_GUIDELINES",
    "BOUNDARY_RULES",
    # Chapter detection
    "OVERFLOW_DETECTION",
    "CHAPTER_BOUNDARY_SIGNALS",
    "OCR_ERROR_PATTERNS",
    # Output format
    "OUTPUT_RULES",
    "VERIFICATION_CHECKLIST",
    # Reference tables
    "ROMAN_NUMERAL_NOTES",
    "ROMAN_NUMERAL_TABLE",
]

