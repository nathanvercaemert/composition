"""Entry points for the chapter boundary refinement suite."""

from __future__ import annotations

from .binary_aggregator import (
    detect_boundary_conflicts,
    detect_unassigned_pages,
    merge_chapter_memberships,
    validate_binary_response,
    validate_tagging_response,
    extract_chapter_pages,
)
from .refiner_instructions import (
    build_refinement_system_prompt,
    build_refinement_user_prompt,
    format_refinement_batch_content,
    get_expected_tagging_schema,
)
from .refiner_orchestrator import (
    ChapterRefinementResult,
    RefinementOptions,
    assemble_refined_boundaries,
    main,
    process_book_refinement,
)

__all__ = [
    "RefinementOptions",
    "ChapterRefinementResult",
    "process_book_refinement",
    "assemble_refined_boundaries",
    "main",
    "build_refinement_user_prompt",
    "build_refinement_system_prompt",
    "get_expected_tagging_schema",
    "format_refinement_batch_content",
    "validate_binary_response",
    "validate_tagging_response",
    "extract_chapter_pages",
    "merge_chapter_memberships",
    "detect_unassigned_pages",
    "detect_boundary_conflicts",
]

