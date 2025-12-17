#!/usr/bin/env python3
"""
orchestrator.py

Main execution logic for chapter boundary detection using a multi-model
ensemble. Coordinates batching, model invocation, aggregation, and persistence
of intermediate outputs for observability while selecting a single best
contiguous range per chapter (bridging only one-page gaps).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from instructions import (
    MIN_EXPECTED_TAGS_PER_PAGE,
    BatchContent,
    OCRFormattingOptions,
    approx_token_count,
    build_system_output_prompt,
    build_user_prompt_preamble,
    format_batch_content,
    validate_tagging_density,
)
from model_client import (
    APIKeyError,
    ModelCallError,
    ModelClient,
    ModelClientError,
    ResponseParseError,
)

logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 25
OVERLAP = 5
DEFAULT_DELAY_BETWEEN_BATCHES = 1.5
DEFAULT_AGGREGATION = "union"

# Model identifiers
MODEL_KEY_TO_ID = {
    "qwen": "qwen/qwen3-235b-a22b-instruct-2507",
    "gpt-oss": "openai/gpt-oss-120b",
    "deepseek": "deepseek/deepseek-v3.2-exp",
    "kimi": "moonshotai/kimi-k2-thinking",
}
MODEL_ID_TO_KEY = {v: k for k, v in MODEL_KEY_TO_ID.items()}

PageTagMap = Dict[str, List[str]]


def _extract_page_index(key: str) -> Optional[int]:
    """Extract trailing digits from a page key like page_001 or '1'."""

    import re

    match = re.search(r"(\d+)$", key.strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _build_page_index_map(expected_stems: Sequence[str]) -> Dict[int, str]:
    """Map numeric page indices to canonical stems."""

    mapping: Dict[int, str] = {}
    for stem in expected_stems:
        idx = _extract_page_index(stem)
        if idx is not None and idx not in mapping:
            mapping[idx] = stem
    return mapping


def _normalize_chapter_tags(raw_value: Any) -> List[str]:
    """Coerce various tag shapes (scalar, list) into a clean list of strings."""

    values: List[Any]
    if isinstance(raw_value, list):
        values = raw_value
    elif raw_value is None:
        return []
    else:
        values = [raw_value]

    normalized: List[str] = []
    for val in values:
        if val is None:
            continue
        text = str(val).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


@dataclass
class ProcessingOptions:
    """Runtime options for batch processing."""

    batch_size: int = BATCH_SIZE
    overlap: int = OVERLAP
    delay_between_batches: float = DEFAULT_DELAY_BETWEEN_BATCHES
    min_tags_per_page: float = MIN_EXPECTED_TAGS_PER_PAGE
    aggregation: str = DEFAULT_AGGREGATION
    sequential: bool = True
    head_chars: int = 1500
    tail_chars: int = 1500
    max_chars_per_provider: int = 4000
    max_request_tokens: int | None = None
    max_batch_retries: int = 1


@dataclass
class ModelBatchResponse:
    """Response from a single model for a single batch."""

    model_id: str
    batch_index: int
    page_stems: List[str]
    raw_response: Dict[str, List[str]] = field(default_factory=dict)
    valid_tags: Dict[str, List[str]] = field(default_factory=dict)
    missing_pages: List[str] = field(default_factory=list)
    unexpected_pages: List[str] = field(default_factory=list)
    reasoning_content: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


@dataclass
class AggregatedBatchResponse:
    """Aggregated response from all models for a single batch."""

    batch_index: int
    page_stems: List[str]
    model_responses: List[ModelBatchResponse]
    aggregated_tags: Dict[str, Set[str]]
    agreement_stats: Dict[str, Any]
    estimated_tokens: Optional[int] = None
    tag_votes: Dict[str, Dict[str, int]] = field(default_factory=dict)
    degraded: bool = False


@dataclass
class BookProcessingResult:
    """Complete result of processing a book using single best contiguous chapter ranges."""

    book_name: str
    total_pages: int
    total_batches: int
    batch_results: List[AggregatedBatchResponse]
    final_page_tags: Dict[str, Set[str]]
    final_ranges: Dict[str, List[str]]
    processing_stats: Dict[str, Any]


def natural_key(value: str) -> List[object]:
    """Generate a natural sort key so page_2 < page_10."""
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def validate_model_response(
    response: Any, expected_stems: Sequence[str]
) -> Tuple[PageTagMap, List[str], List[str]]:
    """
    Validate and normalize model response against expected page stems.

    Returns: (valid_tags, missing_pages, unexpected_pages)
    """
    expected_set = {stem.strip(): stem.strip() for stem in expected_stems}
    index_map = _build_page_index_map(expected_stems)
    valid: Dict[str, List[str]] = {}
    unexpected: List[str] = []

    if not isinstance(response, dict):
        logger.warning("Model response is not a dict; received type: %s", type(response))
        return valid, list(expected_set.values()), unexpected

    for raw_page, chapters in response.items():
        raw_page_key = str(raw_page).strip()
        candidate_idx = _extract_page_index(raw_page_key)
        canonical_page = index_map.get(candidate_idx) if candidate_idx is not None else None
        page_key = canonical_page or raw_page_key

        if page_key not in expected_set:
            unexpected.append(raw_page_key)
            continue

        normalized = _normalize_chapter_tags(chapters)
        if normalized:
            valid[page_key] = normalized
        else:
            logger.warning("Skipping page %s: expected list of chapters, got %s", page_key, type(chapters))

    missing = [stem for stem in expected_stems if stem not in valid]
    return valid, missing, unexpected


def fill_empty_tags_in_order(
    pages: Sequence[str],
    tags_by_page: Dict[str, Set[str]],
    fallback: Set[str],
) -> None:
    """Ensure every page has at least one tag by borrowing from neighbors."""

    last_nonempty: Set[str] | None = None
    forward_cache: List[Set[str] | None] = [None] * len(pages)

    for idx, page in enumerate(pages):
        if tags_by_page.get(page):
            last_nonempty = set(tags_by_page[page])
        forward_cache[idx] = last_nonempty

    next_nonempty: Set[str] | None = None
    for idx in range(len(pages) - 1, -1, -1):
        page = pages[idx]
        if tags_by_page.get(page):
            next_nonempty = set(tags_by_page[page])
            continue

        candidate: Set[str] = set()
        if forward_cache[idx]:
            candidate |= forward_cache[idx] or set()
        if next_nonempty:
            candidate |= next_nonempty

        tags_by_page[page] = candidate if candidate else set(fallback)


def _derive_fallback_tags(tags_by_page: Dict[str, Set[str]]) -> Set[str]:
    """Choose a deterministic fallback tag set."""

    numeric_tags = sorted(
        {int(tag) for tags in tags_by_page.values() for tag in tags if str(tag).isdigit()}
    )
    if numeric_tags:
        return {str(numeric_tags[0])}
    return {"1"}


def get_page_stems(book_path: Path) -> List[str]:
    """Enumerate page stems from the raw-images directory."""
    raw_dir = book_path / "raw-images"
    if not raw_dir.exists():
        logger.error("Missing raw-images directory for %s", book_path.name)
        return []

    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
    stems: Set[str] = set()
    for file in raw_dir.iterdir():
        if file.suffix.lower() in image_exts and file.is_file():
            stems.add(file.stem)

    sorted_stems = sorted(stems, key=natural_key)
    if not sorted_stems:
        logger.warning("No page images found in %s", raw_dir)
    return sorted_stems


def invert_page_tags_to_chapter_pages(page_tags: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """
    Convert page-centric tags to chapter-centric page sets.

    Input: {"page_001": {"1", "2"}, "page_002": {"2"}}
    Output: {"1": {"page_001"}, "2": {"page_001", "page_002"}}
    """
    chapter_pages: Dict[str, Set[str]] = {}
    for page, chapters in page_tags.items():
        for chapter in chapters:
            chapter_pages.setdefault(chapter, set()).add(page)
    return chapter_pages


def find_contiguous_ranges(pages: Set[str], all_pages_ordered: List[str]) -> List[List[str]]:
    """
    Given a set of pages and the ordered list of all pages,
    return a list of contiguous page ranges (each range is a list of consecutive pages).
    """
    page_indices = {page: idx for idx, page in enumerate(all_pages_ordered)}
    sorted_pages = sorted(pages, key=lambda p: page_indices.get(p, math.inf))

    if not sorted_pages:
        return []

    ranges: List[List[str]] = []
    current_range: List[str] = [sorted_pages[0]]

    for i in range(1, len(sorted_pages)):
        current_page = sorted_pages[i]
        previous_page = sorted_pages[i - 1]

        current_idx = page_indices[current_page]
        previous_idx = page_indices[previous_page]

        if current_idx == previous_idx + 1:
            current_range.append(current_page)
        else:
            ranges.append(current_range)
            current_range = [current_page]

    ranges.append(current_range)
    return ranges


def _merge_close_ranges(ranges: List[List[str]], all_pages: List[str], max_gap: int = 1) -> List[List[str]]:
    """Merge ranges separated by small gaps, filling the gap pages."""

    if not ranges:
        return []

    page_indices = {page: idx for idx, page in enumerate(all_pages)}
    ranges_sorted = sorted(ranges, key=lambda r: page_indices.get(r[0], math.inf))

    merged_indices: List[Tuple[int, int]] = []
    cur_start = page_indices.get(ranges_sorted[0][0], 0)
    cur_end = page_indices.get(ranges_sorted[0][-1], cur_start)

    for nxt in ranges_sorted[1:]:
        start_idx = page_indices.get(nxt[0], cur_end)
        end_idx = page_indices.get(nxt[-1], start_idx)
        gap = start_idx - cur_end - 1
        if gap <= max_gap:
            cur_end = max(cur_end, end_idx)
        else:
            merged_indices.append((cur_start, cur_end))
            cur_start, cur_end = start_idx, end_idx

    merged_indices.append((cur_start, cur_end))
    return [all_pages[s : e + 1] for s, e in merged_indices if s < len(all_pages) and e < len(all_pages)]


def _range_priority_key(page_indices: Dict[str, int]) -> Callable[[List[str]], Tuple[int, int]]:
    """Return a sort key that favors longer ranges and earlier start positions."""

    def _key(range_pages: List[str]) -> Tuple[int, int]:
        start_idx = page_indices.get(range_pages[0], math.inf)
        return (-len(range_pages), start_idx)

    return _key


def select_best_range(ranges: List[List[str]], chapter: str, all_pages: List[str]) -> List[str]:
    """
    Return the single longest contiguous range for a chapter.

    Ranges separated by exactly one page are merged (gap is filled) to avoid
    artificial fragmentation. Among the merged ranges, only the longest range
    is kept; ties are broken by earliest start position.
    """
    if not ranges:
        return []

    merged = _merge_close_ranges(ranges, all_pages, max_gap=1)
    if not merged:
        return []

    page_indices = {page: idx for idx, page in enumerate(all_pages)}
    best_range = min(merged, key=_range_priority_key(page_indices))

    discarded = max(0, len(merged) - 1)
    if discarded:
        logger.debug("Discarding %s additional range(s) for chapter %s", discarded, chapter)

    return sorted(best_range, key=natural_key)


def compute_chapter_ranges(
    contiguous_ranges: Dict[str, List[List[str]]],
    page_stems: Sequence[str],
) -> Tuple[Dict[str, List[str]], int]:
    """
    Compute chapter ranges using the single-best-range policy.

    Ranges separated by exactly one page are merged (gap is filled). For each
    chapter, only the longest merged range is kept; ties favor the earliest
    range. Returns (final_ranges, discarded_ranges_count).
    """
    final_ranges: Dict[str, List[str]] = {}
    discarded_ranges_count = 0
    ordered_pages = list(page_stems)
    page_indices = {page: idx for idx, page in enumerate(ordered_pages)}
    priority_key = _range_priority_key(page_indices)

    for chapter, ranges in contiguous_ranges.items():
        if not ranges:
            continue

        merged = _merge_close_ranges(ranges, ordered_pages, max_gap=1)
        if not merged:
            continue

        merged_sorted = sorted(merged, key=priority_key)
        best_range = merged_sorted[0]
        discarded_here = max(0, len(merged_sorted) - 1)
        if discarded_here:
            discarded_page_count = sum(len(r) for r in merged_sorted[1:])
            discarded_ranges_count += discarded_here
            logger.info(
                "Chapter %s: keeping longest range (%s pages), discarded %s other range(s) covering %s page(s)",
                chapter,
                len(best_range),
                discarded_here,
                discarded_page_count,
            )

        final_ranges[chapter] = sorted(best_range, key=natural_key)

    return final_ranges, discarded_ranges_count


def compute_total_batches(total_pages: int, batch_size: int = BATCH_SIZE, overlap: int = OVERLAP) -> int:
    """Compute the total number of batches given the page count."""

    if total_pages == 0:
        return 0
    if total_pages <= batch_size:
        return 1
    step = max(1, batch_size - overlap)
    return 1 + math.ceil((total_pages - batch_size) / step)


def compute_max_adjacent_page_sum(final_ranges: Dict[str, List[str]]) -> Tuple[int | None, str | None]:
    """
    Compute the maximum pages for any chapter when combined with its adjacent chapters.
    Only numeric chapters are considered for adjacency.
    """
    numeric_counts = {int(ch): len(pages) for ch, pages in final_ranges.items() if ch.isdigit()}
    if not numeric_counts:
        return None, None

    max_total = -1
    max_center: int | None = None

    for chapter_num in sorted(numeric_counts):
        total = (
            numeric_counts.get(chapter_num - 1, 0)
            + numeric_counts[chapter_num]
            + numeric_counts.get(chapter_num + 1, 0)
        )
        if total > max_total:
            max_total = total
            max_center = chapter_num

    return max_total, str(max_center) if max_center is not None else None


def _model_slug(model_id: str) -> str:
    """Return a filesystem-friendly slug for a model identifier."""
    slug = MODEL_ID_TO_KEY.get(model_id, model_id)
    return slug.replace("/", "_").replace("-", "_")


def initialize_model_clients(selected_keys: Optional[Sequence[str]] = None) -> Dict[str, ModelClient]:
    """
    Initialize ModelClient instances for requested models.

    Args:
        selected_keys: Iterable of model keys ("qwen", "gpt-oss", "deepseek") or None for all.
    """
    keys = set(selected_keys or MODEL_KEY_TO_ID.keys())
    if "all" in keys:
        keys = set(MODEL_KEY_TO_ID.keys())

    clients: Dict[str, ModelClient] = {}
    for key in sorted(keys):
        model_id = MODEL_KEY_TO_ID.get(key)
        if not model_id:
            logger.warning("Unknown model key requested: %s", key)
            continue
        try:
            clients[model_id] = ModelClient(model=model_id)
            logger.info("Initialized model client for %s (%s)", key, model_id)
        except (APIKeyError, ModelClientError) as exc:
            logger.error("Failed to initialize client for %s: %s", model_id, exc)
    return clients


def process_batch_with_model(
    client: ModelClient,
    model_id: str,
    system_prompt: str,
    content: str,
    page_stems: Sequence[str],
    book_name: str | None,
    batch_index: int,
    batch_dir: Path,
) -> ModelBatchResponse:
    """Process a single batch with a single model and persist raw outputs."""
    start = time.time()
    slug = _model_slug(model_id)
    response_path = batch_dir / f"{slug}_response.json"
    reasoning_path = batch_dir / f"{slug}_reasoning.txt"

    raw_response: Dict[str, List[str]] = {}
    valid: Dict[str, List[str]] = {}
    missing: List[str] = []
    unexpected: List[str] = []
    reasoning_content: Optional[str] = None
    error: Optional[str] = None

    try:
        response, reasoning = client.complete_json_with_reasoning(
            system_prompt=system_prompt,
            user_content=content,
        )
        raw_response = response if isinstance(response, dict) else {}
        valid, missing, unexpected = validate_model_response(response, page_stems)
        reasoning_content = reasoning
    except (ModelCallError, ResponseParseError, ModelClientError) as exc:
        error = str(exc)
        logger.error("Model %s failed on batch %s: %s", model_id, batch_index + 1, exc)
    except Exception as exc:  # noqa: BLE001
        error = f"Unexpected error: {exc}"
        logger.error("Unexpected error from model %s on batch %s: %s", model_id, batch_index + 1, exc)

    try:
        response_path.write_text(json.dumps(raw_response, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write response for %s: %s", model_id, exc)

    if reasoning_content:
        try:
            reasoning_path.write_text(reasoning_content, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write reasoning for %s: %s", model_id, exc)

    duration = time.time() - start
    return ModelBatchResponse(
        model_id=model_id,
        batch_index=batch_index,
        page_stems=list(page_stems),
        raw_response=raw_response,
        valid_tags=valid,
        missing_pages=missing,
        unexpected_pages=unexpected,
        reasoning_content=reasoning_content,
        error=error,
        duration_seconds=duration,
    )


def aggregate_model_responses(
    responses: List[ModelBatchResponse],
    page_stems: Sequence[str],
    strategy: str = DEFAULT_AGGREGATION,
) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    """
    Aggregate responses from multiple models into a single page->tags mapping.
    """
    aggregated: Dict[str, Set[str]] = {stem: set() for stem in page_stems}
    active = [resp for resp in responses if not resp.error]

    if strategy not in {"union", "intersection", "majority"}:
        logger.warning("Unknown aggregation strategy %s; defaulting to union", strategy)
        strategy = "union"

    if not active:
        logger.error("No successful model responses to aggregate.")
        return aggregated, {"error": "no_successful_models"}

    if strategy == "union":
        for resp in active:
            for page, tags in resp.valid_tags.items():
                if page in aggregated:
                    aggregated[page].update(tags)
    else:
        for stem in page_stems:
            tag_counts: Dict[str, int] = {}
            for resp in active:
                for tag in resp.valid_tags.get(stem, []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if not tag_counts:
                continue

            model_count = len(active)
            if strategy == "intersection":
                threshold = model_count
            else:  # majority
                threshold = max(1, math.ceil(model_count / 2))

            aggregated[stem].update({tag for tag, count in tag_counts.items() if count >= threshold})

    agreement_stats = compute_agreement_stats(active, page_stems)

    for stem in page_stems:
        if not aggregated[stem]:
            logger.warning("Page %s received no tags after aggregation", stem)

    return aggregated, agreement_stats


def compute_agreement_stats(
    responses: List[ModelBatchResponse],
    page_stems: Sequence[str],
) -> Dict[str, Any]:
    """Compute statistics about how often models agree on tagging."""
    stats: Dict[str, Any] = {
        "full_agreement_pages": [],
        "disagreement_pages": [],
        "per_model_unique_tags": {resp.model_id: 0 for resp in responses},
        "pages_without_tags": [],
    }

    if not responses:
        return stats

    for stem in page_stems:
        model_sets = [set(resp.valid_tags.get(stem, [])) for resp in responses]
        union = set().union(*model_sets) if model_sets else set()
        if len(union) == 0:
            stats["pages_without_tags"].append(stem)
        elif all(s == model_sets[0] for s in model_sets):
            stats["full_agreement_pages"].append(stem)
        else:
            stats["disagreement_pages"].append(stem)

        for idx, resp in enumerate(responses):
            others_union = set().union(*(model_sets[:idx] + model_sets[idx + 1 :])) if len(model_sets) > 1 else set()
            unique_tags = model_sets[idx] - others_union
            stats["per_model_unique_tags"][resp.model_id] += len(unique_tags)

    total_pages = len(page_stems)
    stats["agreement_rate"] = (
        len(stats["full_agreement_pages"]) / total_pages if total_pages else 0
    )
    return stats


def process_batch_all_models(
    clients: Dict[str, ModelClient],
    book_path: Path,
    page_stems: Sequence[str],
    batch_index: int,
    temp_dir: Path,
    book_name: str | None,
    aggregation: str = DEFAULT_AGGREGATION,
    sequential: bool = True,
    *,
    options: ProcessingOptions,
    ocr_cache: Dict[tuple[str, str], str] | None = None,
    attempt: int = 1,
) -> AggregatedBatchResponse:
    """
    Process a single batch through all models and aggregate results.
    """
    batch_dir = temp_dir / "batches" / f"batch_{batch_index + 1:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    ocr_options = OCRFormattingOptions(
        head_chars=options.head_chars,
        tail_chars=options.tail_chars,
        max_chars_per_provider=options.max_chars_per_provider,
    )
    batch_content = format_batch_content(
        book_path,
        page_stems,
        cache=ocr_cache,
        options=ocr_options,
    )
    system_prompt = build_system_output_prompt(page_stems)
    user_preamble = build_user_prompt_preamble(page_stems, book_name=book_name)
    user_content = f"{user_preamble}\n\n{batch_content.text}"
    estimated_tokens = approx_token_count(system_prompt) + approx_token_count(user_content)

    budget_truncated = False
    if options.max_request_tokens and estimated_tokens > options.max_request_tokens:
        budget_truncated = True
        revised_options = OCRFormattingOptions(
            head_chars=max(200, ocr_options.head_chars // 2),
            tail_chars=max(200, ocr_options.tail_chars // 2),
            max_chars_per_provider=ocr_options.max_chars_per_provider,
        )
        batch_content = format_batch_content(
            book_path,
            page_stems,
            cache=ocr_cache,
            options=revised_options,
        )
        user_content = f"{user_preamble}\n\n{batch_content.text}"
        estimated_tokens = approx_token_count(system_prompt) + approx_token_count(user_content)

        if estimated_tokens > options.max_request_tokens:
            logger.warning(
                "Estimated tokens (%s) still exceed max_request_tokens (%s) for batch %s after truncation.",
                estimated_tokens,
                options.max_request_tokens,
                batch_index + 1,
            )

    try:
        (batch_dir / "input_content.txt").write_text(user_content, encoding="utf-8")
        (batch_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        meta = {
            "batch_index": batch_index + 1,
            "attempt": attempt,
            "estimated_tokens": estimated_tokens,
            "max_request_tokens": options.max_request_tokens,
            "truncation": batch_content.truncation_map,
            "sequential": sequential,
            "aggregation": aggregation,
            "budget_truncated": budget_truncated,
        }
        (batch_dir / "batch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write input content for batch %s: %s", batch_index + 1, exc)

    model_responses: List[ModelBatchResponse] = []

    def _call(model_id: str, client: ModelClient) -> ModelBatchResponse:
        logger.info("Starting model %s for batch %s", model_id, batch_index + 1)
        result = process_batch_with_model(
            client=client,
            model_id=model_id,
            system_prompt=system_prompt,
            content=user_content,
            page_stems=page_stems,
            book_name=book_name,
            batch_index=batch_index,
            batch_dir=batch_dir,
        )
        logger.info(
            "Finished model %s for batch %s in %.2fs",
            model_id,
            batch_index + 1,
            result.duration_seconds or 0.0,
        )
        return result

    if sequential or len(clients) <= 1:
        for model_id, client in clients.items():
            model_responses.append(_call(model_id, client))
    else:
        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            futures = {executor.submit(_call, model_id, client): model_id for model_id, client in clients.items()}
            for future in as_completed(futures):
                model_responses.append(future.result())

    aggregated_tags, agreement_stats = aggregate_model_responses(
        model_responses,
        page_stems,
        strategy=aggregation,
    )

    page_tag_votes: Dict[str, Dict[str, int]] = {stem: {} for stem in page_stems}
    for resp in model_responses:
        if resp.error:
            continue
        for page, tags in resp.valid_tags.items():
            votes = page_tag_votes.setdefault(page, {})
            for tag in tags:
                votes[tag] = votes.get(tag, 0) + 1

    degraded = bool(agreement_stats.get("error"))
    if degraded and attempt <= options.max_batch_retries:
        logger.warning(
            "Batch %s had no successful model outputs. Retrying sequentially (attempt %s/%s).",
            batch_index + 1,
            attempt + 1,
            options.max_batch_retries + 1,
        )
        return process_batch_all_models(
            clients=clients,
            book_path=book_path,
            page_stems=page_stems,
            batch_index=batch_index,
            temp_dir=temp_dir,
            book_name=book_name,
            aggregation=aggregation,
            sequential=True,
            options=options,
            ocr_cache=ocr_cache,
            attempt=attempt + 1,
        )

    fallback_tags = _derive_fallback_tags(aggregated_tags)
    fill_empty_tags_in_order(page_stems, aggregated_tags, fallback=fallback_tags)

    aggregated_output = {page: sorted(tags, key=natural_key) for page, tags in aggregated_tags.items()}
    try:
        (batch_dir / "aggregated_response.json").write_text(
            json.dumps(aggregated_output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write aggregated response for batch %s: %s", batch_index + 1, exc)

    return AggregatedBatchResponse(
        batch_index=batch_index,
        page_stems=list(page_stems),
        model_responses=model_responses,
        aggregated_tags=aggregated_tags,
        agreement_stats=agreement_stats,
        estimated_tokens=estimated_tokens,
        tag_votes=page_tag_votes,
        degraded=degraded,
    )


def save_intermediate_data(
    temp_dir: Path,
    page_tags: Dict[str, Set[str]],
    chapter_pages: Dict[str, Set[str]],
    contiguous_ranges: Dict[str, List[List[str]]],
    agreement_stats: Dict[str, Any],
    processing_log: List[Dict[str, Any]],
) -> None:
    """Save intermediate processing data to the temp directory."""
    temp_dir.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, payload: Any) -> None:
        try:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write %s: %s", path, exc)

    _write(temp_dir / "page_tags_raw.json", {k: sorted(list(v), key=natural_key) for k, v in page_tags.items()})
    _write(
        temp_dir / "chapter_pages_inverted.json",
        {k: sorted(list(v), key=natural_key) for k, v in chapter_pages.items()},
    )
    _write(temp_dir / "contiguous_ranges.json", contiguous_ranges)
    _write(temp_dir / "model_agreement_stats.json", agreement_stats)
    _write(temp_dir / "processing_log.json", processing_log)


def process_book(
    book_name: str,
    clients: Dict[str, ModelClient],
    *,
    options: ProcessingOptions,
    force: bool = False,
    debug: bool = False,
    temp_dir: Path = Path("temp"),
) -> Optional[BookProcessingResult]:
    """
    Process a single book using all configured models.

    Produces a single contiguous range per chapter (bridging one-page gaps);
    ranges can still overlap across chapters when their best spans intersect.
    """
    book_path = Path("books") / book_name
    if not book_path.exists():
        logger.error("Book directory %s does not exist", book_path)
        return None

    raw_dir = book_path / "raw-images"
    if not raw_dir.exists():
        logger.error("Missing raw-images for %s; skipping", book_name)
        return None

    for ocr_dir in ["deepseek-ocr", "paddleocr-vl", "qwen-vl"]:
        if not (book_path / ocr_dir).exists():
            logger.warning("OCR directory %s missing for %s", ocr_dir, book_name)

    page_stems = get_page_stems(book_path)
    if not page_stems:
        logger.warning("No pages found for %s; skipping", book_name)
        return None

    if debug:
        logger.info("Debug mode enabled; keeping intermediate artifacts under %s", temp_dir / book_name)

    page_tags: Dict[str, Set[str]] = {}
    ocr_cache: Dict[tuple[str, str], str] = {}
    total_pages = len(page_stems)
    total_batches = compute_total_batches(total_pages, batch_size=options.batch_size, overlap=options.overlap)
    processing_log: List[Dict[str, Any]] = []
    batch_results: List[AggregatedBatchResponse] = []
    book_temp_dir = temp_dir / book_name

    logger.info("Processing %s (%s pages) in %s batches", book_name, total_pages, total_batches)

    start = 0
    batch_index = 0
    while start < total_pages:
        end = min(start + options.batch_size, total_pages)
        batch_stems = page_stems[start:end]
        logger.info("Batch %s/%s: pages %s-%s", batch_index + 1, total_batches, start + 1, end)

        batch_result = process_batch_all_models(
            clients=clients,
            book_path=book_path,
            page_stems=batch_stems,
            batch_index=batch_index,
            temp_dir=book_temp_dir,
            book_name=book_name,
            aggregation=options.aggregation,
            sequential=options.sequential,
            options=options,
            ocr_cache=ocr_cache,
        )

        batch_results.append(batch_result)
        for page, tags in batch_result.aggregated_tags.items():
            page_tags.setdefault(page, set()).update(tags)

        processing_log.append(
            {
                "batch_index": batch_index,
                "page_range": [start + 1, end],
                "degraded": batch_result.degraded,
                "estimated_tokens": batch_result.estimated_tokens,
                "agreement": batch_result.agreement_stats,
                "models": {
                    resp.model_id: {
                        "error": resp.error,
                        "missing_pages": resp.missing_pages,
                        "unexpected_pages": resp.unexpected_pages,
                        "duration_seconds": resp.duration_seconds,
                    }
                    for resp in batch_result.model_responses
                },
            }
        )

        if end == total_pages:
            break
        start += max(1, options.batch_size - options.overlap)
        batch_index += 1
        time.sleep(options.delay_between_batches)

    if not page_tags:
        logger.warning("No tagging output for %s", book_name)
        return None

    fill_empty_tags_in_order(page_stems, page_tags, fallback=_derive_fallback_tags(page_tags))

    density_valid, density_stats = validate_tagging_density(
        {k: sorted(list(v), key=natural_key) for k, v in page_tags.items()},
        min_tags_per_page=options.min_tags_per_page,
    )
    if not density_valid:
        logger.warning(
            "Low tag density for %s (avg %.2f tags/page).",
            book_name,
            density_stats.get("avg_tags_per_page", 0),
        )

    chapter_pages = invert_page_tags_to_chapter_pages(page_tags)
    if not chapter_pages:
        logger.warning("No chapter tags determined for %s", book_name)
        return None

    contiguous_ranges: Dict[str, List[List[str]]] = {}
    for chapter, pages in chapter_pages.items():
        ranges = find_contiguous_ranges(pages, page_stems)
        contiguous_ranges[chapter] = ranges

    final_ranges, discarded_ranges = compute_chapter_ranges(
        contiguous_ranges=contiguous_ranges,
        page_stems=page_stems,
    )

    numeric_chapters = sorted({int(ch) for ch in final_ranges if ch.isdigit()})
    if numeric_chapters:
        missing = [str(ch) for ch in range(numeric_chapters[0], numeric_chapters[-1] + 1) if ch not in numeric_chapters]
        if missing:
            logger.warning("Missing numeric chapters for %s: %s", book_name, ", ".join(missing))

    chapter_lengths = {ch: len(pages) for ch, pages in final_ranges.items()}
    max_single_chapter = max(chapter_lengths.values(), default=0)
    max_adjacent_sum, max_adjacent_center = compute_max_adjacent_page_sum(final_ranges)

    page_assignment_counts: Dict[str, int] = {page: 0 for page in page_stems}
    for chapter, pages in final_ranges.items():
        for page in pages:
            page_assignment_counts[page] = page_assignment_counts.get(page, 0) + 1
    overlap_pages = sum(1 for count in page_assignment_counts.values() if count > 1)
    avg_chapters_per_page_final = (
        sum(page_assignment_counts.values()) / len(page_stems) if page_stems else 0
    )

    agreement_aggregate = {
        "batch_agreements": [br.agreement_stats for br in batch_results],
        "density_stats": density_stats,
        "degraded_batches": [br.batch_index for br in batch_results if br.degraded],
    }

    save_intermediate_data(
        temp_dir=book_temp_dir,
        page_tags=page_tags,
        chapter_pages=chapter_pages,
        contiguous_ranges=contiguous_ranges,
        agreement_stats=agreement_aggregate,
        processing_log=processing_log,
    )

    output_path = book_temp_dir / "chapter_boundaries_unrefined.json"
    if output_path.exists() and not force:
        try:
            reply = input(f"{output_path} exists. Overwrite? [y/N]: ").strip().lower()
        except EOFError:
            reply = "n"
        if reply not in {"y", "yes"}:
            logger.info("Skipping write for %s", book_name)
            return None

    output_payload = {ch: pages for ch, pages in final_ranges.items()}
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote chapter boundaries to %s", output_path)

    processing_stats = {
        "pages": total_pages,
        "batches": len(batch_results),
        "avg_tags_per_page": density_stats.get("avg_tags_per_page"),
        "max_single_chapter_pages": max_single_chapter,
        "max_adjacent_sum": max_adjacent_sum,
        "max_adjacent_center": max_adjacent_center,
        "overlap_pages": overlap_pages,
        "avg_chapters_per_page_final": avg_chapters_per_page_final,
        "discarded_ranges": discarded_ranges,
    }

    return BookProcessingResult(
        book_name=book_name,
        total_pages=total_pages,
        total_batches=total_batches,
        batch_results=batch_results,
        final_page_tags=page_tags,
        final_ranges=final_ranges,
        processing_stats=processing_stats,
    )


def parse_models_arg(models: List[str]) -> List[str]:
    """Expand and normalize model selection arguments."""
    if not models:
        return list(MODEL_KEY_TO_ID.keys())
    if "all" in models:
        return list(MODEL_KEY_TO_ID.keys())
    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Determine chapter boundaries for OCR'd books using multi-model ensemble."
    )
    parser.add_argument("book_name", nargs="?", help="Name of a single book directory under books/")
    parser.add_argument("--all", action="store_true", help="Process all books under books/")
    parser.add_argument("--force", action="store_true", help="Overwrite existing chapter_boundaries_unrefined.json without prompting.")
    parser.add_argument("--debug", action="store_true", help="Save additional debug information to temp/ directory.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["qwen", "gpt-oss", "deepseek", "kimi", "all"],
        default=["all"],
        help="Which models to use (default: all)",
    )
    parser.add_argument(
        "--sequential",
        dest="sequential",
        action="store_true",
        default=True,
        help="Process models sequentially (default)",
    )
    parser.add_argument(
        "--parallel",
        dest="sequential",
        action="store_false",
        help="Process models in parallel",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("temp"),
        help="Directory for intermediate data (default: temp/)",
    )
    parser.add_argument(
        "--aggregation",
        choices=["union", "intersection", "majority"],
        default="union",
        help="Aggregation strategy for multi-model results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Pages per batch (default: 25)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP,
        help="Number of overlapping pages between consecutive batches (default: 5)",
    )
    parser.add_argument(
        "--delay-between-batches",
        type=float,
        default=DEFAULT_DELAY_BETWEEN_BATCHES,
        help="Seconds to wait between batches (default: 1.5)",
    )
    parser.add_argument(
        "--min-tags-per-page",
        type=float,
        default=MIN_EXPECTED_TAGS_PER_PAGE,
        help="Minimum average tags per page before warnings (default: 2)",
    )
    parser.add_argument(
        "--max-request-tokens",
        type=int,
        default=None,
        help="Estimated token ceiling per request; warns when exceeded.",
    )
    parser.add_argument(
        "--head-chars",
        type=int,
        default=1500,
        help="Characters to keep from the start of each OCR provider output",
    )
    parser.add_argument(
        "--tail-chars",
        type=int,
        default=1500,
        help="Characters to keep from the end of each OCR provider output",
    )
    parser.add_argument(
        "--max-chars-per-provider",
        type=int,
        default=4000,
        help="Maximum characters per provider after truncation",
    )
    parser.add_argument(
        "--max-batch-retries",
        type=int,
        default=1,
        help="Retries per batch when all models fail (default: 1)",
    )
    args = parser.parse_args()

    if not args.all and not args.book_name:
        parser.error("Specify a BOOK_NAME or use --all")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    selected_model_keys = parse_models_arg(args.models)
    clients = initialize_model_clients(selected_model_keys)
    if not clients:
        logger.error("No model clients could be initialized. Aborting.")
        return

    options = ProcessingOptions(
        batch_size=args.batch_size,
        overlap=args.overlap,
        delay_between_batches=args.delay_between_batches,
        min_tags_per_page=args.min_tags_per_page,
        aggregation=args.aggregation,
        sequential=args.sequential,
        head_chars=args.head_chars,
        tail_chars=args.tail_chars,
        max_chars_per_provider=args.max_chars_per_provider,
        max_request_tokens=args.max_request_tokens,
        max_batch_retries=args.max_batch_retries,
    )

    if args.all:
        books_dir = Path("books")
        book_dirs = [p.name for p in books_dir.iterdir() if p.is_dir()]
        if not book_dirs:
            logger.error("No book directories found under %s", books_dir)
            return
        for name in sorted(book_dirs, key=natural_key):
            process_book(
                name,
                clients,
                options=options,
                force=args.force,
                debug=args.debug,
                temp_dir=args.temp_dir,
            )
    else:
        process_book(
            args.book_name,
            clients,
            options=options,
            force=args.force,
            debug=args.debug,
            temp_dir=args.temp_dir,
        )


if __name__ == "__main__":
    main()

