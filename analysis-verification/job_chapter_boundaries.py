"""
Comprehensive analysis tool for comparing Job chapter boundary predictions
against ground truth ranges.

The script is intentionally verbose and self-documenting so that reviewers can
understand not only the numerical outcomes but also how to interpret them. It
computes coverage, boundary margins, size comparisons, overlap behavior,
contiguity, ordering, and aggregate precision/recall style metrics.

Usage (from repository root):
    python analysis-verification/job_chapter_boundaries.py

Optional overrides:
    python analysis-verification/job_chapter_boundaries.py \
        --ground-truth path/to/gt.json \
        --prediction path/to/pred.json \
        --unrefined   # use temp/Job/chapter_boundaries_unrefined.json
        --refined     # use temp/Job/chapter_boundaries_refined.json

Exit codes:
    0 on success
    non-zero if input files are missing or cannot be parsed
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re

DEFAULT_GROUND_TRUTH = Path("analysis-verification/job_chapter_boundaries.json")
DEFAULT_PREDICTION = Path("books/Job/chapter_boundaries.json")
DEFAULT_UNREFINED_PREDICTION = Path("temp/Job/chapter_boundaries_unrefined.json")
DEFAULT_REFINED_PREDICTION = Path("temp/Job/chapter_boundaries_refined.json")


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def natural_key(text: str) -> Tuple:
    """
    Convert a string into a tuple that enables natural sorting.

    Example:
        "page_9"  -> ("page_", 9, "")
        "page_10" -> ("page_", 10, "")
    """
    return tuple(int(tok) if tok.isdigit() else tok.lower()
                 for tok in re.split(r"(\d+)", str(text)))


def natural_sort(items: Iterable[str]) -> List[str]:
    """Return a new list sorted using natural ordering semantics."""
    return sorted(items, key=natural_key)


def trailing_number(text: str) -> Optional[int]:
    """Return the final integer found in the string, or None if absent."""
    match = re.search(r"(\d+)(?!.*\d)", str(text))
    return int(match.group(1)) if match else None


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Remove duplicates while preserving first occurrence order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def safe_mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def safe_stdev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percent(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Return Pearson correlation or None if undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    denom = denom_x * denom_y
    if denom == 0:
        return None
    return num / denom


def format_float(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def format_bool(value: Optional[bool]) -> str:
    if value is None:
        return "N/A"
    return "YES" if value else "NO"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class GroundTruthRange:
    first_stem: str
    last_stem: str


@dataclass
class RangeExpansion:
    pages: List[str]
    start_idx: Optional[int]
    end_idx: Optional[int]
    start_used: str
    end_used: str
    notes: List[str]


@dataclass
class ChapterMetrics:
    chapter_id: str
    gt_range: GroundTruthRange
    gt_pages: List[str]
    pred_pages: List[str]
    coverage: float
    missing_pages: List[str]
    extra_pages: List[str]
    start_margin: Optional[int]
    end_margin: Optional[int]
    start_on_or_before: Optional[bool]
    end_on_or_after: Optional[bool]
    precision: float
    recall: float
    f1: float
    jaccard: float
    contiguity_gaps: List[int]
    contiguity_is_contiguous: bool
    range_notes: List[str]


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #

def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON from {path}: {exc}") from exc


def find_closest_page_index(stem: str, sorted_pages: List[str]) -> int:
    """
    Choose the closest available page based on trailing number distance.
    Falls back to the first page when numeric distance cannot be computed.
    """
    if not sorted_pages:
        return 0
    target_num = trailing_number(stem)
    best_idx = 0
    best_diff = math.inf
    for idx, page in enumerate(sorted_pages):
        page_num = trailing_number(page)
        if target_num is None or page_num is None:
            # If we cannot compute a numeric distance, rely on ordering only.
            if idx == 0:
                best_idx = 0
                best_diff = 0
            continue
        diff = abs(page_num - target_num)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
    return best_idx


def expand_ground_truth_range(
    gt: GroundTruthRange,
    sorted_pred_pages: List[str],
    page_to_index: Dict[str, int],
) -> RangeExpansion:
    notes: List[str] = []
    if not sorted_pred_pages:
        return RangeExpansion([], None, None, gt.first_stem, gt.last_stem, ["No predicted pages available to expand ground truth range."])

    start_stem = gt.first_stem
    end_stem = gt.last_stem
    start_idx = page_to_index.get(start_stem)
    end_idx = page_to_index.get(end_stem)

    if start_idx is None:
        start_idx = find_closest_page_index(start_stem, sorted_pred_pages)
        start_stem = sorted_pred_pages[start_idx]
        notes.append(
            f"Ground truth start '{gt.first_stem}' not found in predictions; "
            f"using closest available page '{start_stem}' for range expansion."
        )

    if end_idx is None:
        end_idx = find_closest_page_index(end_stem, sorted_pred_pages)
        end_stem = sorted_pred_pages[end_idx]
        notes.append(
            f"Ground truth end '{gt.last_stem}' not found in predictions; "
            f"using closest available page '{end_stem}' for range expansion."
        )

    # If start and end are reversed, swap to keep the range usable.
    if start_idx is not None and end_idx is not None and start_idx > end_idx:
        notes.append(
            "Ground truth range appears reversed relative to predicted ordering; "
            "swapping start/end to compute coverage."
        )
        start_idx, end_idx = end_idx, start_idx
        start_stem, end_stem = end_stem, start_stem

    pages = sorted_pred_pages[start_idx : end_idx + 1] if start_idx is not None and end_idx is not None else []
    return RangeExpansion(
        pages=pages,
        start_idx=start_idx,
        end_idx=end_idx,
        start_used=start_stem,
        end_used=end_stem,
        notes=notes,
    )


def compute_contiguity(pred_pages: List[str], page_to_index: Dict[str, int]) -> Tuple[bool, List[int]]:
    """
    Determine whether predicted pages are contiguous relative to global ordering.
    Returns (is_contiguous, gap_sizes).
    """
    if len(pred_pages) <= 1:
        return True, []
    indices = sorted(page_to_index.get(p, -1) for p in pred_pages if p in page_to_index)
    gaps = []
    for prev, nxt in zip(indices, indices[1:]):
        gap = nxt - prev - 1
        if gap > 0:
            gaps.append(gap)
    return len(gaps) == 0, gaps


def analyze_chapter(
    chapter_id: str,
    gt_range: GroundTruthRange,
    gt_expanded: RangeExpansion,
    pred_pages_raw: Optional[Iterable[str]],
    page_to_index: Dict[str, int],
) -> ChapterMetrics:
    pred_pages = dedupe_preserve_order(natural_sort(pred_pages_raw or []))

    gt_pages = gt_expanded.pages
    gt_set = set(gt_pages)
    pred_set = set(pred_pages)
    intersection = gt_set & pred_set
    union = gt_set | pred_set

    gt_count = len(gt_pages)
    pred_count = len(pred_pages)
    inter_count = len(intersection)

    recall = inter_count / gt_count if gt_count else (1.0 if pred_count == 0 else 0.0)
    precision = inter_count / pred_count if pred_count else (1.0 if gt_count == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    jaccard = inter_count / len(union) if union else 1.0

    missing_pages = [p for p in gt_pages if p not in pred_set]
    extra_pages = [p for p in pred_pages if p not in gt_set]

    start_margin = None
    end_margin = None
    start_on_or_before = None
    end_on_or_after = None

    if gt_expanded.start_idx is not None and pred_pages:
        pred_start_idx = page_to_index[pred_pages[0]]
        start_margin = gt_expanded.start_idx - pred_start_idx
        start_on_or_before = pred_start_idx <= gt_expanded.start_idx

    if gt_expanded.end_idx is not None and pred_pages:
        pred_end_idx = page_to_index[pred_pages[-1]]
        end_margin = pred_end_idx - gt_expanded.end_idx
        end_on_or_after = pred_end_idx >= gt_expanded.end_idx

    is_contiguous, gap_sizes = compute_contiguity(pred_pages, page_to_index)

    coverage = inter_count / gt_count if gt_count else 0.0

    return ChapterMetrics(
        chapter_id=chapter_id,
        gt_range=gt_range,
        gt_pages=gt_pages,
        pred_pages=pred_pages,
        coverage=coverage,
        missing_pages=missing_pages,
        extra_pages=extra_pages,
        start_margin=start_margin,
        end_margin=end_margin,
        start_on_or_before=start_on_or_before,
        end_on_or_after=end_on_or_after,
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
        contiguity_gaps=gap_sizes,
        contiguity_is_contiguous=is_contiguous,
        range_notes=gt_expanded.notes,
    )


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #

def separator(char: str = "=", width: int = 80) -> str:
    return char * width


def print_header(gt_path: Path, pred_path: Path) -> None:
    print(separator("="))
    print("CHAPTER BOUNDARY ANALYSIS: Ground Truth vs Prediction")
    print(separator("="))
    print("\nFILES ANALYZED:")
    print(f"  Ground Truth: {gt_path}")
    print(f"  Prediction:   {pred_path}")
    print(separator("-"))


def print_chapter_coverage(chapter_metrics: List[ChapterMetrics]) -> None:
    print("SECTION 1: GROUND TRUTH PAGE COVERAGE (Per-Chapter)")
    print(separator("-"))
    print("This section checks whether predicted chapter membership captures every")
    print("page defined by the ground truth contiguous ranges. It flags missing")
    print("pages, extra pages, and how boundaries align.")
    for cm in chapter_metrics:
        print(f"\nCHAPTER {cm.chapter_id}:")
        gt_count = len(cm.gt_pages)
        pred_count = len(cm.pred_pages)
        print(f"  Ground Truth Range: {cm.gt_range.first_stem} -> {cm.gt_range.last_stem}")
        if cm.range_notes:
            for note in cm.range_notes:
                print(f"  NOTE: {note}")
        print(f"  Ground Truth Page Count (expanded): {gt_count}")
        print(f"  Predicted Pages in Chapter: {pred_count}")
        print(f"  Coverage: {len(cm.gt_pages) - len(cm.missing_pages)}/{gt_count} ({format_float(cm.coverage * 100)}%) of ground truth pages are present")
        if cm.coverage == 1.0:
            print("  Interpretation: FULL COVERAGE - every ground truth page appears in the prediction.")
        elif cm.coverage == 0.0:
            print("  Interpretation: NO COVERAGE - none of the ground truth pages are captured.")
        else:
            print("  Interpretation: PARTIAL COVERAGE - some ground truth pages are missing.")

        if cm.missing_pages:
            print("  Missing pages (ground truth not found in prediction):")
            for p in cm.missing_pages:
                print(f"    - {p}")
        else:
            print("  Missing pages: None (ideal).")

        if cm.extra_pages:
            print("  Extra pages (predicted but outside ground truth range):")
            for p in cm.extra_pages:
                print(f"    - {p}")
        else:
            print("  Extra pages: None (tight boundaries).")

        # Boundary interpretation
        if cm.start_margin is not None:
            if cm.start_margin > 0:
                desc = f"{cm.start_margin} pages BEFORE ground truth start (prediction starts early)."
            elif cm.start_margin < 0:
                desc = f"{abs(cm.start_margin)} pages AFTER ground truth start (prediction starts late)."
            else:
                desc = "Exact match - prediction starts exactly at the ground truth boundary."
            print(f"  Start boundary: {desc}")
        else:
            print("  Start boundary: Not available (no predicted pages).")

        if cm.end_margin is not None:
            if cm.end_margin > 0:
                desc = f"{cm.end_margin} pages AFTER ground truth end (prediction ends late/extends)."
            elif cm.end_margin < 0:
                desc = f"{abs(cm.end_margin)} pages BEFORE ground truth end (prediction ends early)."
            else:
                desc = "Exact match - prediction ends exactly at the ground truth boundary."
            print(f"  End boundary:   {desc}")
        else:
            print("  End boundary:   Not available (no predicted pages).")

        print(
            "  Boundary containment flags: "
            f"starts_on_or_before_gt={format_bool(cm.start_on_or_before)}, "
            f"ends_on_or_after_gt={format_bool(cm.end_on_or_after)}"
        )


def summarize_margin_distribution(margins: List[int]) -> Dict[str, float]:
    if not margins:
        return {
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "min": 0,
            "max": 0,
            "hist": Counter(),
        }
    hist = Counter(margins)
    return {
        "mean": safe_mean(margins),
        "median": safe_median(margins),
        "stdev": safe_stdev(margins),
        "min": min(margins),
        "max": max(margins),
        "hist": hist,
    }


def print_margin_distribution(
    title: str,
    margins: List[int],
    exact_count: int,
    total_count: int,
) -> None:
    stats = summarize_margin_distribution(margins)
    print(f"{title} BOUNDARY MARGINS:")
    print("  (Positive = prediction extends past ground truth; Negative = prediction falls short)")
    print(f"  Mean:   {format_float(stats['mean'])} pages")
    print(f"  Median: {format_float(stats['median'])} pages")
    print(f"  StdDev: {format_float(stats['stdev'])} pages")
    print(f"  Min:    {stats['min']} pages")
    print(f"  Max:    {stats['max']} pages")
    print("  Histogram (margin: count [percent]):")
    hist: Counter = stats["hist"]
    if hist:
        for margin in sorted(hist.keys()):
            count = hist[margin]
            pct = percent(count, total_count)
            print(f"    {margin:+d}: {count} chapters ({format_float(pct)}%)")
    else:
        print("    No margins available (likely no predictions).")
    pct_exact = percent(exact_count, total_count) if total_count else 0.0
    print(f"  Exact matches (margin = 0): {exact_count}/{total_count} chapters ({format_float(pct_exact)}%)")
    print("")


def print_size_comparison(
    gt_sizes: Dict[str, int],
    pred_sizes: Dict[str, int],
) -> None:
    print("SECTION 3: CHAPTER SIZE COMPARISON")
    print(separator("-"))

    def describe_extrema(sizes: Dict[str, int], label: str) -> Tuple[Optional[str], int]:
        if not sizes:
            return None, 0
        chapter = max(sizes.items(), key=lambda kv: kv[1]) if label == "largest" else min(sizes.items(), key=lambda kv: kv[1])
        return chapter[0], chapter[1]

    gt_largest, gt_largest_size = describe_extrema(gt_sizes, "largest")
    pred_largest, pred_largest_size = describe_extrema(pred_sizes, "largest")
    gt_smallest, gt_smallest_size = describe_extrema(gt_sizes, "smallest")
    pred_smallest, pred_smallest_size = describe_extrema(pred_sizes, "smallest")

    print("Largest Chapter Analysis:")
    if gt_largest is None or pred_largest is None:
        print("  Not enough data to evaluate largest chapters.")
    else:
        print(f"  Ground Truth largest: Chapter {gt_largest} ({gt_largest_size} pages)")
        print(f"  Prediction largest:   Chapter {pred_largest} ({pred_largest_size} pages)")
        diff = pred_largest_size - gt_largest_size
        diff_pct = percent(abs(diff), gt_largest_size) if gt_largest_size else 0
        same = gt_largest == pred_largest
        print(f"  Are they the same chapter? {'YES' if same else 'NO'}")
        print(f"  Size difference: {diff:+d} pages ({format_float(diff_pct)}%)")

    print("\nSmallest Chapter Analysis:")
    if gt_smallest is None or pred_smallest is None:
        print("  Not enough data to evaluate smallest chapters.")
    else:
        print(f"  Ground Truth smallest: Chapter {gt_smallest} ({gt_smallest_size} pages)")
        print(f"  Prediction smallest:   Chapter {pred_smallest} ({pred_smallest_size} pages)")
        diff = pred_smallest_size - gt_smallest_size
        diff_pct = percent(abs(diff), gt_smallest_size) if gt_smallest_size else 0
        same = gt_smallest == pred_smallest
        print(f"  Are they the same chapter? {'YES' if same else 'NO'}")
        print(f"  Size difference: {diff:+d} pages ({format_float(diff_pct)}%)")

    gt_values = list(gt_sizes.values())
    pred_values = [pred_sizes.get(ch, 0) for ch in gt_sizes.keys()]
    avg_gt = safe_mean(gt_values)
    avg_pred = safe_mean(list(pred_sizes.values()))
    corr = pearson_correlation(gt_values, pred_values)
    print("\nOverall Size Statistics:")
    print(f"  Average chapter size (GT):   {format_float(avg_gt)} pages")
    print(f"  Average chapter size (Pred): {format_float(avg_pred)} pages")
    print(f"  Size correlation (Pearson):  {format_float(corr) if corr is not None else 'N/A'} (ideal is 1.0)")

    print("\nChapters differing by more than 2 pages:")
    for ch, gt_size in gt_sizes.items():
        pred_size = pred_sizes.get(ch)
        if pred_size is None:
            print(f"  Chapter {ch}: missing in prediction.")
            continue
        diff = abs(pred_size - gt_size)
        if diff > 2:
            print(f"  Chapter {ch}: GT={gt_size} vs Pred={pred_size} (diff {diff} pages)")
    print("")


def print_missing_extra_chapters(gt_chapters: set, pred_chapters: set) -> None:
    print("SECTION 4: MISSING AND EXTRA CHAPTER DETECTION")
    print(separator("-"))
    missing = sorted(gt_chapters - pred_chapters, key=natural_key)
    extra = sorted(pred_chapters - gt_chapters, key=natural_key)
    if missing:
        print("Chapters present in ground truth but missing in prediction:")
        for ch in missing:
            print(f"  - Chapter {ch}")
    else:
        print("No ground truth chapters are missing from prediction. (Ideal)")

    if extra:
        print("\nChapters present in prediction but not in ground truth:")
        for ch in extra:
            print(f"  - Chapter {ch}")
    else:
        print("\nNo extra chapters present in prediction. (Ideal)")
    print("")


def print_overlap_analysis(
    gt_page_to_chapters: Dict[str, List[str]],
    pred_page_to_chapters: Dict[str, List[str]],
) -> None:
    print("SECTION 5: OVERLAP ANALYSIS (Multi-label Behavior)")
    print(separator("-"))
    gt_overlaps = {p: chs for p, chs in gt_page_to_chapters.items() if len(chs) > 1}
    pred_overlaps = {p: chs for p, chs in pred_page_to_chapters.items() if len(chs) > 1}
    gt_total_unique = len(gt_page_to_chapters)
    pred_total_unique = len(pred_page_to_chapters)
    gt_overlap_pct = percent(len(gt_overlaps), gt_total_unique) if gt_total_unique else 0.0
    pred_overlap_pct = percent(len(pred_overlaps), pred_total_unique) if pred_total_unique else 0.0
    print(f"Ground truth overlapping pages (pages belonging to multiple chapters): {len(gt_overlaps)} ({format_float(gt_overlap_pct)}% of unique GT pages)")
    print(f"Prediction overlapping pages: {len(pred_overlaps)} ({format_float(pred_overlap_pct)}% of unique predicted pages)")
    if gt_overlaps:
        print("  Ground truth expects overlaps where chapter boundaries meet (often last=first).")
    else:
        print("  Ground truth defines strictly disjoint ranges except boundary transitions.")
    if pred_overlaps:
        print("  Prediction assigns some pages to multiple chapters; review if patterns align with GT overlaps.")
    else:
        print("  Prediction assigns pages to exactly one chapter each (no overlaps).")
    diff = pred_overlap_pct - gt_overlap_pct
    if diff > 0:
        print(f"  Prediction shows more overlapping behavior than ground truth by {format_float(diff)} percentage points.")
    elif diff < 0:
        print(f"  Prediction shows less overlapping behavior than ground truth by {format_float(abs(diff))} percentage points.")
    else:
        print("  Overlap rate matches ground truth exactly.")
    print("")


def print_contiguity_and_order(
    chapter_metrics: List[ChapterMetrics],
    page_to_index: Dict[str, int],
) -> None:
    print("SECTION 6: CONTIGUITY AND ORDERING CHECKS")
    print(separator("-"))
    for cm in chapter_metrics:
        if cm.contiguity_is_contiguous:
            print(f"Chapter {cm.chapter_id}: Pages are contiguous (no gaps).")
        else:
            gap_sizes = cm.contiguity_gaps
            largest_gap = max(gap_sizes) if gap_sizes else 0
            total_gap = sum(gap_sizes)
            print(
                f"Chapter {cm.chapter_id}: Non-contiguous with {len(gap_sizes)} gap(s). "
                f"Gap sizes: {gap_sizes} (largest gap {largest_gap} pages; total missing {total_gap} pages)"
            )
    print("")
    # Order verification
    print("Chapter ordering verification (prediction should progress with chapter numbers):")
    order_issues = []
    sorted_chapters = sorted(
        [(int(ch), ch) for ch in [cm.chapter_id for cm in chapter_metrics] if str(ch).isdigit()],
        key=lambda t: t[0],
    )
    chapter_to_min_idx = {}
    for _, ch in sorted_chapters:
        cm = next((c for c in chapter_metrics if c.chapter_id == ch), None)
        if cm and cm.pred_pages:
            chapter_to_min_idx[ch] = page_to_index.get(cm.pred_pages[0], 0)
    prev_ch = None
    prev_idx = None
    for _, ch in sorted_chapters:
        idx = chapter_to_min_idx.get(ch)
        if prev_idx is not None and idx is not None and idx < prev_idx:
            order_issues.append((prev_ch, ch))
        prev_ch, prev_idx = ch, idx
    if order_issues:
        print("Ordering issues detected (later chapters start before earlier ones):")
        for earlier, later in order_issues:
            print(f"  Chapter {later} starts before Chapter {earlier}")
    else:
        print("No ordering violations detected in predicted chapter starts.")
    print("")


def print_aggregate_metrics(
    chapter_metrics: List[ChapterMetrics],
    gt_total_pages: int,
    pred_total_pages_all: int,
    overall_recall: float,
    overall_precision: float,
    gt_chapter_count: int,
    pred_chapter_count: int,
) -> None:
    print("SECTION 7: AGGREGATE METRICS SUMMARY")
    print(separator("-"))
    f1_macro = safe_mean([cm.f1 for cm in chapter_metrics]) if chapter_metrics else 0.0
    mean_jaccard = safe_mean([cm.jaccard for cm in chapter_metrics]) if chapter_metrics else 0.0
    extra_page_counts = [len(cm.extra_pages) for cm in chapter_metrics]
    total_extra_pages = sum(extra_page_counts)
    avg_extra_pages = safe_mean(extra_page_counts)
    min_extra_pages = min(extra_page_counts) if extra_page_counts else 0
    max_extra_pages = max(extra_page_counts) if extra_page_counts else 0
    median_extra_pages = safe_median(extra_page_counts)
    chapters_with_max_extra = (
        [cm.chapter_id for cm in chapter_metrics if len(cm.extra_pages) == max_extra_pages]
        if max_extra_pages > 0
        else []
    )

    print(f"Total ground truth pages (counting overlaps): {gt_total_pages}")
    print(f"Total predicted pages (counting overlaps):  {pred_total_pages_all}")
    print(f"Overall recall:    {format_float(overall_recall * 100)}% of ground truth pages appear in the correct predicted chapter (ideal: 100%)")
    print(f"Overall precision: {format_float(overall_precision * 100)}% of predicted pages fall inside the ground truth range (ideal: 100%)")
    print(f"Macro F1 (per-chapter average): {format_float(f1_macro)} (ideal: 1.0)")
    print(f"Mean Jaccard similarity:        {format_float(mean_jaccard)} (intersection / union, ideal: 1.0)")
    print("")

    print("Per-chapter precision/recall/F1/Jaccard:")
    for cm in chapter_metrics:
        print(
            f"  Chapter {cm.chapter_id}: "
            f"P={format_float(cm.precision)}, "
            f"R={format_float(cm.recall)}, "
            f"F1={format_float(cm.f1)}, "
            f"Jaccard={format_float(cm.jaccard)}"
        )
    print("")

    print("Extra pages statistics (predicted pages outside ground truth range):")
    print(f"  Total extra pages:   {total_extra_pages}")
    print(f"  Average per chapter: {format_float(avg_extra_pages)}")
    print(f"  Minimum:             {min_extra_pages}")
    print(f"  Maximum:             {max_extra_pages}")
    print(f"  Median:              {format_float(median_extra_pages)}")
    if chapters_with_max_extra:
        chapter_list = ", ".join(natural_sort(chapters_with_max_extra))
        print(f"  Chapter(s) with most extra pages: {chapter_list} ({max_extra_pages} extra pages)")
    print("")

    # Summary table for quick scan
    print("SUMMARY TABLE:")
    print("─────────────────────────────────────────────────────────────────")
    print("Metric                              │ Value    │ Ideal  │ Rating")
    print("─────────────────────────────────────────────────────────────────")
    total_chapters_gt = gt_chapter_count
    total_chapters_pred = pred_chapter_count
    print(f"Total Chapters (GT / Pred)          │ {total_chapters_gt} / {total_chapters_pred}  │ Equal  │ {'PASS' if total_chapters_gt == total_chapters_pred else 'CHECK'}")
    print(f"Overall Coverage                    │ {format_float(overall_recall * 100)}%    │ 100%   │ {'GOOD' if overall_recall > 0.95 else 'FAIR' if overall_recall > 0.8 else 'POOR'}")
    start_margins = [cm.start_margin for cm in chapter_metrics if cm.start_margin is not None]
    end_margins = [cm.end_margin for cm in chapter_metrics if cm.end_margin is not None]
    start_exact = sum(1 for m in start_margins if m == 0)
    end_exact = sum(1 for m in end_margins if m == 0)
    if start_margins:
        start_ratio = start_exact / len(start_margins)
        start_rating = "GOOD" if start_exact == len(start_margins) else "FAIR" if start_ratio > 0.5 else "POOR"
    else:
        start_rating = "N/A"
    if end_margins:
        end_ratio = end_exact / len(end_margins)
        end_rating = "GOOD" if end_exact == len(end_margins) else "FAIR" if end_ratio > 0.5 else "POOR"
    else:
        end_rating = "N/A"
    print(f"Exact Start Boundary Matches        │ {format_float(percent(start_exact, len(start_margins)))}%    │ 100%   │ {start_rating}")
    print(f"Exact End Boundary Matches          │ {format_float(percent(end_exact, len(end_margins)))}%    │ 100%   │ {end_rating}")
    print(f"Mean Jaccard Similarity             │ {format_float(mean_jaccard)}     │ 1.0    │ {'GOOD' if mean_jaccard > 0.9 else 'FAIR' if mean_jaccard > 0.75 else 'POOR'}")
    if avg_extra_pages == 0:
        extra_pages_rating = "GOOD"
    elif avg_extra_pages < 2:
        extra_pages_rating = "FAIR"
    else:
        extra_pages_rating = "POOR"
    print(f"Extra Pages per Chapter (Avg)       │ {format_float(avg_extra_pages)}     │ 0      │ {extra_pages_rating}")
    full_coverage_count = sum(1 for cm in chapter_metrics if cm.coverage == 1.0)
    print(f"Chapters with Full Coverage         │ {full_coverage_count}/{total_chapters_gt}    │ {total_chapters_gt}/{total_chapters_gt}  │ {'GOOD' if full_coverage_count==total_chapters_gt else 'FAIR'}")
    print("─────────────────────────────────────────────────────────────────")
    print("")


def run_analysis(gt_path: Path, pred_path: Path) -> int:
    try:
        gt_raw = load_json(gt_path)
        pred_raw = load_json(pred_path)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    # Prepare chapter lists
    gt_chapters = sorted(gt_raw.keys(), key=natural_key)
    pred_chapters = sorted(pred_raw.keys(), key=natural_key)

    # Collect all predicted pages to establish ordering.
    all_pred_pages = natural_sort(
        dedupe_preserve_order(page for pages in pred_raw.values() for page in pages)
    )
    page_to_index = {p: idx for idx, p in enumerate(all_pred_pages)}

    # Build ground truth ranges
    gt_ranges = {
        ch: GroundTruthRange(first_stem=val["first_stem"], last_stem=val["last_stem"])
        for ch, val in gt_raw.items()
    }

    chapter_metrics: List[ChapterMetrics] = []
    gt_page_to_chapters: Dict[str, List[str]] = defaultdict(list)
    pred_page_to_chapters: Dict[str, List[str]] = defaultdict(list)
    gt_sizes: Dict[str, int] = {}
    pred_sizes: Dict[str, int] = {}

    for ch in gt_chapters:
        gt_range = gt_ranges[ch]
        expansion = expand_ground_truth_range(gt_range, all_pred_pages, page_to_index)
        pred_pages = pred_raw.get(ch, [])
        cm = analyze_chapter(ch, gt_range, expansion, pred_pages, page_to_index)
        chapter_metrics.append(cm)
        gt_sizes[ch] = len(cm.gt_pages)
        pred_sizes[ch] = len(cm.pred_pages)
        for p in cm.gt_pages:
            gt_page_to_chapters[p].append(ch)
        for p in cm.pred_pages:
            pred_page_to_chapters[p].append(ch)

    # Include predicted chapters that do not exist in ground truth for completeness.
    for ch in pred_chapters:
        normalized = dedupe_preserve_order(natural_sort(pred_raw[ch]))
        if ch not in pred_sizes:
            pred_sizes[ch] = len(normalized)
        if ch not in set(gt_chapters):
            for p in normalized:
                pred_page_to_chapters[p].append(ch)

    # Derived metrics for boundary distribution
    start_margins = [cm.start_margin for cm in chapter_metrics if cm.start_margin is not None]
    end_margins = [cm.end_margin for cm in chapter_metrics if cm.end_margin is not None]
    start_exact = sum(1 for m in start_margins if m == 0)
    end_exact = sum(1 for m in end_margins if m == 0)
    fully_contained = sum(
        1
        for cm in chapter_metrics
        if cm.start_margin is not None
        and cm.end_margin is not None
        and cm.start_margin <= 0
        and cm.end_margin <= 0
    )
    fully_covering = sum(
        1
        for cm in chapter_metrics
        if cm.start_margin is not None
        and cm.end_margin is not None
        and cm.start_margin >= 0
        and cm.end_margin >= 0
    )

    # Print report
    print_header(gt_path, pred_path)
    print_chapter_coverage(chapter_metrics)
    print(separator("-"))
    print("SECTION 2: BOUNDARY MARGIN DISTRIBUTION")
    print(separator("-"))
    print_margin_distribution("START", start_margins, start_exact, len(start_margins))
    print_margin_distribution("END", end_margins, end_exact, len(end_margins))
    total_chapters = len(chapter_metrics)
    print(
        f"Chapters fully contained within ground truth range (prediction entirely inside GT): "
        f"{fully_contained}/{total_chapters} ({format_float(percent(fully_contained, total_chapters))}%)"
    )
    print(
        f"Chapters fully containing the ground truth range (prediction extends to cover GT): "
        f"{fully_covering}/{total_chapters} ({format_float(percent(fully_covering, total_chapters))}%)\n"
    )

    print_size_comparison(gt_sizes, pred_sizes)
    print_missing_extra_chapters(set(gt_chapters), set(pred_chapters))
    print_overlap_analysis(gt_page_to_chapters, pred_page_to_chapters)
    print_contiguity_and_order(chapter_metrics, page_to_index)
    gt_total_pages = sum(len(cm.gt_pages) for cm in chapter_metrics)
    total_intersection = sum(len(set(cm.gt_pages) & set(cm.pred_pages)) for cm in chapter_metrics)
    total_pred_pages_all = sum(
        len(dedupe_preserve_order(natural_sort(pages))) for pages in pred_raw.values()
    )
    overall_recall = total_intersection / gt_total_pages if gt_total_pages else 0.0
    overall_precision = total_intersection / total_pred_pages_all if total_pred_pages_all else 0.0
    print_aggregate_metrics(
        chapter_metrics,
        gt_total_pages=gt_total_pages,
        pred_total_pages_all=total_pred_pages_all,
        overall_recall=overall_recall,
        overall_precision=overall_precision,
        gt_chapter_count=len(gt_chapters),
        pred_chapter_count=len(pred_chapters),
    )
    print(separator("="))
    print("ANALYSIS COMPLETE")
    print(separator("="))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Job chapter boundary predictions against ground truth.")
    parser.add_argument("--ground-truth", dest="ground_truth", type=Path, default=DEFAULT_GROUND_TRUTH, help="Path to ground truth JSON file.")
    parser.add_argument("--prediction", dest="prediction", type=Path, default=DEFAULT_PREDICTION, help="Path to prediction JSON file.")
    parser.add_argument(
        "--unrefined",
        action="store_true",
        help="Use unrefined boundaries from orchestrator (temp/Job/chapter_boundaries_unrefined.json) unless a custom --prediction is provided.",
    )
    parser.add_argument(
        "--refined",
        action="store_true",
        help="Use refined boundaries from refinement pass (temp/Job/chapter_boundaries_refined.json) unless a custom --prediction is provided.",
    )
    args = parser.parse_args(argv)
    prediction_path = args.prediction
    if args.unrefined and args.prediction == DEFAULT_PREDICTION:
        prediction_path = DEFAULT_UNREFINED_PREDICTION
    elif args.refined and args.prediction == DEFAULT_PREDICTION:
        prediction_path = DEFAULT_REFINED_PREDICTION
    return run_analysis(args.ground_truth, prediction_path)


if __name__ == "__main__":
    sys.exit(main())

