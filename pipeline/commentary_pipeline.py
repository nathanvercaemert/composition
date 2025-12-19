#!/usr/bin/env python3
from __future__ import annotations

"""
End-to-end commentary extraction and structuring pipeline.

Processes every chapter of a book sequentially, running:
1) Batch commentary extraction
2) Batch commentary structuring

The pipeline aggregates token usage across all chapters, enforces strict
integrity checks (missing outputs, verse errors, batch failures), and emits a
comprehensive success/failure summary. Any failure terminates the pipeline.

Example usage:
    python pipeline/commentary_pipeline.py Job
    python pipeline/commentary_pipeline.py Gen --force --poll-interval 30
    python pipeline/commentary_pipeline.py Ps --log-format jsonl --log-file pipeline.log
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Callable, Iterable, Sequence

import requests

# Paths -----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMENTARY_ROOT = PROJECT_ROOT / "commentary-extraction"
EXTRACTION_DIR = COMMENTARY_ROOT / "batch-commentary-extraction"
STRUCTURING_DIR = COMMENTARY_ROOT / "commentary-structuring" / "batch-commentary-structuring"


def _load_module(path: Path, alias: str):
    """Load a module from a specific file path with a unique alias."""
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load batch runners with isolated sys.path to avoid module name collisions.
# Both extraction and structuring have identically-named files (prepare_batch.py, etc.)
# so we must ensure each module's directory is at the front of sys.path when it loads.

# Step 1: Set up sys.path for extraction module, then load it
for _path in (PROJECT_ROOT, COMMENTARY_ROOT, EXTRACTION_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

_extraction_module = _load_module(EXTRACTION_DIR / "batch_all.py", "commentary_extraction_batch_all")
run_extraction_batch = _extraction_module.run_all

# Step 2: Now add structuring directory and load structuring module
# Insert at position 0 so structuring's prepare_batch.py is found first
_structuring_path_str = str(STRUCTURING_DIR)
if _structuring_path_str not in sys.path:
    sys.path.insert(0, _structuring_path_str)

_structuring_module = _load_module(
    STRUCTURING_DIR / "batch_all.py", "commentary_structuring_batch_all"
)

# Late imports that rely on sys.path adjustments
from verses import VersesData  # noqa: E402
from ocr_loader import OCR_PROVIDERS, load_chapter_boundaries  # noqa: E402
run_structuring_batch = _structuring_module.run_all

# Constants -------------------------------------------------------------------
LOGGER_NAME = "commentary_pipeline"
SUMMARY_SEPARATOR = "=" * 80
LOG_FORMAT_HUMAN = "%(asctime)s.%(msecs)03d | %(levelname)8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
TOTAL_STAGES = 2
DEFAULT_TEMP_DIR = Path("temp/commentary-pipeline")
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
EXTRACTION_MAX_ATTEMPTS = 2  # initial attempt + one automatic retry
EXTRACTION_RETRY_DELAY_SECONDS = 20
RETRYABLE_EXTRACTION_MESSAGES = (
    "has no output_file_id",
    "not completed or has no output_file_id",
)


class ExitCode(IntEnum):
    """Enumerated exit codes for pipeline failure states."""

    SUCCESS = 0
    PREFLIGHT = 1
    EXTRACTION = 2
    STRUCTURING = 3
    INTERNAL = 10

    @classmethod
    def for_stage(cls, stage_num: int) -> ExitCode:
        mapping = {1: cls.EXTRACTION, 2: cls.STRUCTURING}
        return mapping.get(stage_num, cls.INTERNAL)


class PipelineFailure(Exception):
    """Represents a fatal pipeline error with structured context."""

    def __init__(
        self,
        exit_code: ExitCode,
        reason: str,
        *,
        stage_num: int | None = None,
        stage_label: str | None = None,
        chapter: int | None = None,
        expected_outputs: Sequence[Path] | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.exit_code = exit_code
        self.stage_num = stage_num
        self.stage_label = stage_label
        self.chapter = chapter
        self.expected_outputs = list(expected_outputs) if expected_outputs else None
        self.suggestion = suggestion


@dataclass
class TokenUsage:
    """Aggregate token usage metrics."""

    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_rate(self) -> float:
        if self.input_tokens == 0:
            return 0.0
        return (self.cached_tokens / self.input_tokens) * 100

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.cached_tokens += other.cached_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> TokenUsage:
        return cls(
            input_tokens=int(data.get("input_tokens", 0) or 0),
            cached_tokens=int(data.get("cached_tokens", 0) or 0),
            output_tokens=int(data.get("output_tokens", 0) or 0),
            reasoning_tokens=int(data.get("reasoning_tokens", 0) or 0),
        )


@dataclass
class StageStatistics:
    """Statistics for an individual stage (extraction/structuring) within a chapter."""

    stage_num: int
    label: str
    chapter: int
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float | None = None
    output_paths: list[Path] = field(default_factory=list)
    output_bytes: list[int] = field(default_factory=list)
    output_exists: bool = False
    verses_with_errors: int | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    status: str = "RUNNING"  # RUNNING, SUCCESS, FAILED


@dataclass
class ChapterStatistics:
    """Aggregate statistics for a single chapter."""

    chapter: int
    extraction: StageStatistics | None = None
    structuring: StageStatistics | None = None
    status: str = "PENDING"  # PENDING, COMPLETED, FAILED
    failure_reason: str | None = None


@dataclass
class PipelineStatistics:
    """Aggregate statistics for the overall pipeline."""

    book_name: str
    total_chapters: int
    start_time: datetime
    end_time: datetime | None = None
    total_duration_seconds: float | None = None
    chapters: list[ChapterStatistics] = field(default_factory=list)
    extraction_tokens: TokenUsage = field(default_factory=TokenUsage)
    structuring_tokens: TokenUsage = field(default_factory=TokenUsage)
    final_status: str = "RUNNING"  # RUNNING, SUCCESS, FAILED
    failure_stage: str | None = None
    failure_chapter: int | None = None
    failure_reason: str | None = None


# Formatting helpers ----------------------------------------------------------
def _elapsed_seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start).total_seconds(), 2)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    return f"{seconds:.2f}s"


def _format_cache_rate(usage: TokenUsage) -> str:
    if usage.input_tokens == 0:
        return "0.0%"
    return f"{usage.cache_rate:.1f}%"


def _format_stage_line(label: str, duration: float | None, bytes_sum: int | None) -> str:
    formatted_bytes = f"{bytes_sum:,} bytes" if bytes_sum is not None else "N/A"
    return f"      Duration: {format_duration(duration)} | Output Size: {formatted_bytes}"


# Logging infrastructure ------------------------------------------------------
class JSONLogFormatter(logging.Formatter):
    """Formatter that emits JSON lines for machine readability."""

    RESERVED_KEYS = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .astimezone()
            .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in self.RESERVED_KEYS:
                continue
            data[key] = self._serialize(value)

        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _serialize(value: object) -> object:
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, datetime):
            return value.astimezone().isoformat(timespec="milliseconds")
        if isinstance(value, IntEnum):
            return int(value)
        if isinstance(value, (list, tuple, set)):
            return [JSONLogFormatter._serialize(item) for item in value]
        return value


class PipelineLogger:
    """Configure logging, track statistics reference, and emit lifecycle events."""

    def __init__(
        self,
        *,
        book_name: str,
        stats: PipelineStatistics,
        log_level: int,
        log_format: str,
        quiet: bool,
        log_file: Path | None,
        stats_enabled: bool,
    ) -> None:
        self.log_format = log_format
        self.quiet = quiet
        self.stats_enabled = stats_enabled
        self.logger = logging.getLogger(LOGGER_NAME)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        self._configure_handlers(log_level, log_format, quiet, log_file)
        self.stats = stats

        self.logger.info(
            "Commentary pipeline starting for book '%s'",
            book_name,
            extra={
                "event": "pipeline_start",
                "book": book_name,
                "total_chapters": stats.total_chapters,
                "stats_enabled": stats_enabled,
            },
        )

    def _configure_handlers(
        self, log_level: int, log_format: str, quiet: bool, log_file: Path | None
    ) -> None:
        self.logger.handlers.clear()
        stdout_level = logging.ERROR if quiet else log_level

        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setLevel(stdout_level)
        stdout_handler.setFormatter(self._human_formatter())
        self.logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(stream=sys.stderr)
        stderr_handler.setLevel(log_level)
        stderr_handler.setFormatter(self._structured_formatter(log_format))
        self.logger.addHandler(stderr_handler)

        if log_file:
            self._validate_log_file(log_file)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(self._structured_formatter(log_format))
            self.logger.addHandler(file_handler)

    def _structured_formatter(self, log_format: str) -> logging.Formatter:
        if log_format == "jsonl":
            return JSONLogFormatter()
        return logging.Formatter(LOG_FORMAT_HUMAN, datefmt=LOG_DATE_FORMAT)

    @staticmethod
    def _human_formatter() -> logging.Formatter:
        return logging.Formatter(LOG_FORMAT_HUMAN, datefmt=LOG_DATE_FORMAT)

    def _validate_log_file(self, log_file: Path) -> None:
        parent = log_file.expanduser().resolve().parent
        if not parent.exists() or not parent.is_dir():
            self.logger.error(
                "Log file directory does not exist: %s",
                parent.as_posix(),
                extra={"event": "preflight_failed", "reason": "log_file_directory_missing"},
            )
            raise PipelineFailure(
                ExitCode.PREFLIGHT,
                reason=f"Log file directory does not exist: {parent.as_posix()}",
                stage_label="preflight",
                suggestion="Create the directory or choose another log file location.",
            )
        if not os.access(parent, os.W_OK):
            self.logger.error(
                "Log file directory is not writable: %s",
                parent.as_posix(),
                extra={"event": "preflight_failed", "reason": "log_file_directory_not_writable"},
            )
            raise PipelineFailure(
                ExitCode.PREFLIGHT,
                reason=f"Log file directory is not writable: {parent.as_posix()}",
                stage_label="preflight",
                suggestion="Adjust permissions or choose another log file location.",
            )

    def pipeline_success(self, summary: str) -> None:
        self.logger.info(summary, extra={"event": "pipeline_summary"})
        if self.log_format == "jsonl":
            self.logger.info(
                "pipeline_complete",
                extra={
                    "event": "pipeline_complete",
                    "status": "SUCCESS",
                    "book": self.stats.book_name,
                    "chapters_attempted": len(self.stats.chapters),
                    "chapters_completed": _completed_chapters(self.stats),
                    "total_chapters": self.stats.total_chapters,
                    "duration_seconds": self.stats.total_duration_seconds,
                    "tokens": {
                        "extraction": vars(self.stats.extraction_tokens),
                        "structuring": vars(self.stats.structuring_tokens),
                    },
                },
            )

    def pipeline_failure(self, summary: str, failure: PipelineFailure) -> None:
        self.logger.error(summary, extra={"event": "pipeline_failed"})
        if self.log_format == "jsonl":
            self.logger.error(
                "pipeline_failed",
                extra={
                    "event": "pipeline_failed",
                    "status": "FAILED",
                    "book": self.stats.book_name,
                    "failure_stage": failure.stage_label,
                    "failure_chapter": failure.chapter,
                    "failure_reason": str(failure),
                    "exit_code": int(failure.exit_code),
                    "duration_seconds": self.stats.total_duration_seconds,
                    "chapters_attempted": len(self.stats.chapters),
                    "chapters_completed": _completed_chapters(self.stats),
                },
            )

    def unexpected_failure(self, exc: Exception) -> None:
        self.logger.error(
            "Unexpected error: %s",
            exc,
            exc_info=True,
            extra={"event": "unexpected_error"},
        )


# Argument parsing ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the commentary extraction + structuring pipeline for a book."
    )
    parser.add_argument("book_name", help="Book abbreviation (e.g., Job, Gen, Ps)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Polling interval in seconds for batch jobs (default: 60).",
    )
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="high",
        help="Reasoning effort level for GPT-5.2 (default: high).",
    )
    reasoning_group.add_argument(
        "--no-reasoning", action="store_true", help="Disable reasoning entirely for API calls."
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete intermediate batch files after successful completion of each chapter.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=DEFAULT_TEMP_DIR,
        help="Base directory for intermediate artifacts (default: temp/commentary-pipeline).",
    )
    parser.add_argument(
        "--log-level",
        choices=sorted(LOG_LEVELS.keys()),
        default="INFO",
        help="Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO).",
    )
    parser.add_argument(
        "--log-format",
        choices=["human", "jsonl"],
        default="human",
        help="Output format: human-readable lines or JSON Lines.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional path to write logs to a file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable stdout output while keeping structured stderr logs.",
    )
    parser.add_argument(
        "--stats",
        "--no-stats",
        dest="stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable final statistics summary (default: enabled).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show planned chapters without running batch jobs.",
    )
    parser.add_argument(
        "--chapter",
        type=int,
        help="Process only a specific chapter (overrides start/end).",
    )
    parser.add_argument(
        "--start-chapter",
        type=int,
        help="First chapter to process (inclusive).",
    )
    parser.add_argument(
        "--end-chapter",
        type=int,
        help="Last chapter to process (inclusive).",
    )
    parser.add_argument(
        "--quiet-structuring",
        action="store_true",
        help=argparse.SUPPRESS,  # retained for forward compatibility; no effect.
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Reuse existing chapter JSON and skip the extraction stage.",
    )
    return parser.parse_args()


# Preflight validation --------------------------------------------------------
def ensure_preflight(
    book_name: str,
    chapters: Iterable[int],
    project_root: Path,
    poll_interval: int,
    logger: PipelineLogger,
) -> None:
    """Validate required inputs before running any stages."""
    if poll_interval <= 0:
        raise PipelineFailure(
            ExitCode.PREFLIGHT,
            reason="Polling interval must be greater than zero.",
            stage_label="preflight",
            suggestion="Provide a positive integer for --poll-interval.",
        )

    book_dir = project_root / "books" / book_name
    chapters_dir = book_dir / "chapters"
    boundaries_path = book_dir / "chapter_boundaries.json"

    logger.logger.debug(
        "Running preflight checks",
        extra={
            "event": "preflight_start",
            "book_dir": book_dir.as_posix(),
            "boundaries_path": boundaries_path.as_posix(),
        },
    )

    if not book_dir.is_dir():
        raise PipelineFailure(
            ExitCode.PREFLIGHT,
            reason=f"Missing book directory: {book_dir.as_posix()}",
            stage_label="preflight",
            suggestion="Ensure the book directory exists under books/.",
        )

    if not boundaries_path.is_file():
        raise PipelineFailure(
            ExitCode.PREFLIGHT,
            reason=f"Missing chapter_boundaries.json at {boundaries_path.as_posix()}",
            stage_label="preflight",
            suggestion="Run the chapter boundary pipeline or place the boundaries file.",
        )

    boundaries = load_chapter_boundaries(boundaries_path)
    if not boundaries:
        raise PipelineFailure(
            ExitCode.PREFLIGHT,
            reason="chapter_boundaries.json is empty or malformed.",
            stage_label="preflight",
            suggestion="Recreate chapter_boundaries.json with valid page lists.",
        )

    if not chapters_dir.exists():
        chapters_dir.mkdir(parents=True, exist_ok=True)

    missing_ocr: list[str] = []
    for chapter in chapters:
        page_stems = boundaries.get(str(chapter), [])
        if not page_stems:
            raise PipelineFailure(
                ExitCode.PREFLIGHT,
                reason=f"No pages listed for chapter {chapter} in chapter_boundaries.json",
                stage_label="preflight",
                suggestion="Verify chapter boundaries include page stems for all chapters.",
            )
        for page in page_stems:
            for _, provider_dir in OCR_PROVIDERS:
                ocr_path = book_dir / provider_dir / f"{page}.txt"
                if not ocr_path.is_file():
                    missing_ocr.append(ocr_path.as_posix())

    if missing_ocr:
        raise PipelineFailure(
            ExitCode.PREFLIGHT,
            reason="Missing OCR text files required for prompts.",
            stage_label="preflight",
            suggestion=f"Create the missing OCR files (first missing: {missing_ocr[0]}).",
        )

    logger.logger.info(
        "Preflight validation passed",
        extra={"event": "preflight_complete", "book_dir": book_dir.as_posix()},
    )


# Core pipeline helpers -------------------------------------------------------
def _determine_chapters(book_name: str, args: argparse.Namespace, verses: VersesData) -> list[int]:
    """Resolve which chapters to process based on CLI flags."""
    if not verses.has_book(book_name):
        raise PipelineFailure(
            ExitCode.PREFLIGHT,
            reason=f"Unknown book abbreviation: {book_name}",
            stage_label="preflight",
            suggestion="Verify the abbreviation matches verses.json entries.",
        )

    total = verses.get_chapter_count(book_name)
    if args.chapter:
        if args.chapter < 1 or args.chapter > total:
            raise PipelineFailure(
                ExitCode.PREFLIGHT,
                reason=f"Chapter {args.chapter} is out of range for {book_name} (1-{total}).",
                stage_label="preflight",
                suggestion="Provide a valid chapter number within range.",
            )
        return [args.chapter]

    start = args.start_chapter or 1
    end = args.end_chapter or total
    if start < 1 or end < 1 or start > total or end > total or start > end:
        raise PipelineFailure(
            ExitCode.PREFLIGHT,
            reason=f"Invalid chapter range {start}-{end} for {book_name} (1-{total}).",
            stage_label="preflight",
            suggestion="Adjust --start-chapter/--end-chapter to a valid inclusive range.",
        )
    return list(range(start, end + 1))


def _ensure_no_existing_outputs(outputs: Sequence[Path], force: bool, stage_label: str, chapter: int) -> None:
    if force:
        return
    collisions = [p for p in outputs if p.exists()]
    if collisions:
        names = ", ".join(p.as_posix() for p in collisions)
        raise PipelineFailure(
            ExitCode.for_stage(1 if stage_label == "Extraction" else 2),
            reason=f"Output already exists: {names}",
            stage_num=1 if stage_label == "Extraction" else 2,
            stage_label=stage_label.lower(),
            chapter=chapter,
            expected_outputs=outputs,
            suggestion="Re-run with --force to overwrite existing outputs.",
        )


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineFailure(
            ExitCode.INTERNAL,
            reason=f"Failed to parse JSON at {path}: {exc}",
            stage_label="validation",
            suggestion="Inspect the JSON file for corruption and rerun.",
        ) from exc


def _finalize_stage(stage: StageStatistics, outputs: Sequence[Path], verses_with_errors: int, tokens: TokenUsage) -> None:
    stage.end_time = datetime.now(tz=timezone.utc).astimezone()
    stage.duration_seconds = _elapsed_seconds(stage.start_time, stage.end_time)
    stage.output_paths = list(outputs)
    stage.output_bytes = [p.stat().st_size for p in outputs if p.exists()]
    stage.output_exists = all(p.exists() for p in outputs)
    stage.verses_with_errors = verses_with_errors
    stage.token_usage = tokens
    stage.status = "SUCCESS"


def _validate_verses_error_count(metadata: dict, stage_num: int, stage_label: str, chapter: int) -> int:
    verses_with_errors = int(metadata.get("verses_with_errors", 0) or 0)
    if verses_with_errors > 0:
        raise PipelineFailure(
            ExitCode.for_stage(stage_num),
            reason=f"{verses_with_errors} verses reported errors in chapter {chapter}.",
            stage_num=stage_num,
            stage_label=stage_label,
            chapter=chapter,
            suggestion="Review the batch outputs and address verse errors before re-running with --force.",
        )
    return verses_with_errors


def _as_token_usage(data: dict[str, int] | None) -> TokenUsage:
    if data is None:
        return TokenUsage()
    return TokenUsage.from_dict(data)


def _is_retryable_file_download_error(exc: Exception) -> bool:
    """Detect retryable failures when downloading files from the API."""
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response else None
        return status in RETRYABLE_STATUS_CODES
    message = str(exc).lower()
    return "server error" in message and "/files/" in message


def _is_retryable_extraction_failure(exc: Exception) -> bool:
    """Determine if an extraction attempt should be retried automatically."""
    message = str(exc).lower()
    if any(token in message for token in RETRYABLE_EXTRACTION_MESSAGES):
        return True
    return _is_retryable_file_download_error(exc)


def _reuse_extraction_output(
    book_name: str, chapter: int, *, logger: logging.Logger
) -> tuple[StageStatistics, Path]:
    """Build a successful extraction stage from an existing output file."""
    stage = StageStatistics(
        stage_num=1,
        label="Extraction",
        chapter=chapter,
        start_time=datetime.now(tz=timezone.utc).astimezone(),
    )
    output_path = PROJECT_ROOT / "books" / book_name / "chapters" / f"{chapter}.json"

    if not output_path.exists():
        raise PipelineFailure(
            ExitCode.EXTRACTION,
            reason=(
                f"--skip-extraction was provided but expected extraction output is missing: "
                f"{output_path.as_posix()}"
            ),
            stage_num=1,
            stage_label="extraction",
            chapter=chapter,
            expected_outputs=[output_path],
            suggestion="Remove --skip-extraction or regenerate the extraction output with --force.",
        )

    logger.info(
        "Skipping extraction for chapter %s; reusing %s",
        chapter,
        output_path,
        extra={"event": "extraction_skipped", "chapter": chapter, "path": output_path.as_posix()},
    )

    payload = _load_json(output_path)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    verses_with_errors = _validate_verses_error_count(metadata, 1, "extraction", chapter)

    _finalize_stage(stage, [output_path], verses_with_errors, TokenUsage())
    logger.info(
        "Reused extraction output for chapter %s",
        chapter,
        extra={
            "event": "extraction_reused",
            "chapter": chapter,
            "verses_with_errors": verses_with_errors,
        },
    )
    return stage, output_path


def run_extraction_stage(
    book_name: str,
    chapter: int,
    *,
    force: bool,
    poll_interval: int,
    temp_dir: Path,
    logger: logging.Logger,
    cleanup: bool,
    reasoning_effort: str,
    use_reasoning: bool,
) -> tuple[StageStatistics, Path]:
    stage = StageStatistics(
        stage_num=1,
        label="Extraction",
        chapter=chapter,
        start_time=datetime.now(tz=timezone.utc).astimezone(),
    )
    expected_output = PROJECT_ROOT / "books" / book_name / "chapters" / f"{chapter}.json"
    _ensure_no_existing_outputs([expected_output], force, "Extraction", chapter)

    logger.info(
        "Starting commentary extraction for chapter %s", chapter, extra={"event": "stage_start", "stage": "extraction", "chapter": chapter}
    )

    extraction_kwargs = {
        "book_name": book_name,
        "chapter_number": chapter,
        "force": force,
        "tokens": True,  # always collect tokens
        "poll_interval": poll_interval,
        "output_dir": expected_output.parent,
        "temp_dir": temp_dir,
        "cleanup": cleanup,
        "return_usage": True,
    }

    # Map reasoning flags for extraction run_all signature
    extraction_kwargs.update(
        {
            "no_reasoning": not use_reasoning,
            "low_reasoning": use_reasoning and reasoning_effort == "low",
            "med_reasoning": use_reasoning and reasoning_effort == "medium",
            "high_reasoning": use_reasoning and reasoning_effort == "high",
            "xhigh_reasoning": use_reasoning and reasoning_effort == "xhigh",
        }
    )

    result: object | None = None
    try:
        for attempt in range(1, EXTRACTION_MAX_ATTEMPTS + 1):
            try:
                result = run_extraction_batch(**extraction_kwargs)
                break
            except FileExistsError:
                # Do not retry when outputs already exist; propagate as before
                raise
            except Exception as exc:  # noqa: BLE001
                if attempt < EXTRACTION_MAX_ATTEMPTS and _is_retryable_extraction_failure(exc):
                    logger.warning(
                        "Extraction attempt %s/%s failed; retrying in %ss: %s",
                        attempt,
                        EXTRACTION_MAX_ATTEMPTS,
                        EXTRACTION_RETRY_DELAY_SECONDS,
                        exc,
                        extra={
                            "event": "extraction_retry",
                            "stage": "extraction",
                            "chapter": chapter,
                            "attempt": attempt,
                            "max_attempts": EXTRACTION_MAX_ATTEMPTS,
                        },
                    )
                    time.sleep(EXTRACTION_RETRY_DELAY_SECONDS)
                    continue
                raise
    except FileExistsError as exc:
        raise PipelineFailure(
            ExitCode.EXTRACTION,
            reason=str(exc),
            stage_num=1,
            stage_label="extraction",
            chapter=chapter,
            expected_outputs=[expected_output],
            suggestion="Re-run with --force to overwrite existing extraction output.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise PipelineFailure(
            ExitCode.EXTRACTION,
            reason=f"Extraction failed for chapter {chapter} after {EXTRACTION_MAX_ATTEMPTS} attempt(s): {exc}",
            stage_num=1,
            stage_label="extraction",
            chapter=chapter,
            expected_outputs=[expected_output],
            suggestion=(
                "Check batch job status and logs; rerun with --force after addressing the underlying issue."
            ),
        ) from exc

    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        final_output, tokens_raw = result
    else:
        final_output = result  # type: ignore[assignment]
        tokens_raw = None

    if not final_output.exists():
        raise PipelineFailure(
            ExitCode.EXTRACTION,
            reason="Expected extraction output was not created.",
            stage_num=1,
            stage_label="extraction",
            chapter=chapter,
            expected_outputs=[expected_output],
            suggestion="Inspect batch logs and ensure process_results completed successfully.",
        )

    payload = _load_json(final_output)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    verses_with_errors = _validate_verses_error_count(metadata, 1, "extraction", chapter)

    _finalize_stage(stage, [final_output], verses_with_errors, _as_token_usage(tokens_raw))
    logger.info(
        "Completed extraction for chapter %s in %s",
        chapter,
        format_duration(stage.duration_seconds),
        extra={
            "event": "stage_complete",
            "stage": "extraction",
            "chapter": chapter,
            "duration_seconds": stage.duration_seconds,
            "verses_with_errors": verses_with_errors,
            "tokens": vars(stage.token_usage),
        },
    )
    return stage, final_output


def run_structuring_stage(
    book_name: str,
    chapter: int,
    *,
    force: bool,
    poll_interval: int,
    temp_dir: Path,
    logger: logging.Logger,
    cleanup: bool,
    reasoning_effort: str,
    use_reasoning: bool,
    input_file: Path,
) -> tuple[StageStatistics, Path, Path]:
    stage = StageStatistics(
        stage_num=2,
        label="Structuring",
        chapter=chapter,
        start_time=datetime.now(tz=timezone.utc).astimezone(),
    )

    output_dir = PROJECT_ROOT / "books" / book_name / "chapters" / str(chapter)
    json_output = output_dir / "structured_commentary.json"
    md_output = output_dir / "structured_commentary.md"
    _ensure_no_existing_outputs([json_output, md_output], force, "Structuring", chapter)

    logger.info(
        "Starting commentary structuring for chapter %s",
        chapter,
        extra={"event": "stage_start", "stage": "structuring", "chapter": chapter},
    )

    structuring_kwargs = {
        "book_name": book_name,
        "chapter_number": chapter,
        "force": force,
        "tokens": True,  # always collect tokens
        "poll_interval": poll_interval,
        "input_file": input_file,
        "output_dir": output_dir,
        "temp_dir": temp_dir,
        "reasoning_effort": reasoning_effort,
        "use_reasoning": use_reasoning,
        "cleanup": cleanup,
        "return_usage": True,
    }

    max_attempts = 2
    attempt = 0
    while True:
        attempt += 1
        try:
            result = run_structuring_batch(**structuring_kwargs)
            break
        except FileExistsError as exc:
            raise PipelineFailure(
                ExitCode.STRUCTURING,
                reason=str(exc),
                stage_num=2,
                stage_label="structuring",
                chapter=chapter,
                expected_outputs=[json_output, md_output],
                suggestion="Re-run with --force to overwrite existing structuring outputs.",
            ) from exc
        except Exception as exc:  # noqa: BLE001
            if attempt < max_attempts and _is_retryable_file_download_error(exc):
                logger.warning(
                    "Structuring attempt %s/%s for chapter %s failed with retryable download error; retrying.",
                    attempt,
                    max_attempts,
                    chapter,
                    extra={"event": "structuring_retry", "chapter": chapter, "attempt": attempt},
                )
                time.sleep(min(5 * attempt, 30))
                continue
            raise PipelineFailure(
                ExitCode.STRUCTURING,
                reason=f"Structuring failed for chapter {chapter}: {exc}",
                stage_num=2,
                stage_label="structuring",
                chapter=chapter,
                expected_outputs=[json_output, md_output],
                suggestion="Check batch job status and logs; fix issues then rerun with --force.",
            ) from exc

    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        outputs_raw, tokens_raw = result
        json_output_path, md_output_path = outputs_raw
    else:
        json_output_path, md_output_path = result  # type: ignore[misc]
        tokens_raw = None

    if not json_output_path.exists() or not md_output_path.exists():
        raise PipelineFailure(
            ExitCode.STRUCTURING,
            reason="Expected structuring outputs were not created.",
            stage_num=2,
            stage_label="structuring",
            chapter=chapter,
            expected_outputs=[json_output_path, md_output_path],
            suggestion="Inspect batch logs and ensure process_results completed successfully.",
        )

    payload = _load_json(json_output_path)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    verses_with_errors = _validate_verses_error_count(metadata, 2, "structuring", chapter)

    _finalize_stage(
        stage,
        [json_output_path, md_output_path],
        verses_with_errors,
        _as_token_usage(tokens_raw),
    )

    logger.info(
        "Completed structuring for chapter %s in %s",
        chapter,
        format_duration(stage.duration_seconds),
        extra={
            "event": "stage_complete",
            "stage": "structuring",
            "chapter": chapter,
            "duration_seconds": stage.duration_seconds,
            "verses_with_errors": verses_with_errors,
            "tokens": vars(stage.token_usage),
        },
    )
    return stage, json_output_path, md_output_path


# Summary builders ------------------------------------------------------------
def _completed_chapters(stats: PipelineStatistics) -> int:
    return sum(1 for ch in stats.chapters if ch.status == "COMPLETED")


def _stage_counts(stats: PipelineStatistics) -> tuple[int, int]:
    extracted = sum(1 for ch in stats.chapters if ch.extraction and ch.extraction.status == "SUCCESS")
    structured = sum(1 for ch in stats.chapters if ch.structuring and ch.structuring.status == "SUCCESS")
    return extracted, structured


def _token_box(extraction: TokenUsage, structuring: TokenUsage) -> list[str]:
    def fmt_rate(usage: TokenUsage) -> str:
        return f"{usage.cache_rate:>4.1f}%" if usage.input_tokens else " 0.0%"

    extraction_total = extraction.total
    structuring_total = structuring.total
    grand_input = extraction.input_tokens + structuring.input_tokens
    grand_cached = extraction.cached_tokens + structuring.cached_tokens
    grand_output = extraction.output_tokens + structuring.output_tokens
    grand_total = grand_input + grand_output
    overall_cache_rate = (extraction.cached_tokens + structuring.cached_tokens) / grand_input * 100 if grand_input else 0.0

    lines = [
        "Token Usage Summary:",
        "  ┌─────────────────────────────────────────────────────────────────┐",
        "  │ EXTRACTION PHASE                                                │",
        f"  │   Input Tokens:      {extraction.input_tokens:>12,}                     │",
        f"  │     └─ Cached:       {extraction.cached_tokens:>12,} ({fmt_rate(extraction)})│",
        f"  │   Output Tokens:     {extraction.output_tokens:>12,}                    │",
        f"  │     └─ Reasoning:    {extraction.reasoning_tokens:>12,}                    │",
        f"  │   Subtotal:          {extraction_total:>12,}                    │",
        "  ├─────────────────────────────────────────────────────────────────┤",
        "  │ STRUCTURING PHASE                                               │",
        f"  │   Input Tokens:      {structuring.input_tokens:>12,}                    │",
        f"  │     └─ Cached:       {structuring.cached_tokens:>12,} ({fmt_rate(structuring)})│",
        f"  │   Output Tokens:     {structuring.output_tokens:>12,}                    │",
        f"  │     └─ Reasoning:    {structuring.reasoning_tokens:>12,}                    │",
        f"  │   Subtotal:          {structuring_total:>12,}                    │",
        "  ├─────────────────────────────────────────────────────────────────┤",
        "  │ GRAND TOTAL                                                     │",
        f"  │   All Input Tokens:  {grand_input:>12,}                    │",
        f"  │     └─ Cached:       {grand_cached:>12,}                    │",
        f"  │   All Output Tokens: {grand_output:>12,}                    │",
        f"  │   All Tokens:        {grand_total:>12,}                    │",
        f"  │   Overall Cache Rate:{overall_cache_rate:>11.1f}%                 │",
        "  └─────────────────────────────────────────────────────────────────┘",
    ]
    return lines


def _chapter_breakdown_line(chapter_stats: ChapterStatistics) -> str:
    ext = chapter_stats.extraction
    struct = chapter_stats.structuring
    ext_status = "✓" if ext and ext.status == "SUCCESS" else "✗"
    struct_status = "✓" if struct and struct.status == "SUCCESS" else "✗"
    ext_tokens = ext.token_usage.total if ext else 0
    struct_tokens = struct.token_usage.total if struct else 0
    ext_duration = format_duration(ext.duration_seconds if ext else None)
    struct_duration = format_duration(struct.duration_seconds if struct else None)
    return (
        f"  Chapter {chapter_stats.chapter:>2}: "
        f"Extraction {ext_status} ({ext_duration}, {ext_tokens:,} tokens) "
        f"→ Structuring {struct_status} ({struct_duration}, {struct_tokens:,} tokens)"
    )


def build_success_summary(stats: PipelineStatistics, chapters_total: int) -> str:
    extracted, structured = _stage_counts(stats)
    lines = [
        SUMMARY_SEPARATOR,
        "COMMENTARY PIPELINE EXECUTION SUMMARY".center(len(SUMMARY_SEPARATOR)),
        SUMMARY_SEPARATOR,
        "Status:           SUCCESS",
        f"Book:             {stats.book_name}",
        f"Total Chapters:   {chapters_total}",
        f"Total Duration:   {format_duration(stats.total_duration_seconds)}",
        "",
        "Phase Summary:",
        f"  Extraction:     {extracted}/{chapters_total} chapters",
        f"  Structuring:    {structured}/{chapters_total} chapters",
        "",
    ]

    lines.extend(_token_box(stats.extraction_tokens, stats.structuring_tokens))
    lines.append("")
    lines.append("Per-Chapter Breakdown:")
    for chapter_stats in stats.chapters:
        lines.append(_chapter_breakdown_line(chapter_stats))
    lines.append("")
    lines.extend(
        [
            "Output Files:",
            f"  Commentary JSON:  books/{stats.book_name}/chapters/*.json",
            f"  Structured JSON:  books/{stats.book_name}/chapters/*/structured_commentary.json",
            f"  Markdown:         books/{stats.book_name}/chapters/*/structured_commentary.md",
            SUMMARY_SEPARATOR,
            f"Pipeline completed successfully. All {chapters_total} chapters processed without errors.",
            SUMMARY_SEPARATOR,
        ]
    )
    return "\n".join(lines)


def build_failure_summary(
    stats: PipelineStatistics,
    failure: PipelineFailure,
    chapters_total: int,
) -> str:
    extracted, structured = _stage_counts(stats)
    lines = [
        SUMMARY_SEPARATOR,
        "COMMENTARY PIPELINE EXECUTION FAILED".center(len(SUMMARY_SEPARATOR)),
        SUMMARY_SEPARATOR,
        "Status:           FAILED",
        f"Book:             {stats.book_name}",
        f"Failed At:        Chapter {failure.chapter}, {failure.stage_label or 'unknown stage'}",
        f"Duration Before Failure: {format_duration(stats.total_duration_seconds)}",
        "",
        "Progress:",
        f"  Chapters Fully Completed:    {_completed_chapters(stats)}/{chapters_total}",
        f"  Extraction Completed:        {extracted}/{chapters_total}",
        f"  Structuring Completed:       {structured}/{chapters_total}",
        "",
        "Error Details:",
        f"  Stage:          {failure.stage_label}",
        f"  Chapter:        {failure.chapter}",
        f"  Message:        {failure}",
    ]
    if failure.expected_outputs:
        lines.append(f"  Expected Outputs: {', '.join(p.as_posix() for p in failure.expected_outputs)}")
    if failure.suggestion:
        lines.append(f"\nSuggestion: {failure.suggestion}")

    lines.extend(
        [
            "",
            "Token Usage (Before Failure):",
            f"  Input Tokens:      {stats.extraction_tokens.input_tokens + stats.structuring_tokens.input_tokens:,}",
            f"  Cached Input:      {stats.extraction_tokens.cached_tokens + stats.structuring_tokens.cached_tokens:,}",
            f"  Output Tokens:     {stats.extraction_tokens.output_tokens + stats.structuring_tokens.output_tokens:,}",
            f"  Total (in+out):    {stats.extraction_tokens.total + stats.structuring_tokens.total:,}",
            SUMMARY_SEPARATOR,
            f"Pipeline terminated due to fatal error. Partial outputs may exist for chapters 1-{_completed_chapters(stats)}.",
            "To resume, fix the issue and re-run with --force to overwrite existing outputs.",
            SUMMARY_SEPARATOR,
        ]
    )
    return "\n".join(lines)


# Main orchestration ----------------------------------------------------------
def run_pipeline() -> None:
    args = parse_args()
    book_name = args.book_name
    log_level = LOG_LEVELS[args.log_level.upper()]
    project_root = PROJECT_ROOT
    verses = VersesData()

    chapters_to_process = _determine_chapters(book_name, args, verses)
    stats = PipelineStatistics(
        book_name=book_name,
        total_chapters=len(chapters_to_process),
        start_time=datetime.now(tz=timezone.utc).astimezone(),
    )
    logger = PipelineLogger(
        book_name=book_name,
        stats=stats,
        log_level=log_level,
        log_format=args.log_format,
        quiet=args.quiet,
        log_file=args.log_file,
        stats_enabled=args.stats,
    )

    try:
        ensure_preflight(book_name, chapters_to_process, project_root, args.poll_interval, logger)

        if args.dry_run:
            stats.final_status = "SUCCESS"
            stats.end_time = datetime.now(tz=timezone.utc).astimezone()
            stats.total_duration_seconds = _elapsed_seconds(stats.start_time, stats.end_time)
            summary = (
                "Dry run complete. No batch jobs executed.\n"
                f"Planned chapters: {', '.join(str(ch) for ch in chapters_to_process)}"
            )
            logger.logger.info(summary)
            logger.pipeline_success(summary if args.stats else "Dry run complete.")
            sys.exit(int(ExitCode.SUCCESS))

        use_reasoning = not args.no_reasoning
        reasoning_effort = "high" if args.no_reasoning else args.reasoning_effort

        for chapter in chapters_to_process:
            chapter_stats = ChapterStatistics(chapter=chapter)
            stats.chapters.append(chapter_stats)
            chapter_temp_base = args.temp_dir / book_name / f"chapter-{chapter:03d}"
            extraction_temp = chapter_temp_base / "extraction"
            structuring_temp = chapter_temp_base / "structuring"

            if args.skip_extraction:
                extraction_stage, extraction_output = _reuse_extraction_output(
                    book_name, chapter, logger=logger.logger
                )
            else:
                extraction_stage, extraction_output = run_extraction_stage(
                    book_name,
                    chapter,
                    force=args.force,
                    poll_interval=args.poll_interval,
                    temp_dir=extraction_temp,
                    logger=logger.logger,
                    cleanup=args.cleanup,
                    reasoning_effort=reasoning_effort,
                    use_reasoning=use_reasoning,
                )
            chapter_stats.extraction = extraction_stage
            stats.extraction_tokens.add(extraction_stage.token_usage)

            structuring_stage, _, _ = run_structuring_stage(
                book_name,
                chapter,
                force=args.force,
                poll_interval=args.poll_interval,
                temp_dir=structuring_temp,
                logger=logger.logger,
                cleanup=args.cleanup,
                reasoning_effort=reasoning_effort,
                use_reasoning=use_reasoning,
                input_file=extraction_output,
            )
            chapter_stats.structuring = structuring_stage
            stats.structuring_tokens.add(structuring_stage.token_usage)
            chapter_stats.status = "COMPLETED"

        stats.final_status = "SUCCESS"
        stats.end_time = datetime.now(tz=timezone.utc).astimezone()
        stats.total_duration_seconds = _elapsed_seconds(stats.start_time, stats.end_time)

        if args.stats:
            summary = build_success_summary(stats, len(chapters_to_process))
            logger.pipeline_success(summary)
        else:
            logger.logger.info("Pipeline completed successfully (statistics reporting disabled).")
        sys.exit(int(ExitCode.SUCCESS))

    except PipelineFailure as failure:
        stats.final_status = "FAILED"
        stats.failure_stage = failure.stage_label
        stats.failure_chapter = failure.chapter
        stats.failure_reason = str(failure)
        stats.end_time = datetime.now(tz=timezone.utc).astimezone()
        stats.total_duration_seconds = _elapsed_seconds(stats.start_time, stats.end_time)

        summary = build_failure_summary(stats, failure, len(chapters_to_process))
        logger.pipeline_failure(summary, failure)
        sys.exit(int(failure.exit_code))
    except KeyboardInterrupt:
        stats.final_status = "FAILED"
        stats.failure_reason = "Pipeline interrupted by user."
        stats.end_time = datetime.now(tz=timezone.utc).astimezone()
        stats.total_duration_seconds = _elapsed_seconds(stats.start_time, stats.end_time)
        failure = PipelineFailure(
            ExitCode.INTERNAL,
            reason="Pipeline interrupted by user.",
            stage_label="interrupt",
        )
        summary = build_failure_summary(stats, failure, len(chapters_to_process))
        logger.pipeline_failure(summary, failure)
        sys.exit(int(ExitCode.INTERNAL))
    except Exception as exc:  # noqa: BLE001
        stats.final_status = "FAILED"
        stats.failure_reason = str(exc)
        stats.end_time = datetime.now(tz=timezone.utc).astimezone()
        stats.total_duration_seconds = _elapsed_seconds(stats.start_time, stats.end_time)
        logger.unexpected_failure(exc)
        sys.exit(int(ExitCode.INTERNAL))


def main() -> None:
    run_pipeline()


__all__ = ["main", "run_pipeline"]


if __name__ == "__main__":
    main()


