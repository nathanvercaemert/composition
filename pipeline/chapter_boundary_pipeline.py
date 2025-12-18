#!/usr/bin/env python3

from __future__ import annotations

"""
Chapter boundary detection pipeline.

Orchestrates the complete workflow: detection → refinement → finalization.
Any stage failure terminates the pipeline immediately.

Test commands:
1. Normal run with stats:
   python pipeline/chapter_boundary_pipeline.py test-book
2. JSONL format:
   python pipeline/chapter_boundary_pipeline.py test-book --log-format jsonl 2>pipeline.jsonl
3. Quiet mode with file logging:
   python pipeline/chapter_boundary_pipeline.py test-book --quiet --log-file pipeline.log
4. Debug verbosity:
   python pipeline/chapter_boundary_pipeline.py test-book --log-level DEBUG
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Sequence

LOGGER_NAME = "chapter_boundary_pipeline"
SUMMARY_SEPARATOR = "=" * 80
LOG_FORMAT_HUMAN = "%(asctime)s.%(msecs)03d | %(levelname)8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
TOTAL_STAGES = 3
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


class ExitCode(IntEnum):
    """Enumerated exit codes for pipeline failure states."""

    SUCCESS = 0
    PREFLIGHT = 1
    STAGE1 = 2
    STAGE2 = 3
    STAGE3 = 4
    INTERNAL = 10

    @classmethod
    def for_stage(cls, stage_num: int) -> ExitCode:
        """Return the stage-specific exit code."""
        mapping = {1: cls.STAGE1, 2: cls.STAGE2, 3: cls.STAGE3}
        return mapping.get(stage_num, cls.INTERNAL)


class PipelineFailure(Exception):
    """Represents a fatal pipeline error with structured context."""

    def __init__(
        self,
        exit_code: ExitCode,
        reason: str,
        stage_num: int | None = None,
        stage_label: str | None = None,
        command: Sequence[str] | None = None,
        expected_output: Path | None = None,
        subprocess_exit_code: int | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.exit_code = exit_code
        self.stage_num = stage_num
        self.stage_label = stage_label
        self.command = list(command) if command else None
        self.expected_output = expected_output
        self.subprocess_exit_code = subprocess_exit_code
        self.suggestion = suggestion


@dataclass
class StageStatistics:
    """Statistics for an individual pipeline stage."""

    stage_num: int
    label: str
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    output_file: Path | None = None
    output_file_exists: bool = False
    output_file_bytes: int | None = None


@dataclass
class PipelineStatistics:
    """Aggregate statistics for the pipeline execution."""

    book_name: str
    start_time: datetime
    input_dir: Path
    expected_output: Path
    end_time: datetime | None = None
    total_duration_seconds: float | None = None
    stages: list[StageStatistics] = field(default_factory=list)
    final_status: str = "RUNNING"  # RUNNING, SUCCESS, FAILED
    failure_stage: int | None = None
    failure_reason: str | None = None
    output_file_bytes: int | None = None


def format_duration(seconds: float | None) -> str:
    """Format a duration in seconds with two decimals."""
    if seconds is None:
        return "N/A"
    return f"{seconds:.2f}s"


def format_bytes(byte_count: int | None) -> str:
    """Format a byte count for human readability."""
    if byte_count is None:
        return "N/A"
    return f"{byte_count:,} bytes"


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
    """Configure logging, track statistics, and emit lifecycle events."""

    def __init__(
        self,
        *,
        book_name: str,
        input_dir: Path,
        final_output: Path,
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

        start_time = datetime.now(tz=timezone.utc).astimezone()
        self.stats = PipelineStatistics(
            book_name=book_name,
            start_time=start_time,
            input_dir=input_dir.resolve(),
            expected_output=final_output.resolve(),
        )

        self.logger.info(
            "Pipeline starting for book '%s'",
            book_name,
            extra={
                "event": "pipeline_start",
                "book": book_name,
                "input_dir": str(self.stats.input_dir),
                "final_output": str(self.stats.expected_output),
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

    def stage_start(
        self, stage_num: int, label: str, cmd: Sequence[str], expected_output: Path, cwd: Path
    ) -> None:
        started = datetime.now(tz=timezone.utc).astimezone()
        self.stats.stages.append(
            StageStatistics(stage_num=stage_num, label=label, start_time=started)
        )
        self.logger.info(
            "Stage %d starting: %s",
            stage_num,
            label,
            extra={
                "event": "stage_start",
                "stage": stage_num,
                "label": label,
                "expected_output": expected_output.as_posix(),
            },
        )
        self.logger.debug(
            "Command: %s",
            " ".join(cmd),
            extra={
                "event": "stage_command",
                "stage": stage_num,
                "command": list(cmd),
                "cwd": cwd.as_posix(),
            },
        )

    def record_stage_outcome(
        self,
        stage_num: int,
        label: str,
        exit_code: int,
        expected_output: Path,
        level: int = logging.INFO,
    ) -> StageStatistics:
        stage_stats = self._get_stage_stats(stage_num)
        finished = datetime.now(tz=timezone.utc).astimezone()
        stage_stats.end_time = finished
        stage_stats.exit_code = exit_code
        stage_stats.output_file = expected_output
        stage_stats.output_file_exists = expected_output.is_file()
        stage_stats.output_file_bytes = (
            expected_output.stat().st_size if stage_stats.output_file_exists else None
        )
        stage_stats.duration_seconds = _elapsed_seconds(stage_stats.start_time, stage_stats.end_time)

        message = (
            f"Stage {stage_num} completed in {format_duration(stage_stats.duration_seconds)}"
            if exit_code == 0 and stage_stats.output_file_exists
            else f"Stage {stage_num} ended with errors after {format_duration(stage_stats.duration_seconds)}"
        )

        self.logger.log(
            level,
            message,
            extra={
                "event": "stage_end",
                "stage": stage_num,
                "label": label,
                "duration_seconds": stage_stats.duration_seconds,
                "exit_code": exit_code,
                "output_file": expected_output.as_posix(),
                "output_file_exists": stage_stats.output_file_exists,
                "output_file_bytes": stage_stats.output_file_bytes,
            },
        )
        return stage_stats

    def pipeline_success(self, final_output: Path) -> None:
        self.stats.final_status = "SUCCESS"
        self.stats.end_time = datetime.now(tz=timezone.utc).astimezone()
        self.stats.total_duration_seconds = _elapsed_seconds(
            self.stats.start_time, self.stats.end_time
        )
        if final_output.is_file():
            self.stats.output_file_bytes = final_output.stat().st_size

        attempted, completed = self._stage_counts()
        if not self.stats_enabled:
            self.logger.info(
                "Pipeline completed successfully (statistics reporting disabled).",
                extra={
                    "event": "pipeline_complete",
                    "status": "SUCCESS",
                    "book": self.stats.book_name,
                    "stages_attempted": attempted,
                    "stages_completed": completed,
                    "stages_total": TOTAL_STAGES,
                    "stats_enabled": False,
                    "final_output": final_output.as_posix(),
                },
            )
            return

        summary = self._build_success_summary(final_output)
        self.logger.info(summary, extra={"event": "pipeline_summary"})
        if self.log_format == "jsonl":
            self.logger.info(
                "pipeline_complete",
                extra={
                    "event": "pipeline_complete",
                    "status": "SUCCESS",
                    "book": self.stats.book_name,
                    "total_duration_seconds": self.stats.total_duration_seconds,
                    "stages_attempted": attempted,
                    "stages_completed": completed,
                    "stages_total": TOTAL_STAGES,
                    "final_output": final_output.as_posix(),
                    "final_output_bytes": self.stats.output_file_bytes,
                },
            )

    def pipeline_failure(self, failure: PipelineFailure) -> None:
        self.stats.final_status = "FAILED"
        self.stats.end_time = datetime.now(tz=timezone.utc).astimezone()
        self.stats.total_duration_seconds = _elapsed_seconds(
            self.stats.start_time, self.stats.end_time
        )
        self.stats.failure_stage = failure.stage_num
        self.stats.failure_reason = str(failure)

        summary = self._build_failure_summary(failure)
        self.logger.error(summary, extra={"event": "pipeline_failed"})
        if self.log_format == "jsonl":
            attempted, completed = self._stage_counts()
            self.logger.error(
                "pipeline_failed",
                extra={
                    "event": "pipeline_failed",
                    "status": "FAILED",
                    "book": self.stats.book_name,
                    "failure_stage": failure.stage_num,
                    "failure_reason": str(failure),
                    "exit_code": int(failure.exit_code),
                    "duration_seconds": self.stats.total_duration_seconds,
                    "stages_attempted": attempted,
                    "stages_completed": completed,
                    "stages_total": TOTAL_STAGES,
                },
            )

    def unexpected_failure(self, exc: Exception) -> None:
        self.logger.error(
            "Unexpected error: %s",
            exc,
            exc_info=True,
            extra={"event": "unexpected_error"},
        )

    def _get_stage_stats(self, stage_num: int) -> StageStatistics:
        for stage in self.stats.stages:
            if stage.stage_num == stage_num:
                return stage
        raise PipelineFailure(
            ExitCode.INTERNAL,
            reason=f"Stage {stage_num} statistics not initialized.",
            stage_num=stage_num,
            stage_label="internal",
            suggestion="This indicates a programming error; please report the issue.",
        )

    def _build_success_summary(self, final_output: Path) -> str:
        attempted, completed = self._stage_counts()
        lines = [
            SUMMARY_SEPARATOR,
            "PIPELINE EXECUTION SUMMARY".center(len(SUMMARY_SEPARATOR)),
            SUMMARY_SEPARATOR,
            f"Status:           SUCCESS",
            f"Book:             {self.stats.book_name}",
            f"Input Directory:  {self.stats.input_dir.as_posix()}",
            f"Total Duration:   {format_duration(self.stats.total_duration_seconds)}",
            f"Stages Attempted: {attempted}/{TOTAL_STAGES}",
            f"Stages Completed: {completed}/{TOTAL_STAGES}",
            "",
            "Stage Details:",
        ]

        for stage in self.stats.stages:
            output_display = stage.output_file.as_posix() if stage.output_file else "N/A"
            lines.extend(
                [
                    f"  [{stage.stage_num}] {stage.label}",
                    f"      Duration: {format_duration(stage.duration_seconds)} | "
                    f"Output: {output_display} ({format_bytes(stage.output_file_bytes)})",
                ]
            )

        lines.extend(
            [
                "",
                f"Final Output: {final_output.as_posix()}",
                SUMMARY_SEPARATOR,
            ]
        )
        return "\n".join(lines)

    def _build_failure_summary(self, failure: PipelineFailure) -> str:
        attempted, completed = self._stage_counts()
        failed_at = (
            f"Stage {failure.stage_num} ({failure.stage_label})"
            if failure.stage_num is not None
            else "Preflight validation"
        )
        lines = [
            SUMMARY_SEPARATOR,
            "PIPELINE EXECUTION FAILED".center(len(SUMMARY_SEPARATOR)),
            SUMMARY_SEPARATOR,
            "Status:           FAILED",
            f"Book:             {self.stats.book_name}",
            f"Failed At:        {failed_at}",
            f"Duration Before Failure: {format_duration(self.stats.total_duration_seconds)}",
            f"Stages Attempted: {attempted}/{TOTAL_STAGES}",
            f"Stages Completed: {completed}/{TOTAL_STAGES}",
            "",
            "Error Details:",
        ]

        if failure.command:
            lines.append(f"  Command: {' '.join(failure.command)}")
        if failure.subprocess_exit_code is not None:
            lines.append(f"  Exit Code: {failure.subprocess_exit_code}")
        if failure.expected_output:
            lines.append(f"  Expected Output: {failure.expected_output.as_posix()}")
        lines.append(f"  Reason: {failure}")
        if failure.suggestion:
            lines.append(f"\nSuggestion: {failure.suggestion}")
        lines.append(SUMMARY_SEPARATOR)
        return "\n".join(lines)

    def _stage_counts(self) -> tuple[int, int]:
        attempted = len(self.stats.stages)
        completed = sum(
            1 for stage in self.stats.stages if stage.exit_code == 0 and stage.output_file_exists
        )
        return attempted, completed


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the chapter boundary detection pipeline for a book under books/."
    )
    parser.add_argument(
        "book_name",
        help="Name of the book directory under books/ to process.",
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
    return parser.parse_args()


def ensure_preflight(book_name: str, project_root: Path, logger: PipelineLogger) -> None:
    """Validate required inputs before running any stages.

    Args:
        book_name: Name of the book directory.
        project_root: Repository root directory.
        logger: PipelineLogger for emitting logs.

    Raises:
        PipelineFailure: If validation fails.
    """
    book_dir = project_root / "books" / book_name
    raw_images_dir = book_dir / "raw-images"

    logger.logger.debug(
        "Running preflight checks",
        extra={
            "event": "preflight_start",
            "book_dir": book_dir.as_posix(),
            "raw_images_dir": raw_images_dir.as_posix(),
        },
    )

    if not book_dir.is_dir():
        reason = f"Pre-flight validation failed: missing book directory {book_dir.as_posix()}"
        logger.logger.error(
            reason,
            extra={"event": "preflight_failed", "reason": "book_directory_missing"},
        )
        raise PipelineFailure(
            ExitCode.PREFLIGHT,
            reason=reason,
            stage_label="preflight",
            suggestion="Ensure the book directory exists under books/.",
        )

    if not raw_images_dir.is_dir():
        reason = f"Pre-flight validation failed: missing raw-images directory {raw_images_dir.as_posix()}"
        logger.logger.error(
            reason,
            extra={"event": "preflight_failed", "reason": "raw_images_missing"},
        )
        raise PipelineFailure(
            ExitCode.PREFLIGHT,
            reason=reason,
            stage_label="preflight",
            suggestion="Ensure a raw-images directory exists under the book directory.",
        )

    has_images = any(item.is_file() for item in raw_images_dir.iterdir())
    if not has_images:
        reason = f"Pre-flight validation failed: no files found in {raw_images_dir.as_posix()}"
        logger.logger.error(
            reason,
            extra={"event": "preflight_failed", "reason": "raw_images_empty"},
        )
        raise PipelineFailure(
            ExitCode.PREFLIGHT,
            reason=reason,
            stage_label="preflight",
            suggestion="Place at least one image file in the raw-images directory.",
        )

    logger.logger.info(
        "Preflight validation passed",
        extra={"event": "preflight_complete", "book_dir": book_dir.as_posix()},
    )


def run_stage(
    stage_num: int,
    label: str,
    cmd: Sequence[str],
    expected_output: Path,
    cwd: Path,
    logger: PipelineLogger,
) -> None:
    """Execute a pipeline stage and enforce success criteria.

    Subprocess output is streamed directly to the terminal for real-time feedback and
    is not separately captured by the pipeline logger.

    Args:
        stage_num: Stage number (1-based).
        label: Human-readable stage label.
        cmd: Command to run.
        expected_output: Expected output file path.
        cwd: Working directory for the subprocess.
        logger: PipelineLogger instance.

    Raises:
        PipelineFailure: If the subprocess fails or the expected output is missing.
    """
    logger.stage_start(stage_num, label, cmd, expected_output, cwd)
    result = subprocess.run(cmd, cwd=cwd)

    if result.returncode != 0:
        logger.record_stage_outcome(stage_num, label, result.returncode, expected_output, level=logging.ERROR)
        reason = "Subprocess returned non-zero exit code"
        logger.logger.error(
            reason,
            extra={
                "event": "stage_failed",
                "stage": stage_num,
                "label": label,
                "command": list(cmd),
                "exit_code": result.returncode,
                "expected_output": expected_output.as_posix(),
            },
        )
        raise PipelineFailure(
            ExitCode.for_stage(stage_num),
            reason=reason,
            stage_num=stage_num,
            stage_label=label,
            command=cmd,
            expected_output=expected_output,
            subprocess_exit_code=result.returncode,
            suggestion="Check the stage logs for detailed error information.",
        )

    if not expected_output.is_file():
        logger.record_stage_outcome(stage_num, label, result.returncode, expected_output, level=logging.ERROR)
        reason = "Expected output file was not created."
        logger.logger.error(
            reason,
            extra={
                "event": "stage_failed",
                "stage": stage_num,
                "label": label,
                "command": list(cmd),
                "exit_code": result.returncode,
                "expected_output": expected_output.as_posix(),
            },
        )
        raise PipelineFailure(
            ExitCode.for_stage(stage_num),
            reason=reason,
            stage_num=stage_num,
            stage_label=label,
            command=cmd,
            expected_output=expected_output,
            subprocess_exit_code=result.returncode,
            suggestion="Inspect the stage output paths and rerun after addressing the issue.",
        )

    logger.record_stage_outcome(stage_num, label, result.returncode, expected_output, level=logging.INFO)


def _elapsed_seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start).total_seconds(), 2)


def main() -> None:
    """Entry point for the chapter boundary pipeline."""
    args = parse_args()
    book_name = args.book_name
    log_level = LOG_LEVELS[args.log_level.upper()]
    project_root = Path(__file__).resolve().parent.parent

    final_output = project_root / "books" / book_name / "chapter_boundaries.json"
    input_dir = project_root / "books" / book_name / "raw-images"

    logger = PipelineLogger(
        book_name=book_name,
        input_dir=input_dir,
        final_output=final_output,
        log_level=log_level,
        log_format=args.log_format,
        quiet=args.quiet,
        log_file=args.log_file,
        stats_enabled=args.stats,
    )

    try:
        ensure_preflight(book_name, project_root, logger)

        stage1_output = project_root / "temp" / book_name / "chapter_boundaries_unrefined.json"
        stage2_output = project_root / "temp" / book_name / "chapter_boundaries_refined.json"
        stage3_output = final_output

        stage1_cmd = [
            sys.executable,
            "chapter-boundaries/orchestrator.py",
            book_name,
            "--models",
            "qwen",
            "--batch-size",
            "15",
            "--overlap",
            "5",
            "--head-chars",
            "10000",
            "--tail-chars",
            "0",
            "--max-chars-per-provider",
            "10000",
            "--max-request-tokens",
            "120000",
            "--min-tags-per-page",
            "1",
            "--delay-between-batches",
            "1.5",
            "--aggregation",
            "union",
            "--max-batch-retries",
            "1",
        ]

        stage2_cmd = [
            sys.executable,
            "chapter-boundaries/refining/refiner_orchestrator.py",
            book_name,
            "--debug",
            "--force",
            "--padding",
            "1",
        ]

        stage3_cmd = [
            sys.executable,
            "chapter-boundaries/refining/finalizer.py",
            book_name,
            "--force",
        ]

        run_stage(1, "Initial chapter boundary detection", stage1_cmd, stage1_output, project_root, logger)
        run_stage(2, "Chapter boundary refinement", stage2_cmd, stage2_output, project_root, logger)
        run_stage(3, "Finalization with end padding", stage3_cmd, stage3_output, project_root, logger)

        logger.pipeline_success(stage3_output)
        sys.exit(int(ExitCode.SUCCESS))
    except PipelineFailure as failure:
        logger.pipeline_failure(failure)
        sys.exit(int(failure.exit_code))
    except Exception as exc:  # noqa: BLE001
        logger.unexpected_failure(exc)
        sys.exit(int(ExitCode.INTERNAL))


if __name__ == "__main__":
    main()

