#!/usr/bin/env python3

"""
Chapter boundary detection pipeline.

Orchestrates the complete workflow: detection → refinement → finalization.

Any stage failure terminates the pipeline immediately.

Usage:
    python pipeline/chapter_boundary_pipeline.py BOOK_NAME
"""

import argparse
import subprocess
import sys
from pathlib import Path

SEPARATOR = "=" * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the chapter boundary detection pipeline for a book under books/."
    )
    parser.add_argument(
        "book_name",
        help="Name of the book directory under books/ to process.",
    )
    return parser.parse_args()


def ensure_preflight(book_name: str, project_root: Path) -> None:
    book_dir = project_root / "books" / book_name
    raw_images_dir = book_dir / "raw-images"

    if not book_dir.is_dir():
        print(f"Pre-flight validation failed: missing book directory {book_dir.as_posix()}")
        print("Pipeline aborted.")
        sys.exit(1)

    if not raw_images_dir.is_dir():
        print(f"Pre-flight validation failed: missing raw-images directory {raw_images_dir.as_posix()}")
        print("Pipeline aborted.")
        sys.exit(1)

    has_images = any(item.is_file() for item in raw_images_dir.iterdir())
    if not has_images:
        print(f"Pre-flight validation failed: no files found in {raw_images_dir.as_posix()}")
        print("Pipeline aborted.")
        sys.exit(1)


def stage_error(
    stage_num: int,
    cmd: list[str],
    returncode: int | None = None,
    expected_output: Path | None = None,
    extra: str | None = None,
) -> None:
    print(f"\nERROR: Stage {stage_num} failed.")
    print(f"Command: {' '.join(cmd)}")
    if returncode is not None:
        print(f"Return code: {returncode}")
    if expected_output is not None:
        print(f"Expected output: {expected_output.as_posix()}")
    if extra:
        print(extra)
    print("Pipeline aborted.")
    sys.exit(1)


def run_stage(stage_num: int, label: str, cmd: list[str], expected_output: Path, cwd: Path) -> None:
    print("\n" + SEPARATOR)
    print(f"Stage {stage_num}: {label}")
    print(f"Command: {' '.join(cmd)}")
    print(SEPARATOR)

    result = subprocess.run(cmd, cwd=cwd)

    if result.returncode != 0:
        stage_error(stage_num, cmd, returncode=result.returncode, expected_output=expected_output)

    if not expected_output.is_file():
        stage_error(
            stage_num,
            cmd,
            returncode=result.returncode,
            expected_output=expected_output,
            extra="Expected output file was not created.",
        )


def main() -> None:
    args = parse_args()
    book_name = args.book_name
    project_root = Path(__file__).resolve().parent.parent

    ensure_preflight(book_name, project_root)

    stage1_output = project_root / "temp" / book_name / "chapter_boundaries_unrefined.json"
    stage2_output = project_root / "temp" / book_name / "chapter_boundaries_refined.json"
    stage3_output = project_root / "books" / book_name / "chapter_boundaries.json"

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

    run_stage(1, "Initial chapter boundary detection", stage1_cmd, stage1_output, project_root)
    run_stage(2, "Chapter boundary refinement", stage2_cmd, stage2_output, project_root)
    run_stage(3, "Finalization with end padding", stage3_cmd, stage3_output, project_root)

    print("\nAll stages completed successfully.")
    print(f"Final output: {stage3_output.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()

