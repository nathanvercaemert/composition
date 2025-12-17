#!/usr/bin/env python3
"""Thin wrapper around batch-commentary-extraction's submit_batch without sys.path side effects."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_EXTRACTION_BATCH = Path(__file__).resolve().parent.parent.parent / "batch-commentary-extraction"
_EXTRACTION_SUBMIT_BATCH = _EXTRACTION_BATCH / "submit_batch.py"


def _load_extraction_submit_batch() -> ModuleType:
    """Load the extraction submit_batch module without mutating sys.path."""
    spec = importlib.util.spec_from_file_location(
        "batch_commentary_extraction.submit_batch", _EXTRACTION_SUBMIT_BATCH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load submit_batch from {_EXTRACTION_SUBMIT_BATCH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_extraction_submit_batch = _load_extraction_submit_batch()

submit_batch = _extraction_submit_batch.submit_batch
main = _extraction_submit_batch.main

__all__ = ["submit_batch", "main"]


if __name__ == "__main__":
    main()

