#!/usr/bin/env python3
"""Thin wrapper around batch-commentary-extraction's retrieve_batch without sys.path side effects."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_EXTRACTION_BATCH = Path(__file__).resolve().parent.parent.parent / "batch-commentary-extraction"
_EXTRACTION_RETRIEVE_BATCH = _EXTRACTION_BATCH / "retrieve_batch.py"


def _load_extraction_retrieve_batch() -> ModuleType:
    """Load the extraction retrieve_batch module without mutating sys.path."""
    spec = importlib.util.spec_from_file_location(
        "batch_commentary_extraction.retrieve_batch", _EXTRACTION_RETRIEVE_BATCH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load retrieve_batch from {_EXTRACTION_RETRIEVE_BATCH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_extraction_retrieve_batch = _load_extraction_retrieve_batch()

retrieve_batch_outputs = _extraction_retrieve_batch.retrieve_batch_outputs
main = _extraction_retrieve_batch.main

__all__ = ["retrieve_batch_outputs", "main"]


if __name__ == "__main__":
    main()

