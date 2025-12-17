#!/usr/bin/env python3
"""Load batch-commentary-extraction's poll_batch without import shadowing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_EXTRACTION_BATCH = Path(__file__).resolve().parent.parent.parent / "batch-commentary-extraction"
_EXTRACTION_POLL_BATCH = _EXTRACTION_BATCH / "poll_batch.py"


def _load_extraction_poll_batch() -> ModuleType:
    """Load the extraction poll_batch module without mutating sys.path."""
    spec = importlib.util.spec_from_file_location(
        "batch_commentary_extraction.poll_batch", _EXTRACTION_POLL_BATCH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load poll_batch from {_EXTRACTION_POLL_BATCH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_extraction_poll_batch = _load_extraction_poll_batch()

poll_batch = _extraction_poll_batch.poll_batch
main = _extraction_poll_batch.main

__all__ = ["poll_batch", "main"]


if __name__ == "__main__":
    main()

