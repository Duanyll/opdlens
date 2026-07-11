"""Thin row loaders that sit *underneath* the behavior-rich benchmarks.

A benchmark (``opdlens/benchmarks``) owns prompt construction and grading; it
delegates raw row access to the loaders here. These return plain ``dict`` rows —
no behavior — exactly the thin convention fc's ``datasets`` package uses.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

from .utils.logging import get_logger

logger = get_logger(__name__)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a ``.jsonl`` file into a list of dict rows (one JSON object per line)."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    logger.info("Loaded %d rows from %s", len(rows), path)
    return rows


def read_hf(
    hf_id: str,
    split: str,
    *,
    name: str | None = None,
    revision: str | None = None,
) -> Sequence[dict[str, Any]]:
    """Load a HuggingFace dataset split. The returned object is indexable and
    iterates dict rows, matching the ``read_jsonl`` contract."""
    from datasets import load_dataset

    ds = load_dataset(hf_id, name=name, split=split, revision=revision)
    logger.info(
        "Loaded HF dataset %s (name=%s, split=%s): %d rows", hf_id, name, split, len(ds)
    )
    return cast("Sequence[dict[str, Any]]", ds)
