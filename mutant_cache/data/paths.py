"""Where raw downloads and processed parquet live.

Everything is under ``$MUTANT_CACHE_SCRATCH`` (default ``~/mutant_cache_scratch``).
Nothing in this module ever writes inside the repo.
"""

from __future__ import annotations

import os
from pathlib import Path


def scratch() -> Path:
    return Path(
        os.environ.get("MUTANT_CACHE_SCRATCH", Path.home() / "mutant_cache_scratch")
    ).expanduser()


def raw_dir(dataset: str) -> Path:
    return scratch() / "raw" / dataset


def processed_dir(dataset: str) -> Path:
    d = scratch() / "processed" / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def splits_dir() -> Path:
    d = scratch() / "processed" / "splits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def exclusion_dir() -> Path:
    d = scratch() / "processed" / "exclusion"
    d.mkdir(parents=True, exist_ok=True)
    return d


def interim_dir(dataset: str) -> Path:
    """Unzipped raw trees; large, re-creatable, not a deliverable."""
    d = scratch() / "interim" / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


DATASETS = ("tsuboyama", "mavedb", "flip_gb1")
