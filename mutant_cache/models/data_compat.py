"""
mutant_cache/models/data_compat.py
----------------------------------
A thin shim for the two helpers INTERFACES.md §2 promises from
``mutant_cache.data``::

    from mutant_cache.data import load_variants, parse_mutant

The data layer is being written in parallel, so everything in this package
imports those names *through here*: the real implementation is used the moment
it exists, and the fallback below keeps the model/eval/baseline code runnable
and testable until then.

Delete this module (and the imports of it) once ``mutant_cache.data`` lands.
"""
from __future__ import annotations

import re
from pathlib import Path

_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z*])$")

try:  # the real thing, once the data agent lands it
    from mutant_cache.data import parse_mutant  # type: ignore
    HAVE_REAL_DATA = True
except Exception:  # pragma: no cover - exercised only before the data layer lands
    HAVE_REAL_DATA = False

    def parse_mutant(mutant: str) -> list[tuple[int, str, str]]:
        """``"A23G:K45R"`` -> ``[(23, "A", "G"), (45, "K", "R")]``, 1-based.

        ``""`` (wild type) parses to the empty list.
        """
        if mutant is None:
            return []
        mutant = str(mutant).strip()
        if not mutant or mutant.lower() in {"wt", "_wt", "nan"}:
            return []
        out = []
        for tok in mutant.split(":"):
            m = _MUT_RE.match(tok.strip())
            if not m:
                raise ValueError(f"unparseable substitution {tok!r} in {mutant!r}")
            out.append((int(m.group(2)), m.group(1), m.group(3)))
        return out


try:
    from mutant_cache.data import load_variants  # type: ignore
except Exception:  # pragma: no cover

    def load_variants(dataset_or_dir, *, split=None, **kw):
        """Read ``variants.parquet`` from a processed dataset directory.

        The real loader takes a dataset name and resolves
        ``$MUTANT_CACHE_SCRATCH/processed/<dataset>/``; this fallback also
        accepts that directory directly, which is what the tests use.
        """
        import os
        import pandas as pd

        p = Path(dataset_or_dir)
        if not p.exists():
            root = Path(os.environ.get("MUTANT_CACHE_SCRATCH",
                                       Path.home() / "mutant_cache_scratch"))
            p = root / "processed" / str(dataset_or_dir)
        df = pd.read_parquet(p / "variants.parquet")
        if split is not None:
            sp = pd.read_parquet(p / "splits" / f"{split}.parquet")
            keys = [c for c in ("target_id", "score_set_id", "mutant")
                    if c in sp.columns and c in df.columns]
            df = df.merge(sp[keys + ["split"]], on=keys, how="inner")
        return df


def load_tables(processed_dir: str | Path) -> dict:
    """targets / variants / score_sets as a dict of DataFrames (§2 layout)."""
    import pandas as pd

    p = Path(processed_dir)
    out = {"variants": pd.read_parquet(p / "variants.parquet")}
    for name in ("targets", "score_sets"):
        f = p / f"{name}.parquet"
        if f.exists():
            out[name] = pd.read_parquet(f)
    return out


def load_split(processed_dir: str | Path, name: str):
    import pandas as pd
    return pd.read_parquet(Path(processed_dir) / "splits" / f"{name}.parquet")


def apply_split(variants, split_df, which: str):
    """Join a §2 split table onto variants and keep one partition.

    Handles both split flavours: protein-level (``target_id, split``) and
    within-assay (``score_set_id, mutant, split``).
    """
    keys = [c for c in ("target_id", "score_set_id", "mutant")
            if c in split_df.columns and c in variants.columns]
    if not keys:
        raise ValueError("split table shares no key column with variants")
    merged = variants.merge(split_df[keys + ["split"]], on=keys, how="inner")
    return merged[merged["split"] == which].drop(columns=["split"]).reset_index(drop=True)
