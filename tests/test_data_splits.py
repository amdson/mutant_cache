"""Tests on the real built split and exclusion files."""

from __future__ import annotations

import polars as pl
import pytest

from mutant_cache.data import paths
from mutant_cache.data.schema import (
    SPLIT_PROTEIN_SCHEMA,
    SPLIT_VALUES,
    SPLIT_VARIANT_SCHEMA,
)

PROTEIN_SPLIT = "protein_level"
FLIP_SPLIT = "flip_gb1_two_vs_rest"


def _split(name: str) -> pl.DataFrame:
    p = paths.splits_dir() / f"{name}.parquet"
    if not p.exists():
        pytest.skip(f"{name}.parquet not built")
    return pl.read_parquet(p)


def _overlap() -> pl.DataFrame:
    p = paths.exclusion_dir() / "proteingym_overlap.parquet"
    if not p.exists():
        pytest.skip("proteingym_overlap.parquet not built")
    return pl.read_parquet(p)


# --- protein-level ----------------------------------------------------------


def test_protein_split_schema():
    s = _split(PROTEIN_SPLIT)
    assert list(s.columns) == list(SPLIT_PROTEIN_SCHEMA)
    for c, dt in SPLIT_PROTEIN_SCHEMA.items():
        assert s.schema[c] == dt
    assert s["target_id"].null_count() == 0
    assert s["split"].null_count() == 0


def test_protein_split_labels_are_valid():
    s = _split(PROTEIN_SPLIT)
    assert set(s["split"].unique()) <= SPLIT_VALUES


def test_protein_split_is_disjoint():
    """One row per target: a target cannot be in two splits."""
    s = _split(PROTEIN_SPLIT)
    assert s["target_id"].n_unique() == s.height, "a target_id appears twice"
    groups = {
        k: set(v["target_id"])
        for k, v in s.partition_by("split", as_dict=True).items()
    }
    keys = list(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            assert not (groups[a] & groups[b]), f"{a} and {b} share targets"


def test_protein_split_has_all_three_parts():
    s = _split(PROTEIN_SPLIT)
    counts = dict(s.group_by("split").len().iter_rows())
    for part in ("train", "val", "test"):
        assert counts.get(part, 0) > 0, f"empty {part} split"


def test_protein_split_honours_exclusion_list():
    s = _split(PROTEIN_SPLIT)
    excluded = set(_overlap()["target_id"])
    leaked = set(s["target_id"]) & excluded
    assert not leaked, (
        f"{len(leaked)} ProteinGym-overlapping targets are in the split, "
        f"e.g. {sorted(leaked)[:3]}"
    )


def test_protein_split_keeps_identity_clusters_whole():
    """No 30%-identity cluster may straddle two splits.

    This is the test that actually protects the held-out numbers: a homolog
    of a training protein sitting in test leaks just as badly as the protein
    itself.

    Checked against the clustering the split was *built* from, which
    ``build_protein_split`` persists alongside it. Re-running mmseqs2 here
    would not be a valid check: its greedy clustering depends on input order,
    so a fresh run returns different representatives and the comparison would
    fail for reasons that have nothing to do with the split.
    """
    s = _split(PROTEIN_SPLIT)
    p = paths.splits_dir() / f"{PROTEIN_SPLIT}_clusters.parquet"
    if not p.exists():
        pytest.skip(f"{PROTEIN_SPLIT}_clusters.parquet not built")
    clusters = pl.read_parquet(p)

    assert set(clusters["target_id"]) == set(s["target_id"]), (
        "the persisted clustering does not cover exactly the split's targets"
    )
    straddling = (
        s.join(clusters, on="target_id")
        .group_by("cluster_rep")
        .agg(pl.col("split").n_unique().alias("n_splits"))
        .filter(pl.col("n_splits") > 1)
    )
    assert straddling.height == 0, (
        f"{straddling.height} clusters span more than one split, "
        f"e.g. {straddling.head(3)['cluster_rep'].to_list()}"
    )


# --- FLIP GB1 within-assay --------------------------------------------------


def test_flip_split_schema():
    s = _split(FLIP_SPLIT)
    assert list(s.columns) == list(SPLIT_VARIANT_SCHEMA)
    for c, dt in SPLIT_VARIANT_SCHEMA.items():
        assert s.schema[c] == dt
    assert set(s["split"].unique()) <= SPLIT_VALUES


def test_flip_split_is_disjoint():
    s = _split(FLIP_SPLIT)
    key = s.select(["score_set_id", "mutant"])
    assert key.n_unique() == s.height, "a (score_set_id, mutant) pair appears twice"


def test_flip_split_is_depth_extrapolation():
    """2-vs-rest means train is depth <= 2 and test is depth >= 3."""
    from mutant_cache.data.mutations import n_mut

    s = _split(FLIP_SPLIT).with_columns(
        pl.col("mutant")
        .map_elements(n_mut, return_dtype=pl.Int32)
        .alias("n_mut")
    )
    tr = s.filter(pl.col("split").is_in(["train", "val"]))
    te = s.filter(pl.col("split") == "test")
    assert tr["n_mut"].max() <= 2, "train contains a mutant deeper than 2"
    assert te["n_mut"].min() >= 3, "test contains a mutant shallower than 3"


def test_flip_split_matches_the_downsampled_set():
    """FLIP downsamples non-functional sequences; the split must match it.

    8,733 of 149,361 rows carry keep=True, and the published split CSVs hold
    exactly those. A split built on the full landscape would not be
    comparable to the FLIP leaderboard.
    """
    s = _split(FLIP_SPLIT)
    assert s.height == 8733, f"expected FLIP's 8,733 downsampled rows, got {s.height:,}"
    assert set(s["score_set_id"].unique()) == {"flip_gb1_four_flip"}


def test_flip_split_variants_exist_in_the_score_set():
    from mutant_cache.data.schema import load_variants

    p = paths.processed_dir("flip_gb1") / "variants.parquet"
    if not p.exists():
        pytest.skip("flip_gb1 not built")
    s = _split(FLIP_SPLIT)
    v = load_variants("flip_gb1").filter(
        pl.col("score_set_id") == "flip_gb1_four_flip"
    )
    missing = set(s["mutant"]) - set(v["mutant"])
    assert not missing, f"{len(missing)} split mutants absent from variants.parquet"


# --- exclusion list ---------------------------------------------------------


def test_overlap_table_shape():
    o = _overlap()
    assert set(o.columns) == {
        "target_id", "dataset", "overlap_type", "proteingym_dms_id", "identity"
    }
    assert o["target_id"].null_count() == 0
    assert o["overlap_type"].null_count() == 0
    assert set(o["overlap_type"].unique()) <= {
        "exact", "substring", "uniprot", "mmseqs30"
    }


def test_overlap_targets_are_real_targets():
    from mutant_cache.data.schema import load_targets

    o = _overlap()
    known: set[str] = set()
    for ds in ("tsuboyama", "mavedb", "flip_gb1"):
        if (paths.processed_dir(ds) / "targets.parquet").exists():
            known |= set(load_targets(ds)["target_id"])
    unknown = set(o["target_id"]) - known
    assert not unknown, f"{len(unknown)} excluded ids are not targets"


def test_tsuboyama_exact_overlap_matches_the_published_count():
    """DATASETS.md: ProteinGym's curated Tsuboyama subset is 64 proteins.

    The exact-sequence pass should recover that number; if it drifts, either
    the ProteinGym reference file changed or the Tsuboyama loader did.
    """
    o = _overlap()
    if "tsuboyama" not in set(o["dataset"]):
        pytest.skip("tsuboyama not screened")
    n = (
        o.filter((pl.col("dataset") == "tsuboyama") & (pl.col("overlap_type") == "exact"))
        ["target_id"].n_unique()
    )
    assert n == 64, f"expected 64 exact Tsuboyama/ProteinGym matches, got {n}"
