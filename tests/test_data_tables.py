"""Tests against the real built parquet under $MUTANT_CACHE_SCRATCH.

These are deliberately not unit tests on synthetic frames: the point is to
catch a loader that wrote something the model layer cannot consume. Datasets
that have not been built are skipped rather than failed, so the suite is
useful before everything is downloaded.

    python -m mutant_cache.data.build       # then
    pytest tests/test_data_tables.py -v
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from mutant_cache.data import paths
from mutant_cache.data.mutations import parse_mutant
from mutant_cache.data.schema import (
    SCORE_SETS_REQUIRED,
    SCORE_SETS_SCHEMA,
    TARGETS_REQUIRED,
    TARGETS_SCHEMA,
    VARIANTS_REQUIRED,
    VARIANTS_SCHEMA,
    validate,
)

DATASETS = ["tsuboyama", "mavedb", "flip_gb1"]


def _load(dataset: str, name: str) -> pl.DataFrame:
    p = paths.processed_dir(dataset) / f"{name}.parquet"
    if not p.exists():
        pytest.skip(f"{dataset}/{name}.parquet not built")
    return pl.read_parquet(p)


@pytest.fixture(scope="module", params=DATASETS)
def dataset(request):
    return request.param


# --- schema conformance -----------------------------------------------------


def test_targets_schema(dataset):
    validate(_load(dataset, "targets"), TARGETS_SCHEMA, TARGETS_REQUIRED, "targets")


def test_variants_schema(dataset):
    validate(_load(dataset, "variants"), VARIANTS_SCHEMA, VARIANTS_REQUIRED, "variants")


def test_score_sets_schema(dataset):
    validate(
        _load(dataset, "score_sets"), SCORE_SETS_SCHEMA, SCORE_SETS_REQUIRED, "score_sets"
    )


def test_no_nan_in_required_columns(dataset):
    v = _load(dataset, "variants")
    for col in ("score", "score_rank"):
        bad = v.filter(pl.col(col).is_nan() | pl.col(col).is_infinite()).height
        assert bad == 0, f"{dataset}.variants.{col}: {bad} NaN/inf values"
    for col in VARIANTS_REQUIRED:
        assert v[col].null_count() == 0, f"{dataset}.variants.{col} has nulls"


def test_targets_are_amino_acids(dataset):
    t = _load(dataset, "targets")
    allowed = set("ACDEFGHIKLMNPQRSTVWYX")
    offenders = []
    for tid, seq in zip(t["target_id"], t["sequence"]):
        bad = set(seq) - allowed
        if bad:
            offenders.append((tid, sorted(bad)[:5]))
    assert not offenders[:5], f"{dataset}: non-amino-acid letters, e.g. {offenders[:5]}"


# --- internal consistency ---------------------------------------------------


def test_n_mut_matches_mutant_string(dataset):
    v = _load(dataset, "variants")
    sample = v.sample(min(v.height, 200_000), seed=0)
    mismatched = [
        (m, n)
        for m, n in zip(sample["mutant"], sample["n_mut"])
        if len(parse_mutant(m)) != n
    ]
    assert not mismatched[:5], f"{dataset}: n_mut != parsed depth, e.g. {mismatched[:5]}"


def test_mutant_positions_are_sorted_and_unique(dataset):
    v = _load(dataset, "variants").filter(pl.col("n_mut") > 1)
    if v.height == 0:
        pytest.skip("no multi-mutants")
    bad = []
    for m in v.sample(min(v.height, 100_000), seed=0)["mutant"]:
        pos = [p for p, _w, _a in parse_mutant(m)]
        if pos != sorted(pos) or len(set(pos)) != len(pos):
            bad.append(m)
    assert not bad[:5], f"{dataset}: unsorted/duplicate positions, e.g. {bad[:5]}"


def test_variant_targets_exist(dataset):
    v = _load(dataset, "variants")
    t = _load(dataset, "targets")
    missing = set(v["target_id"].unique()) - set(t["target_id"])
    assert not missing, f"{dataset}: {len(missing)} variant target_ids not in targets"


def test_score_sets_cover_variants(dataset):
    v = _load(dataset, "variants")
    s = _load(dataset, "score_sets")
    assert set(v["score_set_id"].unique()) == set(s["score_set_id"])
    assert s["score_set_id"].n_unique() == s.height, "duplicate score_set_id rows"
    counted = (
        v.group_by("score_set_id")
        .agg(pl.len().alias("n"), pl.col("n_mut").max().alias("mx"))
        .join(s, on="score_set_id")
    )
    assert counted.filter(pl.col("n") != pl.col("n_variants")).height == 0
    assert counted.filter(pl.col("mx") != pl.col("max_n_mut")).height == 0


def test_rank_is_normalized_within_score_set(dataset):
    v = _load(dataset, "variants")
    assert v["score_rank"].min() > 0.0
    assert v["score_rank"].max() < 1.0
    # Rank must be a monotone function of score inside each score set.
    big = (
        v.group_by("score_set_id")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(5)["score_set_id"]
        .to_list()
    )
    for ssid in big:
        sub = v.filter(pl.col("score_set_id") == ssid)
        if sub.height < 3 or sub["score"].n_unique() < 3:
            continue
        r = sub.select(pl.corr("score", "score_rank", method="spearman")).item()
        assert r is None or r > 0.999, f"{dataset}/{ssid}: rank not monotone in score ({r})"


def test_wt_residues_match_target_sequence(dataset):
    """Tsuboyama and FLIP must be exact; MaveDB's errors must stay confined.

    A global substitution-mismatch rate is the wrong assertion for MaveDB:
    one deposit (PSD95 PDZ3, 1.24M substitutions, numbered one off from its
    own target sequence) moves it to ~13.7% on its own, which says nothing
    about whether the loader is correct. What must hold is that mismatches
    stay confined to a handful of *targets* -- a parsing bug would spread
    them across the whole dump.
    """
    from mutant_cache.data.mutations import verify_wt

    v = _load(dataset, "variants")
    t = _load(dataset, "targets")
    seqs = dict(zip(t["target_id"], t["sequence"]))
    sample = v.sample(min(v.height, 300_000), seed=0)
    chk = verify_wt(sample["mutant"].to_list(), sample["target_id"].to_list(), seqs)
    print(f"\n  {dataset} WT check: {chk}")

    if dataset in ("tsuboyama", "flip_gb1"):
        assert chk.sub_mismatch_rate == 0.0, (
            f"{dataset} must be exact, got {chk.sub_mismatch_rate:.4%}; "
            f"examples {chk.examples[:3]}"
        )
        return

    n_targets = t.height
    frac_bad_targets = len(chk.per_target) / n_targets
    assert frac_bad_targets <= 0.02, (
        f"{dataset}: {len(chk.per_target)} of {n_targets} targets "
        f"({frac_bad_targets:.2%}) have a WT mismatch; examples {chk.examples[:3]}"
    )


def test_wt_report_explains_the_mismatches(dataset):
    """The per-target audit must exist and account for what mismatches remain.

    Every badly-mismatching target (>10%) has to be either explained by a
    constant numbering offset or small enough not to matter. An unexplained
    large mismatch means the loader is placing mutations on the wrong
    sequence, which is the failure this whole audit exists to catch.
    """
    p = paths.processed_dir(dataset) / "wt_report.parquet"
    if not p.exists():
        pytest.skip(f"{dataset}/wt_report.parquet not built")
    w = pl.read_parquet(p)
    assert w["target_id"].n_unique() == w.height
    assert (w["n_sub_mismatch"] <= w["n_sub"]).all()

    unexplained = w.filter(
        (pl.col("mismatch_rate") > 0.10)
        & (pl.col("n_sub") > 100)
        & (
            pl.col("suggested_offset").is_null()
            | (pl.col("rate_after_offset") > 0.10)
        )
    )
    tot = int(w["n_sub"].sum())
    share = int(unexplained["n_sub_mismatch"].sum()) / max(tot, 1)
    print(
        f"\n  {dataset}: {unexplained.height} unexplained bad targets, "
        f"{share:.4%} of all substitutions"
    )
    assert share <= 0.01, (
        f"{dataset}: {unexplained.height} targets mismatch badly with no offset "
        f"explanation, covering {share:.2%} of substitutions: "
        f"{unexplained.head(3)['target_id'].to_list()}"
    )


# --- epsilon ----------------------------------------------------------------


def test_epsilon_table():
    p = paths.processed_dir("tsuboyama") / "epsilon.parquet"
    if not p.exists():
        pytest.skip("epsilon.parquet not built")
    e = pl.read_parquet(p)
    assert e.height > 0
    for col in ("epsilon", "se_epsilon", "ddg_double", "ddg_a", "ddg_b", "ddg_additive"):
        assert e[col].null_count() == 0, f"epsilon.{col} has nulls"
        assert e.filter(pl.col(col).is_nan()).height == 0, f"epsilon.{col} has NaN"
    # epsilon is exactly the three-term difference
    resid = (
        e.select(
            (pl.col("epsilon") - (pl.col("ddg_double") - pl.col("ddg_a") - pl.col("ddg_b")))
            .abs()
            .max()
        ).item()
    )
    assert resid < 1e-9, f"epsilon is not ddg_double - ddg_a - ddg_b (max resid {resid})"
    # se must be at least as large as any of its parts
    assert e.filter(pl.col("se_epsilon") < pl.col("se_double")).height == 0
    assert (e["se_epsilon"] > 0).all()
    # the two singles are distinct positions of the same double
    s = e.sample(min(e.height, 20_000), seed=0)
    for mut, a, b, pa, pb in zip(s["mutant"], s["mut_a"], s["mut_b"], s["pos_a"], s["pos_b"]):
        assert mut == f"{a}:{b}", f"{mut} != {a}:{b}"
        assert pa != pb
    assert (e["seq_sep"] > 0).all()


def test_epsilon_noise_ceiling_is_reported():
    p = paths.processed_dir("tsuboyama") / "epsilon.parquet"
    if not p.exists():
        pytest.skip("epsilon.parquet not built")
    from mutant_cache.data.epsilon import noise_ceiling

    nc = noise_ceiling(pl.read_parquet(p))
    assert nc["n"] > 0
    assert 0.0 <= nc["r_max"] <= 1.0
    assert math.isclose(
        nc["var_total"], nc["var_signal"] + nc["var_noise"], rel_tol=1e-6
    ) or nc["var_signal"] == 0.0
