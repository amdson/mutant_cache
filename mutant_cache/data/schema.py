"""Canonical parquet schema (INTERFACES.md section 2) and its enforcement.

Three tables per dataset, under ``$MUTANT_CACHE_SCRATCH/processed/<dataset>/``:
``targets.parquet``, ``variants.parquet``, ``score_sets.parquet``.

Every loader goes through :func:`write_dataset`, so the column set, the dtypes
and the rank normalization are defined exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .paths import processed_dir

# --- column contracts -------------------------------------------------------

TARGETS_SCHEMA: dict[str, pl.DataType] = {
    "target_id": pl.Utf8,
    "sequence": pl.Utf8,
    "uniprot": pl.Utf8,
    "source": pl.Utf8,
}

VARIANTS_SCHEMA: dict[str, pl.DataType] = {
    "score_set_id": pl.Utf8,
    "target_id": pl.Utf8,
    "mutant": pl.Utf8,
    "n_mut": pl.Int32,
    "score": pl.Float64,
    "score_rank": pl.Float64,
    "score_se": pl.Float64,
    "assay_type": pl.Utf8,
}

SCORE_SETS_SCHEMA: dict[str, pl.DataType] = {
    "score_set_id": pl.Utf8,
    "target_id": pl.Utf8,
    "n_variants": pl.Int64,
    "max_n_mut": pl.Int32,
    "assay_type": pl.Utf8,
    "reliability": pl.Float64,
    "license": pl.Utf8,
}

SPLIT_PROTEIN_SCHEMA: dict[str, pl.DataType] = {
    "target_id": pl.Utf8,
    "split": pl.Utf8,
}

SPLIT_VARIANT_SCHEMA: dict[str, pl.DataType] = {
    "score_set_id": pl.Utf8,
    "mutant": pl.Utf8,
    "split": pl.Utf8,
}

# Columns that may never be null.
TARGETS_REQUIRED = ("target_id", "sequence", "source")
VARIANTS_REQUIRED = ("score_set_id", "target_id", "mutant", "n_mut", "score", "score_rank")
SCORE_SETS_REQUIRED = ("score_set_id", "target_id", "n_variants", "max_n_mut")

SPLIT_VALUES = {"train", "val", "test"}


class SchemaError(ValueError):
    pass


def conform(df: pl.DataFrame, schema: dict[str, pl.DataType], name: str) -> pl.DataFrame:
    """Add missing nullable columns, drop extras, cast, and order as declared."""
    missing_required = [
        c
        for c in schema
        if c not in df.columns
        and c in {
            "targets": TARGETS_REQUIRED,
            "variants": VARIANTS_REQUIRED,
            "score_sets": SCORE_SETS_REQUIRED,
        }.get(name, ())
    ]
    if missing_required:
        raise SchemaError(f"{name}: missing required columns {missing_required}")
    out = df
    for col, dtype in schema.items():
        if col not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=dtype).alias(col))
    out = out.select([pl.col(c).cast(dtype) for c, dtype in schema.items()])
    return out


def validate(df: pl.DataFrame, schema: dict, required: tuple, name: str) -> None:
    if list(df.columns) != list(schema):
        raise SchemaError(f"{name}: columns {df.columns} != {list(schema)}")
    for col, dtype in schema.items():
        if df.schema[col] != dtype:
            raise SchemaError(f"{name}.{col}: dtype {df.schema[col]} != {dtype}")
    for col in required:
        n_null = df[col].null_count()
        if n_null:
            raise SchemaError(f"{name}.{col}: {n_null} nulls in a required column")
    if name == "variants":
        nan = df.filter(pl.col("score").is_nan() | pl.col("score_rank").is_nan()).height
        if nan:
            raise SchemaError(f"variants: {nan} NaN scores")


# --- rank normalization -----------------------------------------------------


def rank_normalize(df: pl.DataFrame, by: str = "score_set_id", col: str = "score") -> pl.DataFrame:
    """Add ``score_rank`` in (0, 1): average rank within group, divided by n+1.

    Ties take the average rank, so tied scores get tied ranks. Dividing by
    ``n + 1`` rather than ``n`` keeps the result strictly inside (0, 1), which
    matters if anything downstream wants to logit-transform it.
    """
    return df.with_columns(
        (
            pl.col(col).rank(method="average").over(by)
            / (pl.len().over(by) + 1)
        ).alias("score_rank")
    )


# --- writing ----------------------------------------------------------------


def write_wt_report(
    dataset: str, targets: pl.DataFrame, variants: pl.DataFrame
) -> pl.DataFrame:
    """Per-target WT-mismatch audit -> ``<dataset>/wt_report.parquet``.

    A single global mismatch rate hides the thing you need to know: whether
    the errors are spread thinly over every target (a parsing bug) or
    concentrated in a few deposits whose numbering is simply shifted (a data
    problem you can see and decide about). This table makes that visible per
    target, together with the integer offset that would fix it.

    Nothing here is applied to ``variants.parquet``. ``suggested_offset`` is a
    finding, not a correction.
    """
    from .mutations import parse_mutant, suggest_offsets

    seqs = dict(zip(targets["target_id"], targets["sequence"]))
    n_sub: dict[str, int] = {t: 0 for t in seqs}
    n_bad: dict[str, int] = {t: 0 for t in seqs}
    n_oor: dict[str, int] = {t: 0 for t in seqs}
    n_var: dict[str, int] = {t: 0 for t in seqs}
    for mut, tid in zip(variants["mutant"].to_list(), variants["target_id"].to_list()):
        seq = seqs.get(tid, "")
        n_var[tid] = n_var.get(tid, 0) + 1
        try:
            subs = parse_mutant(mut)
        except ValueError:
            n_bad[tid] = n_bad.get(tid, 0) + 1
            continue
        for pos, wt, _alt in subs:
            n_sub[tid] += 1
            if pos > len(seq):
                n_oor[tid] += 1
                n_bad[tid] += 1
            elif seq[pos - 1] != wt:
                n_bad[tid] += 1

    offsets = suggest_offsets(
        variants["mutant"].to_list(), variants["target_id"].to_list(), seqs
    )
    tids = list(seqs)
    df = pl.DataFrame(
        {
            "target_id": tids,
            "n_variants": [n_var.get(t, 0) for t in tids],
            "n_sub": [n_sub[t] for t in tids],
            "n_sub_mismatch": [n_bad[t] for t in tids],
            "n_out_of_range": [n_oor[t] for t in tids],
            "mismatch_rate": [
                (n_bad[t] / n_sub[t]) if n_sub[t] else 0.0 for t in tids
            ],
            "suggested_offset": [
                offsets[t][0] if t in offsets else None for t in tids
            ],
            "rate_after_offset": [
                offsets[t][2] if t in offsets else None for t in tids
            ],
        },
        schema={
            "target_id": pl.Utf8, "n_variants": pl.Int64, "n_sub": pl.Int64,
            "n_sub_mismatch": pl.Int64, "n_out_of_range": pl.Int64,
            "mismatch_rate": pl.Float64, "suggested_offset": pl.Int32,
            "rate_after_offset": pl.Float64,
        },
    ).sort("n_sub_mismatch", descending=True)
    df.write_parquet(processed_dir(dataset) / "wt_report.parquet")
    return df


@dataclass
class BuildReport:
    dataset: str
    n_targets: int
    n_variants: int
    n_score_sets: int
    depth_counts: dict[int, int]
    notes: list[str]

    def __str__(self) -> str:
        depth = ", ".join(f"{k}:{v:,}" for k, v in sorted(self.depth_counts.items()))
        return (
            f"[{self.dataset}] targets={self.n_targets:,} score_sets={self.n_score_sets:,} "
            f"variants={self.n_variants:,}\n  depth {depth}"
            + ("".join(f"\n  {n}" for n in self.notes) if self.notes else "")
        )


def write_dataset(
    dataset: str,
    targets: pl.DataFrame,
    variants: pl.DataFrame,
    score_sets: pl.DataFrame | None = None,
    notes: list[str] | None = None,
) -> BuildReport:
    """Conform, validate, derive score_sets if absent, and write the parquet."""
    targets = conform(targets, TARGETS_SCHEMA, "targets").unique(subset=["target_id"], keep="first")
    variants = conform(variants, VARIANTS_SCHEMA, "variants")

    unknown = set(variants["target_id"].unique()) - set(targets["target_id"])
    if unknown:
        raise SchemaError(
            f"{dataset}: {len(unknown)} variant target_ids absent from targets, "
            f"e.g. {sorted(unknown)[:3]}"
        )

    if score_sets is None:
        score_sets = (
            variants.group_by("score_set_id")
            .agg(
                pl.col("target_id").first(),
                pl.len().alias("n_variants"),
                pl.col("n_mut").max().alias("max_n_mut"),
                pl.col("assay_type").first(),
            )
            .sort("score_set_id")
        )
    score_sets = conform(score_sets, SCORE_SETS_SCHEMA, "score_sets")

    validate(targets, TARGETS_SCHEMA, TARGETS_REQUIRED, "targets")
    validate(variants, VARIANTS_SCHEMA, VARIANTS_REQUIRED, "variants")
    validate(score_sets, SCORE_SETS_SCHEMA, SCORE_SETS_REQUIRED, "score_sets")

    out = processed_dir(dataset)
    targets.write_parquet(out / "targets.parquet")
    variants.write_parquet(out / "variants.parquet")
    score_sets.write_parquet(out / "score_sets.parquet")

    wt = write_wt_report(dataset, targets, variants)
    worst = wt.filter(pl.col("n_sub_mismatch") > 0).sort("n_sub_mismatch", descending=True)
    notes = list(notes or [])
    if worst.height:
        tot_sub = int(wt["n_sub"].sum())
        tot_bad = int(wt["n_sub_mismatch"].sum())
        fixable = worst.filter(pl.col("suggested_offset").is_not_null())
        notes.append(
            f"wt_report.parquet: {worst.height} of {wt.height} targets have a WT "
            f"mismatch; {tot_bad:,}/{tot_sub:,} substitutions ({100 * tot_bad / tot_sub:.2f}%)"
        )
        if fixable.height:
            share = int(fixable["n_sub_mismatch"].sum())
            notes.append(
                f"  {fixable.height} of them are explained by a constant numbering "
                f"offset and hold {share:,} ({100 * share / tot_bad:.1f}%) of all "
                f"mismatching substitutions — reported, NOT applied"
            )

    depth = dict(
        variants.group_by("n_mut").len().sort("n_mut").iter_rows()
    )
    return BuildReport(
        dataset=dataset,
        n_targets=targets.height,
        n_variants=variants.height,
        n_score_sets=score_sets.height,
        depth_counts=depth,
        notes=notes,
    )


# --- reading ----------------------------------------------------------------


def _read(dataset: str, name: str) -> pl.DataFrame:
    p = processed_dir(dataset) / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} — build it first (python -m mutant_cache.data.build {dataset})")
    return pl.read_parquet(p)


def load_targets(dataset: str) -> pl.DataFrame:
    return _read(dataset, "targets")


def load_variants(dataset: str) -> pl.DataFrame:
    return _read(dataset, "variants")


def load_score_sets(dataset: str) -> pl.DataFrame:
    return _read(dataset, "score_sets")


def load_split(name: str) -> pl.DataFrame:
    from .paths import splits_dir

    p: Path = splits_dir() / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} — build it first (python -m mutant_cache.data.splits)")
    return pl.read_parquet(p)
