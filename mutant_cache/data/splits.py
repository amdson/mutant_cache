"""Train/val/test splits.

Two kinds, both written under ``$MUTANT_CACHE_SCRATCH/processed/splits/``:

``protein_level.parquet`` — ``target_id, split``. Honours the ProteinGym
    exclusion list (overlapping targets are *absent from the table*, so any
    inner join drops them; they are not quietly parked in test) and keeps
    30%-identity clusters whole, so a homolog of a training protein cannot
    turn up in test.

``flip_gb1_two_vs_rest.parquet`` — ``score_set_id, mutant, split``. FLIP's own
    within-assay depth-extrapolation split, green semaphore, on the
    downsampled rows only. See ``flip_gb1.py`` for why the downsampling
    matters.

Cluster-level assignment is greedy-by-size rather than random: with a handful
of large clusters and a long tail of singletons, random assignment of clusters
misses the requested fractions badly. Greedy ("put the next biggest cluster
wherever we are furthest behind quota") lands within a percent or so.
"""

from __future__ import annotations

import hashlib

import polars as pl

from . import cluster as clust
from .paths import splits_dir
from .schema import (
    SPLIT_PROTEIN_SCHEMA,
    SPLIT_VARIANT_SCHEMA,
    SPLIT_VALUES,
    conform,
    load_targets,
)

DEFAULT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _stable_hash(s: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{s}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def build_protein_split(
    datasets: list[str] | None = None,
    fractions: dict[str, float] | None = None,
    seed: int = 0,
    use_clusters: bool = True,
    exclude: bool = True,
    name: str = "protein_level",
) -> pl.DataFrame:
    datasets = datasets or ["tsuboyama", "mavedb", "flip_gb1"]
    fractions = fractions or DEFAULT_FRACTIONS
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    frames = []
    for ds in datasets:
        try:
            frames.append(load_targets(ds).with_columns(pl.lit(ds).alias("dataset")))
        except FileNotFoundError:
            print(f"  {ds}: not built, skipping")
    if not frames:
        raise RuntimeError("no built datasets")
    targets = pl.concat(frames, how="vertical_relaxed").unique(
        subset=["target_id"], keep="first"
    )
    n_all = targets.height

    n_excluded = 0
    if exclude:
        from .proteingym import load_excluded_targets

        try:
            excluded = load_excluded_targets()
        except FileNotFoundError:
            print("  no exclusion list yet — run python -m mutant_cache.data.proteingym")
            excluded = set()
        n_excluded = targets.filter(pl.col("target_id").is_in(list(excluded))).height
        targets = targets.filter(~pl.col("target_id").is_in(list(excluded)))

    # group -> cluster representative (or the target itself if unclustered)
    group_of: dict[str, str] = {t: t for t in targets["target_id"].to_list()}
    if use_clusters and clust.available() and targets.height > 1:
        records, name_of = [], {}
        for i, r in enumerate(targets.iter_rows(named=True)):
            s = (r["sequence"] or "").strip().upper()
            if len(s) < 10:
                continue
            nm = f"S|{i}"
            name_of[nm] = r["target_id"]
            records.append((nm, s))
        print(f"  clustering {len(records):,} targets at 30% identity")
        mapping = clust.cluster(records, min_seq_id=0.3, coverage=0.5)
        if mapping:
            for nm, rep in mapping.items():
                if nm in name_of:
                    group_of[name_of[nm]] = rep
            print(f"  {len(set(group_of.values())):,} clusters")

    members: dict[str, list[str]] = {}
    for tid, g in group_of.items():
        members.setdefault(g, []).append(tid)

    # Greedy largest-first, deterministic tie-break by hashed cluster id.
    order = sorted(
        members.items(), key=lambda kv: (-len(kv[1]), _stable_hash(kv[0], seed))
    )
    total = sum(len(v) for v in members.values())
    quota = {k: v * total for k, v in fractions.items()}
    filled = {k: 0.0 for k in fractions}
    assignment: dict[str, str] = {}
    for g, mem in order:
        deficit = {k: quota[k] - filled[k] for k in fractions}
        pick = max(deficit, key=lambda k: (deficit[k], -_stable_hash(g + k, seed)))
        filled[pick] += len(mem)
        for tid in mem:
            assignment[tid] = pick

    df = conform(
        pl.DataFrame(
            {
                "target_id": list(assignment),
                "split": [assignment[t] for t in assignment],
            }
        ),
        SPLIT_PROTEIN_SCHEMA,
        "split",
    ).sort("target_id")

    out = splits_dir() / f"{name}.parquet"
    df.write_parquet(out)

    # Persist the clustering this split was built from. mmseqs2's greedy
    # clustering depends on input order, so re-running it does not reproduce
    # the same representatives; without this file "no cluster straddles two
    # splits" is not a checkable claim.
    pl.DataFrame(
        {
            "target_id": list(assignment),
            "cluster_rep": [group_of[t] for t in assignment],
        },
        schema={"target_id": pl.Utf8, "cluster_rep": pl.Utf8},
    ).sort("target_id").write_parquet(splits_dir() / f"{name}_clusters.parquet")
    counts = dict(df.group_by("split").len().iter_rows())
    print(f"  -> {out}")
    print(
        f"  {n_all:,} targets, {n_excluded:,} excluded (ProteinGym overlap), "
        f"{df.height:,} split: "
        + ", ".join(f"{k} {counts.get(k, 0):,}" for k in ("train", "val", "test"))
    )
    return df


def build_flip_gb1_split(split_name: str = "two_vs_rest", val_frac: float = 0.1,
                         seed: int = 0) -> pl.DataFrame:
    """FLIP's green within-assay split, as ``score_set_id, mutant, split``.

    FLIP ships ``*_validation`` columns but for GB1 they are empty, so a
    deterministic hash-based 10% of train is carved out for validation.
    """
    from .flip_gb1 import flip_split_table

    df = flip_split_table(split_name)
    if df.filter(pl.col("split") == "val").height == 0:
        df = df.with_columns(
            pl.when(
                (pl.col("split") == "train")
                & (
                    pl.col("mutant").map_elements(
                        lambda m: _stable_hash(m, seed), return_dtype=pl.Float64
                    )
                    < val_frac
                )
            )
            .then(pl.lit("val"))
            .otherwise(pl.col("split"))
            .alias("split")
        )
    bad = set(df["split"].unique()) - SPLIT_VALUES
    if bad:
        raise ValueError(f"unexpected split labels {bad}")

    df = conform(df, SPLIT_VARIANT_SCHEMA, "split")
    out = splits_dir() / f"flip_gb1_{split_name}.parquet"
    df.write_parquet(out)
    counts = dict(df.group_by("split").len().iter_rows())
    print(f"  -> {out}")
    print("  " + ", ".join(f"{k} {counts.get(k, 0):,}" for k in ("train", "val", "test")))
    return df


if __name__ == "__main__":
    print("protein-level split")
    build_protein_split()
    print("FLIP GB1 2-vs-rest")
    build_flip_gb1_split("two_vs_rest")
