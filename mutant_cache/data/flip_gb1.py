"""FLIP GB1 four-site landscape -> canonical tables.

Source: the FLIP repo's ``splits/gb1`` directory. DATASETS.md points at
``data.bioembeddings.com/public/FLIP`` for the raw data; that host 404s for
every path tried on 2026-10-16, and the GB1 data is committed in the repo
anyway, so ``datasets/Makefile`` pulls it from GitHub.

Two quirks DATASETS.md warns about, both handled here:

* **Downsampling.** Over 96% of the 149,361 variants are non-functional, and
  FLIP downsamples them. The flag is the ``keep`` column in
  ``four_mutations_full_data.csv``: 8,733 of 149,361 rows. The published split
  CSVs contain exactly those 8,733 rows, so ``keep`` *is* the downsampling, and
  matching it is the difference between a comparable number and a made-up one.
* **Semaphore.** Only green splits are exported: ``one_vs_rest``,
  ``two_vs_rest``, ``three_vs_rest``, ``low_vs_high``. ``sampled`` is orange
  and is deliberately not written.

Both the full landscape and the downsampled subset are written as separate
score sets, so nothing is thrown away but the default comparison is the
FLIP-comparable one.

Mutated sites are V39, D40, G41, V54, 1-based on the 265-residue 5LDE chain-A
sequence that FLIP uses as the reference.
"""

from __future__ import annotations

import zipfile

import polars as pl

from .mutations import format_mutant, verify_wt
from .paths import interim_dir, raw_dir
from .schema import BuildReport, rank_normalize, write_dataset

DATASET = "flip_gb1"
TARGET_ID = "FLIP_GB1_5LDE_1"
SITES = (39, 40, 41, 54)  # 1-based, V/D/G/V
GREEN_SPLITS = ("one_vs_rest", "two_vs_rest", "three_vs_rest", "low_vs_high")


def _extract() -> tuple[str, str]:
    raw = raw_dir(DATASET)
    work = interim_dir(DATASET)
    for z in ("four_mutations_full_data.csv.zip", "splits.zip"):
        with zipfile.ZipFile(raw / z) as zf:
            zf.extractall(work)
    return work / "four_mutations_full_data.csv", work


def build() -> BuildReport:
    csv_path, _work = _extract()
    d = pl.read_csv(csv_path)

    wt_seq = d.filter(pl.col("HD") == 0)["sequence"][0]
    observed = "".join(wt_seq[p - 1] for p in SITES)
    if observed != "VDGV":
        raise ValueError(f"expected VDGV at sites {SITES}, found {observed}")

    # Variants column is the 4-letter combination at the 4 sites, so the
    # mutant string is a direct read-off; no alignment needed.
    def mutant_of(combo: str) -> str:
        subs = [
            (pos, wt_seq[pos - 1], alt)
            for pos, alt in zip(SITES, combo)
            if wt_seq[pos - 1] != alt
        ]
        return format_mutant(subs)

    d = d.with_columns(
        pl.col("Variants")
        .map_elements(mutant_of, return_dtype=pl.Utf8)
        .alias("mutant")
    )

    # Sanity: the mutant string must reconstruct the sequence FLIP ships.
    from .mutations import mutate

    recon_bad = sum(
        1
        for mut, seq in zip(d["mutant"].to_list(), d["sequence"].to_list())
        if mutate(wt_seq, mut) != seq
    )
    if recon_bad:
        raise ValueError(f"{recon_bad} rows where the mutant string does not rebuild FLIP's sequence")

    targets = pl.DataFrame(
        {
            "target_id": [TARGET_ID],
            "sequence": [wt_seq],
            "uniprot": [None],
            "source": ["flip_gb1"],
        }
    )

    frames = []
    for score_set_id, sel in (
        ("flip_gb1_four_full", pl.lit(True)),
        ("flip_gb1_four_flip", pl.col("keep")),
    ):
        sub = d.filter(sel)
        frames.append(
            sub.select(
                pl.lit(score_set_id).alias("score_set_id"),
                pl.lit(TARGET_ID).alias("target_id"),
                pl.col("mutant"),
                pl.col("HD").cast(pl.Int32).alias("n_mut"),
                pl.col("Fitness").cast(pl.Float64).alias("score"),
                pl.lit(None, dtype=pl.Float64).alias("score_se"),
                pl.lit("binding").alias("assay_type"),
            )
        )
    variants = rank_normalize(pl.concat(frames))

    chk = verify_wt(
        variants["mutant"].to_list(),
        variants["target_id"].to_list(),
        {TARGET_ID: wt_seq},
    )

    score_sets = (
        variants.group_by("score_set_id")
        .agg(
            pl.col("target_id").first(),
            pl.len().alias("n_variants"),
            pl.col("n_mut").max().alias("max_n_mut"),
            pl.col("assay_type").first(),
        )
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("reliability"),
            pl.lit("AFL-3 (FLIP) / CC-BY-4.0 (eLife source)").alias("license"),
        )
        .sort("score_set_id")
    )

    notes = [
        f"WT check: {chk}",
        f"downsampling: keep=True is {d['keep'].sum():,} of {d.height:,} rows "
        f"(score_set flip_gb1_four_flip); full landscape kept as flip_gb1_four_full",
        f"green splits available: {', '.join(GREEN_SPLITS)} (sampled is orange, not exported)",
    ]
    return write_dataset(DATASET, targets, variants, score_sets, notes)


def flip_split_table(split_name: str = "two_vs_rest") -> pl.DataFrame:
    """FLIP's own train/test assignment, as ``score_set_id, mutant, split``.

    Only the downsampled score set gets a split, because that is the only one
    FLIP defines. ``*_validation`` rows inside ``train`` become ``val``.
    """
    if split_name not in GREEN_SPLITS:
        raise ValueError(
            f"{split_name!r} is not a green split; allowed: {GREEN_SPLITS}"
        )
    csv_path, _ = _extract()
    d = pl.read_csv(csv_path).filter(pl.col("keep"))
    wt_seq = pl.read_csv(csv_path).filter(pl.col("HD") == 0)["sequence"][0]

    def mutant_of(combo: str) -> str:
        return format_mutant(
            [
                (pos, wt_seq[pos - 1], alt)
                for pos, alt in zip(SITES, combo)
                if wt_seq[pos - 1] != alt
            ]
        )

    val_col = f"{split_name}_validation"
    return (
        d.with_columns(
            pl.col("Variants").map_elements(mutant_of, return_dtype=pl.Utf8).alias("mutant")
        )
        .with_columns(
            pl.when(pl.col(val_col).cast(pl.Utf8).str.to_lowercase() == "true")
            .then(pl.lit("val"))
            .otherwise(pl.col(split_name))
            .alias("split")
        )
        .select(
            pl.lit("flip_gb1_four_flip").alias("score_set_id"),
            pl.col("mutant"),
            pl.col("split"),
        )
        .drop_nulls("split")
    )


if __name__ == "__main__":
    print(build())
