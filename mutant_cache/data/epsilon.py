"""Epistasis (epsilon) table for Tsuboyama doubles.

For a double mutant with substitutions *a* and *b* in one domain,

    epsilon = ddG(ab) - ddG(a) - ddG(b)

computed only where both singles were measured in the *same* score set (and
therefore the same domain, same assay, same wild-type reference), so no
cross-assay comparison sneaks in.

**Why the uncertainties are carried.** Epsilon is a difference of three noisy
numbers, so its measurement error is larger than any single ddG's. With
``se_eps = sqrt(se_ab^2 + se_a^2 + se_b^2)``, the fraction of the observed
variance in epsilon that is real rather than noise is

    signal_var = var(eps) - mean(se_eps^2)

and the best correlation any predictor could achieve against the measured
epsilon is ``r_max = sqrt(signal_var / var(eps))``. Reporting a correlation on
epsilon without that number beside it is not interpretable: epsilon values are
small and the noise on them is not.

Output: ``$MUTANT_CACHE_SCRATCH/processed/tsuboyama/epsilon.parquet``

| column | note |
|---|---|
| score_set_id, target_id | as in the canonical tables |
| mutant | the double, canonical form |
| mut_a, mut_b | the two singles, canonical form |
| pos_a, pos_b | 1-based positions |
| seq_sep | abs(pos_b - pos_a) |
| ddg_double, ddg_a, ddg_b | kcal/mol |
| ddg_additive | ddg_a + ddg_b, the additive prediction |
| epsilon | ddg_double - ddg_additive |
| dg_wt, dg_double | absolute folding stability, for the saturation filter |
| se_double, se_a, se_b, se_epsilon | propagated standard errors |
| z_epsilon | epsilon / se_epsilon |

**Read this before using epsilon as a training target.** Mean epsilon is
**+0.89 kcal/mol**, not ~0, and it varies monotonically with the additive
prediction: +3.25 where ``ddg_additive ~ -7``, +0.01 where it is ~0,
-0.69 where it is ~+2. ``corr(epsilon, ddg_additive) = -0.72``.

That is overwhelmingly the proteolysis assay's finite dynamic range, not
protein biology. ΔG is resolvable over roughly 0-5 kcal/mol; a double whose
true stability falls off the bottom reads as less destabilizing than additive,
which shows up as positive epsilon. It is *not* hard clipping — only 0.45% of
doubles sit near the extreme ΔΔG value — it is smooth compression as the
variant approaches the unfolded baseline, and it survives restricting to the
unsaturated range (mean still +0.63 for ``ddg_additive > -4``).

So DATASETS.md's framing — Tsuboyama is the clean case, fitness assays are the
ones needing a monotone link — is only half right. Tsuboyama needs one too; it
is simply a much milder and better-characterized one, because ΔG is a real
physical quantity with a known measurement window. Fit a per-domain monotone
link, or restrict to the unsaturated window, before calling any of this
epsilon "learned epistasis". ``dg_wt`` and ``dg_double`` are in the table so
that filter can be applied without going back to the raw CSV.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .mutations import parse_mutant
from .paths import processed_dir
from .schema import load_variants

DATASET = "tsuboyama"


def _wt_dg() -> dict[str, float]:
    """Absolute wild-type dG_ML per domain, for the saturation filter."""
    from .tsuboyama import load_raw

    raw = load_raw().filter(pl.col("mut_type") == "wt")
    agg = raw.group_by("WT_name").agg(pl.col("dG_ML").mean())
    return {k: v for k, v in agg.iter_rows() if v is not None}


def build(dataset: str = DATASET) -> pl.DataFrame:
    v = load_variants(dataset)

    singles = v.filter(pl.col("n_mut") == 1).select(
        "score_set_id",
        pl.col("mutant").alias("single"),
        pl.col("score").alias("ddg_single"),
        pl.col("score_se").alias("se_single"),
    )
    # A domain can report the same single more than once; average.
    singles = singles.group_by(["score_set_id", "single"]).agg(
        pl.col("ddg_single").mean(),
        # averaging k independent measurements shrinks the error by sqrt(k)
        (pl.col("se_single").pow(2).sum().sqrt() / pl.len()).alias("se_single"),
    )

    wt_dg = _wt_dg()
    doubles = v.filter(pl.col("n_mut") == 2)
    if doubles.height == 0:
        raise RuntimeError(f"{dataset} has no double mutants")

    parsed = [parse_mutant(m) for m in doubles["mutant"].to_list()]
    doubles = doubles.with_columns(
        pl.Series("mut_a", [f"{s[0][1]}{s[0][0]}{s[0][2]}" for s in parsed]),
        pl.Series("mut_b", [f"{s[1][1]}{s[1][0]}{s[1][2]}" for s in parsed]),
        pl.Series("pos_a", [s[0][0] for s in parsed], dtype=pl.Int32),
        pl.Series("pos_b", [s[1][0] for s in parsed], dtype=pl.Int32),
    )

    df = (
        doubles.join(
            singles.rename({"single": "mut_a", "ddg_single": "ddg_a", "se_single": "se_a"}),
            on=["score_set_id", "mut_a"],
            how="inner",
        )
        .join(
            singles.rename({"single": "mut_b", "ddg_single": "ddg_b", "se_single": "se_b"}),
            on=["score_set_id", "mut_b"],
            how="inner",
        )
        .with_columns(
            (pl.col("ddg_a") + pl.col("ddg_b")).alias("ddg_additive"),
            (pl.col("pos_b") - pl.col("pos_a")).abs().alias("seq_sep"),
            pl.col("target_id")
            .replace_strict(wt_dg, default=None, return_dtype=pl.Float64)
            .alias("dg_wt"),
            (
                pl.col("score_se").fill_null(0.0) ** 2
                + pl.col("se_a").fill_null(0.0) ** 2
                + pl.col("se_b").fill_null(0.0) ** 2
            )
            .sqrt()
            .alias("se_epsilon"),
        )
        .with_columns(
            (pl.col("score") - pl.col("ddg_additive")).alias("epsilon"),
            (pl.col("dg_wt") + pl.col("score")).alias("dg_double"),
        )
        .with_columns(
            (pl.col("epsilon") / pl.col("se_epsilon")).alias("z_epsilon")
        )
        .select(
            "score_set_id", "target_id", "mutant", "mut_a", "mut_b",
            "pos_a", "pos_b", "seq_sep",
            pl.col("score").alias("ddg_double"), "ddg_a", "ddg_b",
            "ddg_additive", "epsilon", "dg_wt", "dg_double",
            pl.col("score_se").alias("se_double"), "se_a", "se_b",
            "se_epsilon", "z_epsilon",
        )
        .sort(["score_set_id", "mutant"])
    )

    out = processed_dir(dataset) / "epsilon.parquet"
    df.write_parquet(out)
    print(f"  -> {out}")
    print(summary(df, n_doubles_total=doubles.height))
    return df


def noise_ceiling(df: pl.DataFrame) -> dict[str, float]:
    """Variance decomposition of epsilon and the implied max correlation."""
    eps = df["epsilon"].to_numpy()
    se = df["se_epsilon"].to_numpy()
    ok = np.isfinite(eps) & np.isfinite(se)
    eps, se = eps[ok], se[ok]
    var_total = float(np.var(eps, ddof=1))
    var_noise = float(np.mean(se**2))
    var_signal = max(var_total - var_noise, 0.0)
    return {
        "n": int(eps.size),
        "mean": float(np.mean(eps)),
        "sd": float(np.sqrt(var_total)),
        "median_abs": float(np.median(np.abs(eps))),
        "p01": float(np.percentile(eps, 1)),
        "p99": float(np.percentile(eps, 99)),
        "mean_se": float(np.mean(se)),
        "median_se": float(np.median(se)),
        "var_total": var_total,
        "var_noise": var_noise,
        "var_signal": var_signal,
        "signal_frac": var_signal / var_total if var_total else 0.0,
        "r_max": float(np.sqrt(var_signal / var_total)) if var_total else 0.0,
        "frac_abs_z_gt_2": float(np.mean(np.abs(eps / se) > 2)),
    }


def summary(df: pl.DataFrame, n_doubles_total: int | None = None) -> str:
    nc = noise_ceiling(df)
    head = (
        f"  epsilon: {nc['n']:,} doubles with both singles measured"
        + (f" (of {n_doubles_total:,} doubles)" if n_doubles_total else "")
        + f", {df['target_id'].n_unique()} domains"
    )
    return (
        head
        + f"\n  spread: mean {nc['mean']:+.3f}, sd {nc['sd']:.3f} kcal/mol, "
          f"median |eps| {nc['median_abs']:.3f}, 1-99% [{nc['p01']:+.2f}, {nc['p99']:+.2f}]"
        + f"\n  noise:  mean se(eps) {nc['mean_se']:.3f}, median {nc['median_se']:.3f}"
        + f"\n  ceiling: var total {nc['var_total']:.4f} = signal {nc['var_signal']:.4f} "
          f"+ noise {nc['var_noise']:.4f}; signal frac {nc['signal_frac']:.3f}, "
          f"r_max {nc['r_max']:.3f}"
        + f"\n  {100 * nc['frac_abs_z_gt_2']:.1f}% of doubles have |epsilon| > 2 se"
        + "\n" + additivity_bias(df)
    )


def additivity_bias(df: pl.DataFrame) -> str:
    """How much of epsilon is the assay's dynamic range rather than coupling."""
    r = df.select(pl.corr("epsilon", "ddg_additive")).item()
    unsat = df.filter(pl.col("ddg_additive") > -4.0)
    lines = [
        f"  BIAS: mean epsilon {df['epsilon'].mean():+.3f} (not 0); "
        f"corr(epsilon, ddg_additive) = {r:+.3f}",
        "  epsilon vs additive prediction (dynamic-range compression, see module docstring):",
    ]
    binned = (
        df.with_columns(pl.col("ddg_additive").round(0).clip(-7, 2).alias("bin"))
        .group_by("bin")
        .agg(pl.len().alias("n"), pl.col("epsilon").mean().alias("mean_eps"))
        .sort("bin")
    )
    lines.append(
        "    "
        + "  ".join(
            f"{int(b):+d}:{m:+.2f}" for b, _n, m in binned.iter_rows()
        )
    )
    if unsat.height:
        nu = noise_ceiling(unsat)
        lines.append(
            f"  unsaturated subset (ddg_additive > -4): n {nu['n']:,}, "
            f"mean {nu['mean']:+.3f}, sd {nu['sd']:.3f}, r_max {nu['r_max']:.3f}"
        )
    return "\n".join(lines)


def load(dataset: str = DATASET) -> pl.DataFrame:
    p = processed_dir(dataset) / "epsilon.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} — python -m mutant_cache.data.epsilon")
    return pl.read_parquet(p)


if __name__ == "__main__":
    build()
