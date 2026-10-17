"""
mutant_cache/eval/depth.py
--------------------------
Depth-stratified Spearman (INTERFACES.md §3, PLAN.md 24).

    depth_spearman(df, pred_col, score_col) -> DataFrame

one row per (score_set_id, depth bucket) with buckets 1 / 2 / 3 / 4 / 5+ and
``all``, plus the ``reliability`` ceiling carried through wherever the data
layer knew it.

Why the ceiling rides along in the same frame rather than being pasted on
later: DATASETS.md is explicit that "0.6 against a test-retest of 0.7 reads
nothing like 0.6 against 0.95", and the only way that stays true in practice is
if it is impossible to produce a correlation table from this module without
also producing its ceiling column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DEPTH_BUCKETS = ("1", "2", "3", "4", "5+")
ALL = "all"


def depth_bucket(n_mut) -> np.ndarray:
    """1/2/3/4 verbatim, everything >= 5 collapsed to '5+' (ProteinGym's
    stratification, so the columns line up with the published table)."""
    n = np.asarray(n_mut, dtype=int)
    out = np.where(n >= 5, "5+", n.astype(str))
    return out


def _spearman(x, y) -> float:
    if len(x) < 3:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    r = stats.spearmanr(x, y).statistic
    return float(r)


def depth_spearman(
    df: pd.DataFrame,
    pred_col: str = "pred",
    score_col: str = "score",
    *,
    group_col: str = "score_set_id",
    depth_col: str = "n_mut",
    reliability_col: str = "reliability",
    min_n: int = 3,
    include_overall: bool = True,
) -> pd.DataFrame:
    """Per score set, Spearman at each depth bucket and overall.

    Columns: ``score_set_id, depth, n, spearman, reliability,
    spearman_over_ceiling``.  ``depth == "all"`` is the pooled row for that
    score set; with ``include_overall`` a final ``score_set_id == "ALL"`` block
    gives the *mean over score sets* at each depth — the number a paper
    reports, and a mean of per-assay correlations rather than a correlation of
    pooled scores, which would be dominated by between-assay scale.
    """
    need = {pred_col, score_col, depth_col}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"depth_spearman needs columns {sorted(missing)}")
    if group_col not in df.columns:
        df = df.assign(**{group_col: "ALL_VARIANTS"})

    rel = {}
    if reliability_col in df.columns:
        rel = (df.groupby(group_col, observed=True)[reliability_col]
                 .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
                 .to_dict())

    work = df.copy()
    work["_depth"] = depth_bucket(work[depth_col])

    rows = []
    for ssid, sub in work.groupby(group_col, observed=True, sort=True):
        for b in DEPTH_BUCKETS:
            s = sub[sub["_depth"] == b]
            if len(s) < min_n:
                continue
            rows.append({group_col: ssid, "depth": b, "n": len(s),
                         "spearman": _spearman(s[pred_col].to_numpy(),
                                               s[score_col].to_numpy())})
        rows.append({group_col: ssid, "depth": ALL, "n": len(sub),
                     "spearman": _spearman(sub[pred_col].to_numpy(),
                                           sub[score_col].to_numpy())})

    out = pd.DataFrame(rows, columns=[group_col, "depth", "n", "spearman"])
    out["reliability"] = out[group_col].map(rel).astype(float) if rel else np.nan
    out["spearman_over_ceiling"] = out["spearman"] / out["reliability"]

    if include_overall and len(out):
        agg = (out.groupby("depth", observed=True)
                  .agg(n=("n", "sum"),
                       spearman=("spearman", "mean"),
                       reliability=("reliability", "mean"),
                       n_score_sets=("spearman", "count"))
                  .reset_index())
        agg[group_col] = "ALL"
        agg["spearman_over_ceiling"] = agg["spearman"] / agg["reliability"]
        out = pd.concat([out, agg[out.columns.tolist() + ["n_score_sets"]]],
                        ignore_index=True)
    # stable, readable ordering: per-set blocks first, the ALL block last
    order = {b: i for i, b in enumerate(DEPTH_BUCKETS + (ALL,))}
    out["_o"] = out["depth"].map(order)
    out = (out.assign(_is_all=(out[group_col] == "ALL"))
              .sort_values(["_is_all", group_col, "_o"])
              .drop(columns=["_o", "_is_all"]).reset_index(drop=True))
    return out


def depth_summary(table: pd.DataFrame, group_col: str = "score_set_id") -> dict:
    """The ALL block as a flat dict, for the run record."""
    a = table[table[group_col] == "ALL"]
    return {f"spearman_depth_{r.depth}": float(r.spearman) for r in a.itertuples()}
