"""
mutant_cache/eval/epsilon.py
----------------------------
ε — the part of a multi-mutant that its singles do not explain (PLAN.md 25).

    ε(AB)  = ΔΔG(AB)  − ΔΔG(A) − ΔΔG(B)
    ε(ABC) = ΔΔG(ABC) − ΔΔG(A) − ΔΔG(B) − ΔΔG(C)

with ΔΔG(X) = score(X) − score(WT).  This is the one evaluation that cannot be
passed by an additive model at any level of skill, which is why DATASETS.md
makes it the point of pulling Tsuboyama separately: "the interaction training
signal — the whole point".

Two cautions, both enforced here rather than left to the caller:

* **ε only means something on an additive scale.**  DATASETS.md: "Unlike
  Tsuboyama, these are fitness assays, so apply a monotone link per assay
  before computing ε. Without it a sigmoid fit will masquerade as learned
  epistasis."  ``epsilon_correlation`` therefore refuses non-dG assay types
  unless ``allow_nonadditive=True`` or a link has been applied.

* **ε is a difference of three or four noisy numbers, so its ceiling is much
  lower than the scores'.**  σ²_ε = Σ σ² over the terms, and the attenuation
  bound ρ_max = sqrt(1 − E[σ²_ε]/Var(ε)) is reported beside every correlation.
  Skipping it makes a ρ of 0.3 on ε look like failure when it can be close to
  the ceiling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from mutant_cache.models.data_compat import parse_mutant

ADDITIVE_ASSAYS = {"dg", "ddg", "dG", "ddG", "stability"}


def _single_key(p: int, ref: str, alt: str) -> str:
    return f"{ref}{p}{alt}"


def build_epsilon_table(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    pred_col: str | None = "pred",
    group_col: str = "score_set_id",
    mutant_col: str = "mutant",
    depth_col: str = "n_mut",
    se_col: str | None = "score_se",
    depths: tuple[int, ...] = (2, 3),
    wt_score: float | dict | None = None,
    require_all_singles: bool = True,
) -> pd.DataFrame:
    """One row per multi-mutant whose constituent singles are all measured.

    ``wt_score`` defaults to the score of the ``mutant == ""`` row in each
    score set, or 0.0 when there is none (i.e. the scores are already ΔΔG).
    """
    out_rows = []
    for ssid, sub in df.groupby(group_col, observed=True, sort=True):
        singles = {}
        for r in sub[sub[depth_col] == 1].itertuples():
            muts = parse_mutant(getattr(r, mutant_col))
            if len(muts) != 1:
                continue
            singles[_single_key(*muts[0])] = r

        if isinstance(wt_score, dict):
            wt = float(wt_score.get(ssid, 0.0))
        elif wt_score is not None:
            wt = float(wt_score)
        else:
            w = sub[sub[mutant_col].fillna("") == ""]
            wt = float(w[score_col].iloc[0]) if len(w) else 0.0
        wt_pred = 0.0
        if pred_col and pred_col in sub.columns:
            w = sub[sub[mutant_col].fillna("") == ""]
            wt_pred = float(w[pred_col].iloc[0]) if len(w) else 0.0

        for r in sub[sub[depth_col].isin(depths)].itertuples():
            muts = parse_mutant(getattr(r, mutant_col))
            keys = [_single_key(*m) for m in muts]
            have = [k for k in keys if k in singles]
            if require_all_singles and len(have) != len(keys):
                continue
            if not have:
                continue
            parts = [singles[k] for k in have]
            ddg = getattr(r, score_col) - wt
            eps = ddg - sum(getattr(s, score_col) - wt for s in parts)
            row = {group_col: ssid, "mutant": getattr(r, mutant_col),
                   depth_col: int(getattr(r, depth_col)),
                   "ddg": float(ddg), "eps_obs": float(eps),
                   "n_singles": len(have)}
            if "target_id" in sub.columns:
                row["target_id"] = getattr(r, "target_id")
            if pred_col and pred_col in sub.columns:
                pddg = getattr(r, pred_col) - wt_pred
                row["eps_pred"] = float(
                    pddg - sum(getattr(s, pred_col) - wt_pred for s in parts))
                row["ddg_pred"] = float(pddg)
            if se_col and se_col in sub.columns:
                v = float(getattr(r, se_col) or 0.0) ** 2
                v += sum(float(getattr(s, se_col) or 0.0) ** 2 for s in parts)
                row["eps_se"] = float(np.sqrt(v))
            out_rows.append(row)
    cols = [group_col, "target_id", "mutant", depth_col, "ddg", "eps_obs",
            "eps_pred", "ddg_pred", "eps_se", "n_singles"]
    out = pd.DataFrame(out_rows)
    return out.reindex(columns=[c for c in cols if c in out.columns])


def attenuation_ceiling(values: np.ndarray, se: np.ndarray) -> float:
    """ρ_max = sqrt(1 − E[σ²]/Var(x)) — the most any predictor can correlate
    with a quantity measured this noisily.  NaN when the noise swamps the
    signal (which is itself the finding)."""
    values = np.asarray(values, dtype=float)
    se = np.asarray(se, dtype=float)
    ok = np.isfinite(values) & np.isfinite(se)
    if ok.sum() < 3:
        return float("nan")
    var = float(np.var(values[ok], ddof=1))
    noise = float(np.mean(se[ok] ** 2))
    if var <= 0 or noise >= var:
        return float("nan")
    return float(np.sqrt(1.0 - noise / var))


def epsilon_correlation(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    pred_col: str = "pred",
    group_col: str = "score_set_id",
    depth_col: str = "n_mut",
    se_col: str | None = "score_se",
    assay_col: str | None = "assay_type",
    depths: tuple[int, ...] = (2, 3),
    allow_nonadditive: bool = False,
    min_n: int = 5,
    wt_score: float | dict | None = None,
    table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Spearman and Pearson of predicted vs observed ε, per score set and depth.

    Columns: ``score_set_id, depth, n, spearman, pearson, eps_ceiling,
    spearman_over_ceiling, eps_sd, eps_noise_sd``, with an ``ALL`` block
    averaging over score sets.
    """
    if assay_col and assay_col in df.columns and not allow_nonadditive:
        bad = sorted({str(a) for a in df[assay_col].dropna().unique()
                      if str(a) not in ADDITIVE_ASSAYS})
        if bad:
            raise ValueError(
                f"ε is only interpretable on an additive scale; assay_type(s) "
                f"{bad} are not ΔG-like. Fit a monotone link per assay first, "
                f"or pass allow_nonadditive=True and label the result as such.")

    tab = table if table is not None else build_epsilon_table(
        df, score_col=score_col, pred_col=pred_col, group_col=group_col,
        depth_col=depth_col, se_col=se_col, depths=depths, wt_score=wt_score)
    if not len(tab) or "eps_pred" not in tab.columns:
        return pd.DataFrame(columns=[group_col, "depth", "n", "spearman",
                                     "pearson", "eps_ceiling",
                                     "spearman_over_ceiling"])

    rows = []
    for (ssid, d), sub in tab.groupby([group_col, depth_col], observed=True,
                                      sort=True):
        if len(sub) < min_n:
            continue
        o, p = sub["eps_obs"].to_numpy(), sub["eps_pred"].to_numpy()
        ceiling = (attenuation_ceiling(o, sub["eps_se"].to_numpy())
                   if "eps_se" in sub.columns else float("nan"))
        rows.append({
            group_col: ssid, "depth": str(int(d)), "n": len(sub),
            "spearman": float(stats.spearmanr(o, p).statistic),
            "pearson": float(stats.pearsonr(o, p).statistic),
            "eps_sd": float(np.std(o, ddof=1)),
            "eps_noise_sd": (float(np.sqrt(np.mean(sub["eps_se"].to_numpy() ** 2)))
                             if "eps_se" in sub.columns else float("nan")),
            "eps_ceiling": ceiling})
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    out["spearman_over_ceiling"] = out["spearman"] / out["eps_ceiling"]
    agg = (out.groupby("depth", observed=True)
              .agg(n=("n", "sum"), spearman=("spearman", "mean"),
                   pearson=("pearson", "mean"), eps_sd=("eps_sd", "mean"),
                   eps_noise_sd=("eps_noise_sd", "mean"),
                   eps_ceiling=("eps_ceiling", "mean"))
              .reset_index())
    agg[group_col] = "ALL"
    agg["spearman_over_ceiling"] = agg["spearman"] / agg["eps_ceiling"]
    return pd.concat([out, agg[out.columns]], ignore_index=True)
