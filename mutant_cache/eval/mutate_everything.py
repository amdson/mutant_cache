"""Matched-row evaluation using Mutate Everything's metrics.py @ 034dd91.

Canonical inputs: target_id, mutant, n_mut, score, pred (positive stabilizing).
Pooled regression; protein-macro detection, threshold score >= 0.5, linear
rectified gain with sklearn's default tie averaging. Proteins without a
threshold-positive variant are excluded from detection, as upstream does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from .megascale import per_protein_metrics, pooled_metrics, summarize
from mutant_cache.data.mutations import format_mutant, parse_mutant

KEYS = ["target_id", "mutant"]


def canonical_predictions(frame):
    d = frame.copy()
    required = {*KEYS, "score", "pred", "n_mut"}
    if required - set(d):
        raise ValueError(f"missing prediction columns {sorted(required-set(d))}")
    if d[list(required)].isna().any().any():
        raise ValueError("null prediction keys or values")
    d["target_id"] = d.target_id.astype(str)
    d["mutant"] = d.mutant.map(lambda m: format_mutant(parse_mutant(m.upper())))
    if not np.isfinite(d[["score", "pred"]].to_numpy(float)).all():
        raise ValueError("nonfinite predictions or labels")
    if d.duplicated(KEYS).any():
        raise ValueError("duplicate prediction keys within a seed")
    depth = d.mutant.map(lambda m: len(parse_mutant(m)))
    if not depth.eq(d.n_mut).all() or (depth < 1).any():
        raise ValueError("mutation depth disagrees with n_mut")
    return d


def join_predictions(ours, theirs):
    a, b = canonical_predictions(ours), canonical_predictions(theirs)
    j = a.merge(b, on=KEYS, how="outer", suffixes=("_ours", "_theirs"),
                indicator=True, validate="one_to_one")
    unmatched = j.loc[j._merge != "both"].copy()
    matched = j.loc[j._merge == "both"].copy()
    if not np.allclose(matched.score_ours, matched.score_theirs, atol=1e-6, rtol=0):
        raise ValueError("joined labels disagree (check delta-delta-G sign and duplicate aggregation)")
    if not matched.n_mut_ours.eq(matched.n_mut_theirs).all():
        raise ValueError("joined mutation depths disagree")
    matched = matched.rename(columns={"score_ours": "score", "n_mut_ours": "n_mut"})
    return matched, unmatched


def detection(df, pred_col="pred", *, k=30, threshold=0.5):
    if k < 1:
        raise ValueError("k must be positive")
    rows = []
    for target, g in df.groupby("target_id", sort=True):
        # Match upstream sort_values before sklearn tie handling and precision.
        g = g.sort_values(pred_col, ascending=False)
        eligible = len(g) > 1 and bool((g.score >= threshold).any())
        kk = min(k, len(g))
        rows.append(dict(target_id=target, n=len(g), eligible=eligible,
            ndcg=float(ndcg_score(np.maximum(g.score.to_numpy(), 0)[None],
                                  g[pred_col].to_numpy()[None], k=k)) if eligible else np.nan,
            precision=float((g.score.iloc[:kk] >= threshold).mean()) if eligible else np.nan))
    return pd.DataFrame(rows, columns=["target_id", "n", "eligible", "ndcg", "precision"])


def evaluate(df, pred_col="pred"):
    tables, detections = [], []
    pp = per_protein_metrics(df, pred_col, min_n=2, depths=("single", "double", "multi", "all"))
    macro = summarize(pp)
    for depth, sub in (("all", df[df.n_mut > 0]), ("single", df[df.n_mut == 1]),
                       ("double", df[df.n_mut == 2]), ("multi", df[df.n_mut > 2])):
        if sub.empty:
            continue
        det = detection(sub, pred_col).assign(depth=depth)
        detections.append(det)
        pooled = pooled_metrics(sub, pred_col, depths=("all",)).iloc[0].to_dict()
        pooled["depth"] = depth
        pooled.update(ndcg30=det.ndcg.mean(), precision30=det.precision.mean(),
                      detection_proteins=int(det.eligible.sum()), total_proteins=len(det))
        tables.append(pooled)
    return {"table": pd.DataFrame(tables), "per_protein": pp, "macro": macro,
            "detection": pd.concat(detections, ignore_index=True) if detections else pd.DataFrame()}


def reproduction_gate(actual, expected, *, atol=0.005):
    """Expected is {depth: {metric: value}} from a pinned published source.

    No expected numbers means no verified reproduction, never an implicit pass.
    """
    checks = []
    for depth, metrics in expected.items():
        row = actual[actual.depth == depth]
        for metric, value in metrics.items():
            observed = float(row.iloc[0][metric]) if len(row) and metric in row else np.nan
            checks.append(dict(depth=depth, metric=metric, expected=float(value),
                               actual=observed, passed=bool(np.isfinite(observed) and
                               abs(observed-float(value)) <= atol)))
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks,
            "atol": atol}


def paired_seed_bootstrap(per_seed, *, n_boot=10000, seed=0):
    """Cluster bootstrap of protein differences, averaging paired seeds first.

    Input seed,target_id,difference. Missing/duplicate seed-protein cells fail;
    seeds are repeated fits, not independent protein replicates.
    """
    if per_seed.duplicated(["seed", "target_id"]).any():
        raise ValueError("duplicate seed/protein cells")
    matrix = per_seed.pivot(index="target_id", columns="seed", values="difference")
    if matrix.empty or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("paired bootstrap needs complete finite seed/protein cells")
    v = matrix.mean(axis=1).to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(mean_diff=float(v.mean()), lo=float(lo), hi=float(hi),
                n_proteins=len(v), n_seeds=len(matrix.columns),
                three_seed_requirement_met=len(matrix.columns) >= 3)
