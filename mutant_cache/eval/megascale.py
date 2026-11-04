"""
mutant_cache/eval/megascale.py
------------------------------
Held-out-protein stability metrics for the ESM-MSR Megascale experiment
(TODO_ESM_MSR_EXPERIMENT.md §4).

Every metric is computed **per protein** and summarized as a macro mean over
proteins (primary), with median and pooled numbers as labelled secondary
summaries. Thousands of variants of one protein are not independent
replicates, so uncertainty comes from a protein bootstrap.

Inputs are a frame with ``target_id, mutant, n_mut, score`` (raw ΔΔG, kcal/mol,
WT-relative) and one or more prediction columns (WT-relative d(x); a
zero-anchored model gives d(WT) = 0 by construction).

* ``per_protein_metrics`` — Spearman / Pearson / MAE / RMSE per protein and
  depth. Correlations are NaN (undefined, not zero) below ``min_n`` rows or on
  a constant observed or predicted vector; error metrics need only ``min_n``.
  Depth ``all`` pools every non-WT row of a protein: its Spearman is ESM-MSR's
  headline ``rho`` (the per-protein term of ``val_rho_combined_avg``,
  src/esm_msr/stats.py:compute_metrics @ c34c315). ESM-MSR's ``safe_spearman``
  needs only 2 rows; ``min_n=ESM_MSR_MIN_N`` matches it, and ``min_n_drops``
  lists which proteins each rule leaves undefined.
* ``epistasis_metrics`` — on complete A/B/AB groups within one protein:
  ε_obs = ΔΔG(AB) − ΔΔG(A) − ΔΔG(B), ε_pred = d(AB) − d(A) − d(B) with the
  model's *predicted* singles. RMSE/MAE of ε_pred vs ε_obs per protein, and
  Spearman where ε_pred varies (an additive model has ε_pred ≡ 0: its residual
  error is meaningful, its correlation is undefined).
* ``summarize`` — macro / median / pooled over proteins, with counts of
  proteins that had a defined value.
* ``retrieval_metrics`` — stabilizing-variant retrieval per protein, ESM-MSR's
  definition (src/esm_msr/stats.py:11-84 @ c34c315): relevance = ΔΔG rectified
  at 0 (positive = stabilizing), linear gain, sklearn ``ndcg_score`` (log2
  discount, ``ignore_ties=True``) at k = min(96, n); NaN for a protein with no
  ΔΔG > 0 row (it leaves the macro mean, and the count says so). Precision and
  recall at the same k are the interpretable companions. ``ndcg_t0`` is their
  second variant, ``safe_ndcg_t0`` (``threshold=0.0``, k = None: the full
  ranking).
* ``paired_protein_bootstrap`` — mean per-protein difference between two
  predictors on identical eligible proteins, with a percentile interval.
* ``selection_key`` — checkpoint-selection criterion; ``SELECTION_RULES``
  describes each rule for the run record.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from mutant_cache.eval.epsilon import build_epsilon_table

MIN_N = 10
ESM_MSR_MIN_N = 2           # their safe_spearman: >= 2 rows, both vectors non-constant
METRICS = ("spearman", "pearson", "mae", "rmse")
DEPTHS = ("single", "double", "all")


def _corr(fn, o, p) -> float:
    if len(o) < 2 or np.ptp(o) == 0 or np.ptp(p) == 0:
        return float("nan")
    return float(fn(o, p).statistic)


def _block(o: np.ndarray, p: np.ndarray, min_n: int) -> dict:
    n = len(o)
    if n < min_n:
        return {"n": n, **{m: float("nan") for m in METRICS}}
    e = p - o
    return {"n": n,
            "spearman": _corr(stats.spearmanr, o, p),
            "pearson": _corr(stats.pearsonr, o, p),
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2)))}


def depth_label(n_mut) -> np.ndarray:
    n = np.asarray(n_mut)
    return np.where(n == 1, "single", np.where(n == 2, "double",
                    np.where(n == 0, "wt", "multi")))


def _depth_subsets(df: pd.DataFrame, depths: tuple[str, ...]):
    """(depth, rows) for each requested depth with rows; ``all`` = every
    non-WT row."""
    lab = depth_label(df["n_mut"])
    for dep in depths:
        g = df[df["n_mut"].to_numpy() > 0] if dep == "all" else df[lab == dep]
        if len(g):
            yield dep, g


def per_protein_metrics(df: pd.DataFrame, pred_col: str, *, score_col: str = "score",
                        protein_col: str = "target_id", min_n: int = MIN_N,
                        depths: tuple[str, ...] = DEPTHS) -> pd.DataFrame:
    """One row per (protein, depth) with ``n`` and ``METRICS``."""
    rows = []
    for prot, sub in df.groupby(protein_col, sort=True, observed=True):
        for dep, g in _depth_subsets(sub, depths):
            rows.append({protein_col: prot, "depth": dep,
                         **_block(g[score_col].to_numpy(float),
                                  g[pred_col].to_numpy(float), min_n)})
    return pd.DataFrame(rows, columns=[protein_col, "depth", "n", *METRICS])


def pooled_metrics(df: pd.DataFrame, pred_col: str, *, score_col: str = "score",
                   depths: tuple[str, ...] = DEPTHS) -> pd.DataFrame:
    """Secondary: all proteins' rows pooled (dominated by large proteins and by
    between-protein offsets; never the headline number)."""
    rows = [{"depth": dep, **_block(g[score_col].to_numpy(float),
                                    g[pred_col].to_numpy(float), 1)}
            for dep, g in _depth_subsets(df, depths)]
    return pd.DataFrame(rows)


def min_n_drops(df: pd.DataFrame, pred_col: str, *, score_col: str = "score",
                protein_col: str = "target_id", metric: str = "spearman",
                rules: tuple[int, ...] = (MIN_N, ESM_MSR_MIN_N),
                depths: tuple[str, ...] = DEPTHS) -> pd.DataFrame:
    """Which (protein, depth) cells each ``min_n`` rule leaves undefined for
    ``metric``: one row per cell undefined under at least one rule, with a
    ``dropped_min_n_<k>`` flag per rule. A cell can be undefined under every
    rule (constant observed or predicted vector)."""
    flags = {k: per_protein_metrics(df, pred_col, score_col=score_col,
                                    protein_col=protein_col, min_n=k, depths=depths)
             for k in rules}
    base = next(iter(flags.values()))[[protein_col, "depth", "n"]]
    for k, pp in flags.items():
        base = base.assign(**{f"dropped_min_n_{k}": pp[metric].isna().to_numpy()})
    cols = [f"dropped_min_n_{k}" for k in rules]
    return base[base[cols].any(axis=1)].reset_index(drop=True)


def epistasis_table(df: pd.DataFrame, pred_col: str, *, score_col: str = "score",
                    protein_col: str = "target_id") -> pd.DataFrame:
    """Complete A/B/AB groups per protein; scores and predictions are already
    WT-relative, so the WT term is 0 on both sides."""
    t = df[[protein_col, "mutant", "n_mut", score_col, pred_col]
           + (["score_se"] if "score_se" in df.columns else [])].copy()
    t = t[t["n_mut"].isin([1, 2])]
    t["mutant"] = t["mutant"].fillna("")
    tab = build_epsilon_table(t.rename(columns={protein_col: "_grp"}).assign(
                                  target_id=t[protein_col].to_numpy()),
                              score_col=score_col, pred_col=pred_col, group_col="_grp",
                              se_col="score_se" if "score_se" in t.columns else None,
                              depths=(2,), wt_score=0.0)
    return tab.drop(columns=["_grp"], errors="ignore")


def epistasis_metrics(df: pd.DataFrame, pred_col: str, *, score_col: str = "score",
                      protein_col: str = "target_id", min_n: int = MIN_N,
                      table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per protein: n complete groups, ε RMSE / MAE, ε Spearman / Pearson
    (NaN where ε_pred is constant, e.g. any additive model).

    A/B/AB groups are always formed within one ``target_id`` (one sequence
    background); with a coarser ``protein_col`` such as ``code_wt`` they are
    then summarized per that column."""
    tab = table if table is not None else epistasis_table(
        df, pred_col, score_col=score_col,
        protein_col="target_id" if "target_id" in df.columns else protein_col)
    if protein_col != "target_id" and protein_col not in tab.columns and len(tab):
        grp = df.drop_duplicates("target_id").set_index("target_id")[protein_col]
        tab = tab.assign(**{protein_col: tab["target_id"].map(grp).to_numpy()})
    key = protein_col if protein_col in tab.columns else "target_id"
    rows = []
    if len(tab):
        for prot, sub in tab.groupby(key, sort=True):
            b = _block(sub["eps_obs"].to_numpy(float), sub["eps_pred"].to_numpy(float), min_n)
            rows.append({protein_col: prot, "depth": "epistasis", **b})
    return pd.DataFrame(rows, columns=[protein_col, "depth", "n", *METRICS])


def ndcg_at_k(pred: np.ndarray, true: np.ndarray, k: int | None = 96) -> dict:
    """One protein: ESM-MSR's ``compute_ndcg_flexible(top_n=k)``; ``k=None`` is
    their ``threshold=0.0`` call (``safe_ndcg_t0``), which ranks the full list."""
    from sklearn.metrics import ndcg_score
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    total = int((true > 0).sum())
    if total == 0 or len(true) < 2:
        return {"ndcg": float("nan"), "hits": 0, "ideal_hits": 0, "total_hits": total,
                "k": 0, "precision": float("nan"), "recall": float("nan")}
    rel = np.where(true <= 0, 0.0, true)
    kk = len(true) if k is None else min(int(k), len(true))
    nd = float(ndcg_score(rel[None], pred[None], k=None if k is None else kk,
                          ignore_ties=True))
    hits = int((rel[np.argsort(-pred)][:kk] > 0).sum())
    return {"ndcg": nd, "hits": hits, "ideal_hits": min(total, kk), "total_hits": total,
            "k": kk, "precision": hits / kk, "recall": hits / total}


def retrieval_metrics(df: pd.DataFrame, pred_col: str, *, score_col: str = "score",
                      protein_col: str = "target_id", k: int = 96) -> pd.DataFrame:
    """Per protein and row subset (``all`` = singles + doubles, ``single``,
    ``double``): NDCG@k and hits/precision/recall at k, plus ``ndcg_t0``
    (full-list NDCG, ESM-MSR's ``safe_ndcg_t0``)."""
    d = df[df["n_mut"] > 0].assign(_depth=depth_label(df.loc[df["n_mut"] > 0, "n_mut"]))
    rows = []
    for prot, sub in d.groupby(protein_col, sort=True):
        for subset, g in (("all", sub), ("single", sub[sub["_depth"] == "single"]),
                          ("double", sub[sub["_depth"] == "double"])):
            if len(g):
                p, t = g[pred_col].to_numpy(), g[score_col].to_numpy()
                rows.append({protein_col: prot, "subset": subset, "n": len(g),
                             **ndcg_at_k(p, t, k), "ndcg_t0": ndcg_at_k(p, t, None)["ndcg"]})
    return pd.DataFrame(rows)


def summarize_retrieval(r: pd.DataFrame) -> pd.DataFrame:
    """Macro over proteins with a defined NDCG (at least one ΔΔG > 0 row)."""
    out = []
    for subset, g in r.groupby("subset", sort=False):
        ok = g["ndcg"].notna()
        out.append({"subset": subset, "ndcg_macro": float(g.loc[ok, "ndcg"].mean()),
                    "ndcg_t0_macro": float(g.loc[ok, "ndcg_t0"].mean()),
                    "precision_macro": float(g.loc[ok, "precision"].mean()),
                    "recall_macro": float(g.loc[ok, "recall"].mean()),
                    "n_proteins": int(ok.sum()), "n_proteins_total": len(g),
                    "n_proteins_without_stabilizing": int((~ok).sum())})
    return pd.DataFrame(out)


def summarize(per_protein: pd.DataFrame, protein_col: str = "target_id") -> pd.DataFrame:
    """Macro (primary) and median over proteins per depth and metric, with the
    number of proteins contributing a defined value."""
    rows = []
    for dep, sub in per_protein.groupby("depth", sort=False, observed=True):
        for m in METRICS:
            v = sub[m].to_numpy(float)
            ok = np.isfinite(v)
            rows.append({"depth": dep, "metric": m,
                         "macro": float(v[ok].mean()) if ok.any() else float("nan"),
                         "median": float(np.median(v[ok])) if ok.any() else float("nan"),
                         "n_proteins": int(ok.sum()),
                         "n_proteins_total": int(sub[protein_col].nunique()),
                         "n_rows": int(sub.loc[ok, "n"].sum())})
    return pd.DataFrame(rows)


def evaluate_frame(df: pd.DataFrame, pred_col: str, *, score_col: str = "score",
                   protein_col: str = "target_id", min_n: int = MIN_N) -> dict:
    """Everything §4 and §5.3 ask for, for one prediction column.

    Returns ``per_protein`` (single, double, all-depth and epistasis rows), ``summary``
    (macro/median), ``pooled`` and ``epistasis_table``.
    """
    pp = per_protein_metrics(df, pred_col, score_col=score_col,
                             protein_col=protein_col, min_n=min_n)
    tab = epistasis_table(df, pred_col, score_col=score_col,
                          protein_col="target_id" if "target_id" in df.columns else protein_col)
    ep = epistasis_metrics(df, pred_col, score_col=score_col, protein_col=protein_col,
                           min_n=min_n, table=tab)
    allp = pd.concat([pp, ep], ignore_index=True)
    rt = retrieval_metrics(df, pred_col, score_col=score_col, protein_col=protein_col)
    return {"per_protein": allp, "summary": summarize(allp, protein_col),
            "pooled": pooled_metrics(df, pred_col, score_col=score_col),
            "epistasis_table": tab, "retrieval": rt,
            "retrieval_summary": summarize_retrieval(rt)}


def macro(summary: pd.DataFrame, depth: str, metric: str) -> float:
    s = summary[(summary["depth"] == depth) & (summary["metric"] == metric)]
    return float(s["macro"].iloc[0]) if len(s) else float("nan")


SELECTION_RULES = {
    "double_rmse": "macro val double RMSE, then macro val single RMSE, then earliest",
    "spearman": "macro val all-depth Spearman (higher), then macro val double Spearman, "
                "then earliest",
    "double_spearman": "macro val double Spearman (higher), then macro val single Spearman, then earliest",
    "ndcg30": "macro val double nDCG@30 (Mutate Everything definition), then double Spearman, then earliest",
}


def selection_key(summary: pd.DataFrame, rule: str = "double_rmse", *, retrieval=None) -> tuple[float, float]:
    """Checkpoint-selection key, lower is better (earliest checkpoint breaks
    any remaining tie in the caller). NaN sorts last.

    ``double_rmse`` is TODO §3's original criterion. ``spearman`` (§5.5) is the
    macro within-protein Spearman over singles and doubles pooled, ESM-MSR's
    early-stopping quantity: in the 2026-11-02 early-stopping screen 7 of 8
    runs selected inside LR warm-up under ``double_rmse``, because double RMSE
    wanders while Spearman is flat. Chosen on validation only."""
    def f(x):
        return x if np.isfinite(x) else float("inf")
    if rule == "double_rmse":
        return (f(macro(summary, "double", "rmse")), f(macro(summary, "single", "rmse")))
    if rule == "spearman":
        return (f(-macro(summary, "all", "spearman")), f(-macro(summary, "double", "spearman")))
    if rule == "double_spearman":
        return (f(-macro(summary, "double", "spearman")), f(-macro(summary, "single", "spearman")))
    if rule == "ndcg30":
        if retrieval is None:
            raise ValueError("ndcg30 selection requires explicit retrieval metrics")
        return (f(-retrieval["ndcg30_double"]), f(-macro(summary, "double", "spearman")))
    raise ValueError(f"unknown selection rule {rule!r}; one of {sorted(SELECTION_RULES)}")


def paired_protein_bootstrap(a: pd.DataFrame, b: pd.DataFrame, metric: str, depth: str, *,
                             protein_col: str = "target_id", n_boot: int = 10_000,
                             seed: int = 0, ci: float = 0.95) -> dict:
    """Mean over proteins of (a − b) on proteins where both are defined, with a
    percentile bootstrap interval resampling proteins."""
    x = a[a["depth"] == depth].set_index(protein_col)[metric]
    y = b[b["depth"] == depth].set_index(protein_col)[metric]
    d = (x - y).dropna()
    if not len(d):
        return {"n_proteins": 0, "mean_diff": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "frac_a_better": float("nan")}
    v = d.to_numpy(float)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    lower_better = metric in ("mae", "rmse")
    return {"n_proteins": len(v), "mean_diff": float(v.mean()), "lo": float(lo),
            "hi": float(hi),
            "frac_a_better": float(np.mean(v < 0 if lower_better else v > 0))}
