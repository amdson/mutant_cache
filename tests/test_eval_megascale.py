"""Megascale held-out-protein metrics (mutant_cache/eval/megascale.py)."""
import numpy as np
import pandas as pd
import pytest

from mutant_cache.eval import megascale as M


def _frame(seed=0, n_prot=3):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_prot):
        s = {f"A{i}C": rng.normal() for i in range(1, 16)}
        for m, v in s.items():
            rows.append((f"P{p}", m, 1, v))
        for i in range(1, 15):
            a, b = f"A{i}C", f"A{i + 1}C"
            rows.append((f"P{p}", f"{a}:{b}", 2, s[a] + s[b] + rng.normal(scale=0.5)))
        rows.append((f"P{p}", "", 0, 0.0))
    df = pd.DataFrame(rows, columns=["target_id", "mutant", "n_mut", "score"])
    return df


def test_perfect_and_additive_predictions():
    df = _frame()
    df["perfect"] = df["score"]
    sing = dict(zip(df.loc[df.n_mut == 1, "target_id"] + df.loc[df.n_mut == 1, "mutant"],
                    df.loc[df.n_mut == 1, "score"]))
    df["additive"] = [0.0 if m == "" else sum(sing[t + x] for x in m.split(":"))
                      for t, m in zip(df.target_id, df.mutant)]
    perf = M.evaluate_frame(df, "perfect")
    assert M.macro(perf["summary"], "double", "rmse") == 0.0
    assert M.macro(perf["summary"], "epistasis", "rmse") == 0.0
    assert M.macro(perf["summary"], "single", "spearman") == pytest.approx(1.0)
    add = M.evaluate_frame(df, "additive")
    ep = add["per_protein"][add["per_protein"].depth == "epistasis"]
    assert (ep["n"] == 14).all()
    assert ep["spearman"].isna().all()                  # eps_pred == 0: undefined
    assert (ep["rmse"] > 0).all()
    # additive RMSE on doubles equals the epistasis RMSE (same residual)
    dbl = add["per_protein"][add["per_protein"].depth == "double"].set_index("target_id")
    assert np.allclose(dbl["rmse"], ep.set_index("target_id")["rmse"])


def test_min_n_and_selection_key():
    df = _frame(n_prot=2)
    df["pred"] = df["score"] + 0.1
    df = df[~((df.target_id == "P1") & (df.n_mut == 2))]           # P1: no doubles
    out = M.evaluate_frame(df, "pred", min_n=10)
    s = out["summary"]
    row = s[(s.depth == "double") & (s.metric == "rmse")].iloc[0]
    assert row.n_proteins == 1 and row.macro == pytest.approx(0.1)
    assert M.selection_key(s) == (pytest.approx(0.1), pytest.approx(0.1))


def test_paired_bootstrap():
    df = _frame(n_prot=6)
    df["good"] = df["score"] + 0.05
    df["bad"] = df["score"] + 0.5
    a = M.evaluate_frame(df, "good")["per_protein"]
    b = M.evaluate_frame(df, "bad")["per_protein"]
    r = M.paired_protein_bootstrap(a, b, "rmse", "double", n_boot=500)
    assert r["n_proteins"] == 6 and r["mean_diff"] == pytest.approx(-0.45)
    assert r["hi"] < 0 and r["frac_a_better"] == 1.0


def test_ndcg_matches_upstream_definition():
    from sklearn.metrics import ndcg_score
    rng = np.random.default_rng(1)
    true = rng.normal(size=200)
    pred = true + rng.normal(scale=0.5, size=200)
    r = M.ndcg_at_k(pred, true, k=96)
    rel = np.where(true <= 0, 0.0, true)
    assert r["ndcg"] == pytest.approx(ndcg_score(rel[None], pred[None], k=96, ignore_ties=True))
    assert r["k"] == 96 and r["total_hits"] == int((true > 0).sum())
    assert M.ndcg_at_k(true, true)["ndcg"] == pytest.approx(1.0)
    assert np.isnan(M.ndcg_at_k(pred, -np.abs(true))["ndcg"])       # no stabilizing rows
    small = M.ndcg_at_k(pred[:10], true[:10])
    assert small["k"] == 10


def test_retrieval_summary_counts_undefined_proteins():
    df = _frame(n_prot=3)
    df.loc[df.target_id == "P2", "score"] = -np.abs(df.loc[df.target_id == "P2", "score"])
    df["pred"] = df["score"]
    out = M.evaluate_frame(df, "pred")
    rs = out["retrieval_summary"].set_index("subset")
    assert rs.loc["all", "n_proteins"] == 2 and rs.loc["all", "n_proteins_without_stabilizing"] == 1
    assert rs.loc["all", "ndcg_macro"] == pytest.approx(1.0)


def test_all_depth_spearman_pools_singles_and_doubles():
    from scipy import stats
    df = _frame(n_prot=3)
    rng = np.random.default_rng(2)
    df["pred"] = df["score"] + rng.normal(scale=0.7, size=len(df))
    pp = M.evaluate_frame(df, "pred")["per_protein"]
    for prot, sub in df[df.n_mut > 0].groupby("target_id"):
        row = pp[(pp.target_id == prot) & (pp.depth == "all")].iloc[0]
        assert row.n == 29
        assert row.spearman == pytest.approx(stats.spearmanr(sub.score, sub.pred).statistic)
    # the per-depth rows are unchanged by adding the pooled one
    old = M.per_protein_metrics(df, "pred", depths=("single", "double"))
    new = pp[pp.depth.isin(["single", "double"])]
    pd.testing.assert_frame_equal(old.sort_values(["target_id", "depth"]).reset_index(drop=True),
                                  new.sort_values(["target_id", "depth"]).reset_index(drop=True))


def _upstream_safe_spearman(preds, targets):
    # src/esm_msr/stats.py:safe_spearman @ c34c315, verbatim logic
    from scipy.stats import spearmanr
    if len(preds) < 2 or len(np.unique(targets)) < 2:
        return np.nan
    if len(np.unique(preds)) < 2:
        return np.nan
    rho = spearmanr(preds, targets)[0]
    return np.nan if np.isnan(rho) else float(rho)


def test_esm_msr_min_n_matches_upstream_and_reports_drops():
    df = _frame(n_prot=3)
    rng = np.random.default_rng(3)
    df["pred"] = df["score"] + rng.normal(size=len(df))
    small = df[(df.target_id == "P2") & (df.n_mut == 1)].head(4)     # P2: 4 singles only
    df = pd.concat([df[df.target_id != "P2"], small], ignore_index=True)
    pp = M.per_protein_metrics(df, "pred", min_n=M.ESM_MSR_MIN_N)
    for (prot, dep), g in pp.set_index(["target_id", "depth"]).iterrows():
        sub = df[df.target_id == prot]
        sub = sub[sub.n_mut > 0] if dep == "all" else sub[M.depth_label(sub.n_mut) == dep]
        up = _upstream_safe_spearman(sub.pred.to_numpy(), sub.score.to_numpy())
        assert (np.isnan(up) and np.isnan(g.spearman)) or g.spearman == pytest.approx(up)
    assert pp.query("target_id == 'P2' and depth == 'single'").spearman.notna().all()
    two = df[df.target_id == "P0"].head(2).assign(target_id="P3")
    assert M.per_protein_metrics(two, "pred", min_n=2).spearman.notna().all()   # n = 2 defined
    drops = M.min_n_drops(df, "pred")
    assert set(zip(drops.target_id, drops.depth)) == {("P2", "single"), ("P2", "all")}
    assert drops["dropped_min_n_10"].all() and not drops["dropped_min_n_2"].any()


def test_macro_skips_undefined_proteins_like_pandas_mean():
    df = _frame(n_prot=4)
    df["pred"] = df["score"] + np.random.default_rng(4).normal(size=len(df))
    df.loc[df.target_id == "P3", "pred"] = 1.0                       # constant: rho undefined
    pp = M.evaluate_frame(df, "pred")["per_protein"]
    s = M.summarize(pp)
    for dep in ("single", "double", "all"):
        v = pp[pp.depth == dep].set_index("target_id")["spearman"]
        assert np.isnan(v["P3"])
        assert M.macro(s, dep, "spearman") == pytest.approx(v.mean())   # pandas skips NaN
        assert s.query("depth == @dep and metric == 'spearman'").n_proteins.iloc[0] == 3


def test_ndcg_t0_matches_upstream_threshold_call():
    from sklearn.metrics import ndcg_score
    rng = np.random.default_rng(5)
    true = rng.normal(size=200)
    pred = true + rng.normal(scale=0.5, size=200)
    rel = np.where(true <= 0, 0.0, true)
    r = M.ndcg_at_k(pred, true, k=None)
    assert r["ndcg"] == pytest.approx(ndcg_score(rel[None], pred[None], k=None, ignore_ties=True))
    assert r["hits"] == r["total_hits"] == r["ideal_hits"]
    assert np.isnan(M.ndcg_at_k(pred, -np.abs(true), k=None)["ndcg"])
    df = _frame(n_prot=2)
    df["pred"] = df["score"]
    rs = M.evaluate_frame(df, "pred")["retrieval_summary"].set_index("subset")
    assert rs.loc["all", "ndcg_t0_macro"] == pytest.approx(1.0)


def test_spearman_selection_rule():
    df = _frame(n_prot=3)
    rng = np.random.default_rng(6)
    good = M.evaluate_frame(df.assign(p=df.score + rng.normal(scale=0.1, size=len(df))), "p")
    bad = M.evaluate_frame(df.assign(p=df.score + rng.normal(scale=3.0, size=len(df))), "p")
    assert M.selection_key(good["summary"], "spearman") < M.selection_key(bad["summary"], "spearman")
    k = M.selection_key(good["summary"], "spearman")
    assert k[0] == pytest.approx(-M.macro(good["summary"], "all", "spearman"))
    assert M.selection_key(good["summary"]) == M.selection_key(good["summary"], "double_rmse")
    with pytest.raises(ValueError):
        M.selection_key(good["summary"], "bogus")


def test_code_wt_grouping_keeps_epistasis_within_background():
    df = _frame(n_prot=2)
    bg = df[df.target_id == "P1"].assign(target_id="P0_bg")         # same mutants, other background
    bg["score"] = bg["score"] + np.random.default_rng(7).normal(size=len(bg))
    df = pd.concat([df, bg], ignore_index=True)
    df["code_wt"] = df.target_id.str.split("_").str[0]
    df["pred"] = df["score"]
    out = M.evaluate_frame(df, "pred", protein_col="code_wt")
    pp = out["per_protein"]
    assert set(pp.code_wt) == {"P0", "P1"}
    ep = pp[pp.depth == "epistasis"].set_index("code_wt")
    assert ep.loc["P0", "n"] == 28 and ep.loc["P0", "rmse"] == 0.0     # 14 groups per background
    assert pp.query("code_wt == 'P0' and depth == 'all'").n.iloc[0] == 58
