"""
tests/test_eval.py
------------------
depth_spearman, epsilon and the results table on constructed input where the
answer is known in advance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mutant_cache.eval import (
    ResultsTable, attenuation_ceiling, build_epsilon_table, depth_bucket,
    depth_spearman, depth_summary, epsilon_correlation,
)


# ── depth_spearman ───────────────────────────────────────────────────────────

def test_depth_bucket_collapses_five_and_up():
    assert list(depth_bucket([1, 2, 3, 4, 5, 6, 17])) == \
        ["1", "2", "3", "4", "5+", "5+", "5+"]


def _frame(rows):
    return pd.DataFrame(rows, columns=["score_set_id", "n_mut", "score", "pred",
                                       "reliability"])


def test_depth_spearman_known_answers():
    """Depth 1 is a perfect ranking, depth 2 a perfectly inverted one."""
    rows = []
    for i in range(6):
        rows.append(["S1", 1, float(i), float(i), 0.80])        # rho = +1
    for i in range(6):
        rows.append(["S1", 2, float(i), float(-i), 0.80])       # rho = -1
    t = depth_spearman(_frame(rows), "pred", "score")
    s1 = t[t["score_set_id"] == "S1"].set_index("depth")
    assert s1.loc["1", "spearman"] == pytest.approx(1.0)
    assert s1.loc["2", "spearman"] == pytest.approx(-1.0)
    assert s1.loc["1", "n"] == 6
    # the pooled row mixes the two and lands near zero
    assert abs(s1.loc["all", "spearman"]) < 0.2
    # the ceiling rides along, and the attainment ratio is rho / ceiling
    assert s1.loc["1", "reliability"] == pytest.approx(0.80)
    assert s1.loc["1", "spearman_over_ceiling"] == pytest.approx(1.25)


def test_depth_spearman_all_block_is_the_mean_over_score_sets():
    rows = []
    for i in range(5):
        rows.append(["A", 1, float(i), float(i), 0.9])          # +1
    for i in range(5):
        rows.append(["B", 1, float(i), float(-i), 0.7])         # -1
    t = depth_spearman(_frame(rows), "pred", "score")
    allrow = t[(t["score_set_id"] == "ALL") & (t["depth"] == "1")].iloc[0]
    assert allrow["spearman"] == pytest.approx(0.0)             # mean(+1, -1)
    assert allrow["n"] == 10
    assert allrow["n_score_sets"] == 2
    assert allrow["reliability"] == pytest.approx(0.8)          # mean(0.9, 0.7)


def test_depth_spearman_skips_buckets_below_min_n():
    rows = [["S", 1, float(i), float(i), np.nan] for i in range(5)]
    rows += [["S", 5, 1.0, 1.0, np.nan], ["S", 7, 2.0, 2.0, np.nan]]  # n=2 at 5+
    t = depth_spearman(_frame(rows), "pred", "score", min_n=3)
    depths = set(t[t["score_set_id"] == "S"]["depth"])
    assert depths == {"1", "all"}


def test_depth_spearman_requires_its_columns():
    with pytest.raises(KeyError):
        depth_spearman(pd.DataFrame({"a": [1]}), "pred", "score")


def test_depth_summary_is_flat():
    rows = [["S", 1, float(i), float(i), 0.9] for i in range(5)]
    s = depth_summary(depth_spearman(_frame(rows), "pred", "score"))
    assert s["spearman_depth_1"] == pytest.approx(1.0)
    assert s["spearman_depth_all"] == pytest.approx(1.0)


# ── epsilon ──────────────────────────────────────────────────────────────────

def _megascale(eps_true, eps_pred, n=12, seed=0):
    """A ΔG score set whose doubles carry a prescribed ε, exactly."""
    rng = np.random.default_rng(seed)
    singles = {f"A{i}G": rng.normal() for i in range(1, n + 1)}
    rows = [{"score_set_id": "M", "target_id": "T", "mutant": "", "n_mut": 0,
             "score": 0.0, "pred": 0.0, "score_se": 0.01, "assay_type": "dG"}]
    for k, v in singles.items():
        rows.append({"score_set_id": "M", "target_id": "T", "mutant": k,
                     "n_mut": 1, "score": v, "pred": v, "score_se": 0.01,
                     "assay_type": "dG"})
    keys = list(singles)
    for j in range(len(keys) - 1):
        a, b = keys[j], keys[j + 1]
        e_o, e_p = eps_true[j], eps_pred[j]
        rows.append({"score_set_id": "M", "target_id": "T",
                     "mutant": f"{a}:{b}", "n_mut": 2,
                     "score": singles[a] + singles[b] + e_o,
                     "pred": singles[a] + singles[b] + e_p,
                     "score_se": 0.01, "assay_type": "dG"})
    return pd.DataFrame(rows)


def test_epsilon_is_recovered_exactly():
    eps = np.linspace(-1, 1, 11)
    df = _megascale(eps, eps)
    t = build_epsilon_table(df)
    assert len(t) == 11
    np.testing.assert_allclose(np.sort(t["eps_obs"].to_numpy()), np.sort(eps),
                               atol=1e-9)
    np.testing.assert_allclose(t["eps_obs"].to_numpy(), t["eps_pred"].to_numpy(),
                               atol=1e-9)


def test_epsilon_correlation_known_answers():
    eps = np.linspace(-1, 1, 11)
    perfect = epsilon_correlation(_megascale(eps, eps))
    inverted = epsilon_correlation(_megascale(eps, -eps))
    assert perfect[perfect["score_set_id"] == "M"]["spearman"].iloc[0] == \
        pytest.approx(1.0)
    assert inverted[inverted["score_set_id"] == "M"]["spearman"].iloc[0] == \
        pytest.approx(-1.0)


def test_an_additive_predictor_scores_zero_on_epsilon():
    """The whole reason ε exists: a model that is additive by construction has
    ε_pred identically 0 and cannot correlate with anything."""
    eps = np.linspace(-1, 1, 11)
    df = _megascale(eps, np.zeros(11))
    t = build_epsilon_table(df)
    np.testing.assert_allclose(t["eps_pred"].to_numpy(), 0.0, atol=1e-9)
    out = epsilon_correlation(df)
    assert np.isnan(out[out["score_set_id"] == "M"]["spearman"].iloc[0])


def test_epsilon_refuses_a_fitness_scale_without_a_link():
    eps = np.linspace(-1, 1, 11)
    df = _megascale(eps, eps)
    df["assay_type"] = "fitness"
    with pytest.raises(ValueError, match="additive scale"):
        epsilon_correlation(df)
    out = epsilon_correlation(df, allow_nonadditive=True)
    assert len(out)


def test_epsilon_skips_multimutants_whose_singles_are_missing():
    eps = np.linspace(-1, 1, 11)
    df = _megascale(eps, eps)
    df = df[df["mutant"] != "A1G"]          # drop one single
    t = build_epsilon_table(df)
    assert not t["mutant"].str.contains("A1G").any()
    assert len(t) == 10


def test_attenuation_ceiling():
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 1, 5000)
    se = np.full(5000, 0.5)
    obs = truth + rng.normal(0, 0.5, 5000)
    # Var(obs) ~= 1.25, noise 0.25 -> ceiling ~= sqrt(1 - .25/1.25) = sqrt(.8)
    assert attenuation_ceiling(obs, se) == pytest.approx(np.sqrt(0.8), abs=0.02)
    assert np.isnan(attenuation_ceiling(np.ones(10), np.ones(10)))


def test_epsilon_ceiling_is_lower_than_the_score_ceiling():
    """ε is a difference of three noisy numbers, so its ceiling must be worse
    than any single measurement's."""
    eps = np.linspace(-1, 1, 11)
    df = _megascale(eps, eps)
    df["score_se"] = 0.3
    out = epsilon_correlation(df)
    row = out[out["score_set_id"] == "M"].iloc[0]
    assert row["eps_noise_sd"] > 0.3            # sqrt(3) * 0.3 = 0.52
    assert row["eps_noise_sd"] == pytest.approx(np.sqrt(3) * 0.3, rel=1e-6)
    assert 0.0 < row["eps_ceiling"] < 1.0


# ── results table ────────────────────────────────────────────────────────────

def test_zero_shot_rows_are_in_their_own_labelled_block():
    t = ResultsTable("demo")
    t.add("variant cache", {"depth_1": 0.71, "depth_2": 0.64},
          ceiling={"depth_1": 0.9, "depth_2": 0.9})
    t.add("ridge on one-hot", {"depth_1": 0.60}, kind="baseline", ceiling=0.9)
    t.add_zero_shot("ESM-C masked marginals", {"depth_1": 0.42}, ceiling=0.9)
    md = t.to_markdown()

    assert "no DMS labels" in md
    # the supervised rows come before the context block, always
    assert md.index("variant cache") < md.index("no DMS labels")
    assert md.index("ridge on one-hot") < md.index("no DMS labels")
    assert md.index("no DMS labels") < md.index("ESM-C masked marginals")

    frame = t.to_frame()
    zs = frame[frame["kind"] == "zero_shot"]
    assert len(zs) == 1 and zs["note"].iloc[0] == "no labels seen"


def test_every_correlation_prints_its_ceiling():
    t = ResultsTable("demo")
    t.add("a", {"spearman": 0.6, "n_params": 1000.0}, ceiling=0.75)
    t.add("b", {"spearman": 0.6})                     # ceiling unknown
    md = t.to_markdown()
    assert "0.600 (≤0.750)" in md
    assert "0.600 (≤ ?)" in md
    assert "1000.000 (≤" not in md                    # not a correlation
    f = t.to_frame()
    row = f[(f["name"] == "a") & (f["metric"] == "spearman")].iloc[0]
    assert row["attainment"] == pytest.approx(0.8)
    assert np.isnan(f[(f["name"] == "b") & (f["metric"] == "spearman")]
                    ["attainment"].iloc[0])


def test_add_depth_table_carries_the_ceiling(tmp_path):
    rows = [["S", 1, float(i), float(i), 0.85] for i in range(6)]
    rows += [["S", 2, float(i), float(i), 0.85] for i in range(6)]
    dt = depth_spearman(_frame(rows), "pred", "score")
    t = ResultsTable("demo").add_depth_table("model", dt)
    md = t.to_markdown()
    assert "1.000 (≤0.850)" in md
    p = t.write(tmp_path / "results")
    assert p.exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "results.json").exists()


def test_throughput_row():
    t = ResultsTable("demo").add_throughput("throughput",
                                            variants_per_s_cache=900.0,
                                            variants_per_s_full=100.0,
                                            accuracy_delta=-0.004)
    f = t.to_frame()
    assert f[f["metric"] == "throughput_ratio"]["value"].iloc[0] == \
        pytest.approx(9.0)
    assert "Controls" in t.to_markdown()


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        ResultsTable("x").add("a", {}, kind="vibes")
