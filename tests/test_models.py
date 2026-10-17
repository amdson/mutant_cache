"""
tests/test_models.py
--------------------
The model and loss layers against the stub backbone and synthetic tables:
contract plumbing, the masked-marginal initialisation, the sampler, and the
end-to-end claim that the thing trains on a task with real epistasis.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fixtures import AA_TO_ID, make_stub_backbone, make_synthetic_dataset

from mutant_cache.models import (
    LossConfig, ScoreSetBatchSampler, VariantEffectModel, VariantLoss,
    VariantModelConfig, build_variant_batches, pairwise_rank_loss,
)
from mutant_cache.models.data_compat import load_tables, parse_mutant


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    out, truth = make_synthetic_dataset(d, n_targets=2, seq_len=16,
                                        n_singles=80, n_doubles=100, n_triples=30,
                                        seed=7)
    return out, truth


@pytest.fixture(scope="module")
def backbone():
    return make_stub_backbone(d_model=32, n_layers=2, n_heads=4, seed=1)


@pytest.fixture(scope="module")
def tables(dataset):
    out, truth = dataset
    t = load_tables(out)
    seqs = dict(zip(t["targets"]["target_id"], t["targets"]["sequence"]))
    return t["variants"], seqs, truth


def make_model(backbone, n_assays, **kw):
    cfg = VariantModelConfig(n_assays=n_assays, **kw)
    torch.manual_seed(0)
    return VariantEffectModel(backbone, cfg)


# ── contract plumbing ────────────────────────────────────────────────────────

def test_parse_mutant_roundtrip():
    assert parse_mutant("A23G:K45R") == [(23, "A", "G"), (45, "K", "R")]
    assert parse_mutant("") == []


def test_batches_never_cross_a_score_set(tables):
    variants, seqs, _ = tables
    batches, idx = build_variant_batches(variants, seqs, AA_TO_ID, batch_size=16)
    assert batches
    rows = variants.reset_index()
    for b in batches:
        got = rows.loc[b.row_index.numpy(), "score_set_id"].unique()
        assert list(got) == [b.score_set_id]
        assert b.variant_ids.shape == (b.n, b.k)
        # padded slots must carry the reference residue, per §1
        pad = ~b.var_mask
        ref_at = b.ref_ids[b.positions][None].expand(b.n, -1)
        assert torch.equal(b.variant_ids[pad], ref_at[pad])
        assert int(b.n_mut.max()) == int(b.var_mask.sum(-1).max())


def test_batch_covers_every_variant_exactly_once(tables):
    variants, seqs, _ = tables
    batches, _ = build_variant_batches(variants, seqs, AA_TO_ID, batch_size=16)
    seen = np.concatenate([b.row_index.numpy() for b in batches])
    assert sorted(seen) == sorted(variants.index.tolist())


def test_cache_matches_bruteforce_oracle(tables, backbone):
    """The stub's two paths must agree — otherwise the ablation and control
    tests below are measuring the stub, not the model."""
    variants, seqs, _ = tables
    batches, _ = build_variant_batches(variants, seqs, AA_TO_ID, batch_size=8)
    m = make_model(backbone, 4)
    b = batches[0]
    with torch.no_grad():
        a = m(b, backend="cache").pred
        c = m(b, backend="recompute").pred
    assert torch.allclose(a, c, atol=1e-4), (a - c).abs().max()


# ── masked-marginal initialisation ───────────────────────────────────────────

def test_init_is_the_summed_masked_marginal(tables, backbone):
    from mutant_cache.baselines import build_llr_matrices

    variants, seqs, _ = tables
    llr = build_llr_matrices(backbone, seqs)
    batches, _ = build_variant_batches(variants, seqs, AA_TO_ID, batch_size=8,
                                       marginals=llr)
    m = make_model(backbone, 4, zero_init_additive=True, marginal_weight_init=1.0)
    b = batches[0]
    with torch.no_grad():
        out = m(b)
    expect = (b.marginals * b.var_mask).sum(-1)
    assert torch.allclose(out.pred, expect, atol=1e-5)
    assert torch.allclose(out.interaction, torch.zeros_like(out.interaction))


def test_marginal_matrix_is_zero_at_wildtype(tables, backbone):
    from mutant_cache.baselines import llr_matrix

    _, seqs, _ = tables
    tid = sorted(seqs)[0]
    M = llr_matrix(backbone, seqs[tid])
    ids = backbone.tokenizer.encode(seqs[tid])
    assert torch.allclose(M.gather(1, ids[:, None]).squeeze(1),
                          torch.zeros(len(ids)), atol=1e-6)


# ── loss ─────────────────────────────────────────────────────────────────────

def test_pairwise_rank_loss_is_zero_for_a_perfect_order():
    y = torch.tensor([0.0, 1.0, 2.0, 3.0])
    perfect = pairwise_rank_loss(y * 50, y)
    inverted = pairwise_rank_loss(-y * 50, y)
    assert perfect < 1e-6
    assert inverted > 10
    # exactly ln 2 per pair when the model is indifferent
    flat = pairwise_rank_loss(torch.zeros(4), y)
    assert abs(float(flat) - np.log(2)) < 1e-6


def test_pairwise_rank_loss_respects_groups():
    pred = torch.tensor([0.0, 1.0, 1.0, 0.0])
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    g = [0, 0, 1, 1]
    # within group 0 the order is right, within group 1 it is wrong
    assert float(pairwise_rank_loss(pred, y, g)) > float(
        pairwise_rank_loss(pred[:2], y[:2]))
    # ignoring groups mixes incomparable pairs and gives a different number
    assert float(pairwise_rank_loss(pred, y, None)) != pytest.approx(
        float(pairwise_rank_loss(pred, y, g)))


def test_min_gap_drops_unresolvable_pairs():
    # the (0.0, 0.01) pair is below any plausible assay resolution; the model
    # gets it badly wrong and both other pairs right.
    y = torch.tensor([0.0, 0.01, 1.0])
    pred = torch.tensor([5.0, -5.0, 10.0])
    keep_all = float(pairwise_rank_loss(pred, y, min_gap=0.0))
    gapped = float(pairwise_rank_loss(pred, y, min_gap=0.1))
    assert gapped < keep_all / 3, (keep_all, gapped)
    # a flat model pays exactly ln 2 per surviving pair either way
    flat = torch.zeros(3)
    assert float(pairwise_rank_loss(flat, y, min_gap=0.1)) == \
        pytest.approx(np.log(2), abs=1e-6)


def test_inverse_variance_pair_weighting_downweights_noisy_pairs():
    y = torch.tensor([0.0, 1.0, 2.0])
    tight = torch.tensor([0.01, 0.01, 0.01])
    loose = torch.tensor([0.01, 10.0, 0.01])

    # an indifferent model pays ln 2 per pair whatever the weights are
    flat = torch.zeros(3)
    for se in (tight, loose):
        assert float(pairwise_rank_loss(flat, y, se=se,
                                        pair_weight="inv_var")) == \
            pytest.approx(np.log(2), abs=1e-6)

    # the only badly ordered pair is (2, 1), and item 1 is the unreliable one;
    # the (2, 0) pair, which the assay can actually resolve, is ordered right.
    pred = torch.tensor([0.0, 5.0, 2.0])
    a = float(pairwise_rank_loss(pred, y, se=tight, pair_weight="inv_var"))
    b = float(pairwise_rank_loss(pred, y, se=loose, pair_weight="inv_var"))
    assert b < a / 5, (a, b)


def test_gap_pair_weighting_prefers_getting_big_gaps_right():
    y = torch.tensor([0.0, 0.1, 1.0])
    # (a) the tiny-gap pair is wrong; (b) the big-gap pair is wrong
    wrong_small = torch.tensor([1.0, 0.0, 2.0])
    wrong_big = torch.tensor([2.0, 1.0, 0.0])
    for kw in ({"pair_weight": "gap"},):
        a = float(pairwise_rank_loss(wrong_small, y, **kw))
        b = float(pairwise_rank_loss(wrong_big, y, **kw))
        assert b > a, (a, b)
    # uniform weighting cares much less about which pair was missed
    u_a = float(pairwise_rank_loss(wrong_small, y))
    u_b = float(pairwise_rank_loss(wrong_big, y))
    assert (b / a) > (u_b / u_a)


def test_mse_dg_inverse_variance():
    from mutant_cache.models.loss import mse_dg_loss

    pred = torch.tensor([0.0, 0.0])
    dg = torch.tensor([1.0, 1.0])
    se = torch.tensor([0.05, 5.0])
    plain = float(mse_dg_loss(pred, dg, se, inverse_variance=False))
    weighted = float(mse_dg_loss(pred, dg, se, inverse_variance=True))
    assert plain == pytest.approx(1.0)
    assert weighted == pytest.approx(1.0)  # identical residuals
    pred = torch.tensor([1.0, 0.0])        # only the noisy one is wrong
    assert float(mse_dg_loss(pred, dg, se, inverse_variance=True)) < \
        float(mse_dg_loss(pred, dg, se, inverse_variance=False))


def test_listmle_and_soft_spearman_prefer_the_right_order():
    from mutant_cache.models.loss import listmle_loss, soft_spearman_loss

    y = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert float(listmle_loss(y * 10, y)) < float(listmle_loss(-y * 10, y))
    assert float(soft_spearman_loss(y, y)) < 1e-4
    assert float(soft_spearman_loss(-y, y)) > 1.9


def test_score_set_sampler_never_mixes_assays():
    ids = ["a"] * 10 + ["b"] * 7 + ["c"] * 1
    s = ScoreSetBatchSampler(ids, batch_size=4, seed=0)
    batches = list(s)
    assert len(batches) == len(s)
    for b in batches:
        assert len({ids[i] for i in b}) == 1
        assert len(b) >= 2
    # the singleton score set contributes no pairs, so it is dropped entirely
    assert not any(ids[i] == "c" for b in batches for i in b)
    covered = sorted(i for b in batches for i in b)
    assert covered == list(range(17))


def test_variant_loss_mixes_terms():
    fn = VariantLoss(LossConfig(kind="rank+mse_rank", rank_weight=1.0,
                                mse_weight=0.5))
    pred = torch.randn(8)
    y = torch.rand(8)
    total, parts = fn(pred, y)
    assert set(parts) == {"rank", "mse_rank"}
    assert float(total) == pytest.approx(sum(parts.values()), rel=1e-6)


# ── the real claim: it trains, on a task with epistasis ──────────────────────

def _spearman(a, b):
    from scipy import stats
    return float(stats.spearmanr(a, b).statistic)


@pytest.mark.parametrize("loss_kind", ["rank", "mse_rank"])
def test_model_trains_and_loss_drops(tables, backbone, loss_kind):
    """Starts at the masked-marginal prior (which on a random stub backbone is
    worth nothing) and has to learn the synthetic landscape from there."""
    from mutant_cache.baselines import build_llr_matrices

    variants, seqs, _ = tables
    sub = variants[variants["score_set_id"] == variants["score_set_id"].iloc[0]]
    llr = build_llr_matrices(backbone, seqs)
    batches, idx = build_variant_batches(sub, seqs, AA_TO_ID, batch_size=24,
                                         marginals=llr,
                                         target_col="score_rank", shuffle_seed=0)
    m = make_model(backbone, len(idx), head_hidden=64)
    for p in backbone.parameters():
        p.requires_grad_(False)
    fn = VariantLoss(LossConfig(kind=loss_kind))
    opt = torch.optim.AdamW(m.head_parameters(), lr=3e-3)

    def epoch_loss(train: bool):
        tot = 0.0
        for b in batches:
            out = m(b)
            l, _ = fn(out.pred, b.target)
            if train:
                opt.zero_grad(set_to_none=True)
                l.backward()
                opt.step()
            tot += float(l.detach())
        return tot / len(batches)

    first = epoch_loss(False)
    pred0 = torch.cat([m(b).pred.detach() for b in batches])
    y = torch.cat([b.target for b in batches])
    rho0 = _spearman(pred0.numpy(), y.numpy())
    for _ in range(40):
        epoch_loss(True)
    last = epoch_loss(False)
    pred1 = torch.cat([m(b).pred.detach() for b in batches])
    rho1 = _spearman(pred1.numpy(), y.numpy())

    assert last < first * 0.9, (first, last)
    assert rho1 > rho0 + 0.05, (rho0, rho1)
    assert rho1 > 0.6, rho1


def test_interaction_head_beats_the_additive_ablation_on_doubles(tables, backbone):
    """The point of the whole architecture, in miniature.

    Train on one score set's singles+doubles with and without the interaction
    path; the synthetic ground truth has real pairwise couplings, so the
    additive model must be strictly worse on the depth-2 rows.
    """
    variants, seqs, _ = tables
    ssid = variants["score_set_id"].iloc[0]
    sub = variants[(variants["score_set_id"] == ssid) & (variants["n_mut"] <= 2)]
    batches, idx = build_variant_batches(sub, seqs, AA_TO_ID, batch_size=32,
                                         target_col="score_rank", shuffle_seed=0)
    doubles = [b for b in batches]
    y_depth = torch.cat([b.n_mut for b in batches])
    y = torch.cat([b.target for b in batches])

    def fit(ablation):
        for p in backbone.parameters():
            p.requires_grad_(False)
        m = make_model(backbone, len(idx), ablation=ablation, head_hidden=64)
        fn = VariantLoss(LossConfig(kind="mse_rank"))
        opt = torch.optim.AdamW(m.head_parameters(), lr=5e-3)
        for _ in range(80):
            for b in doubles:
                opt.zero_grad(set_to_none=True)
                l, _ = fn(m(b).pred, b.target)
                l.backward()
                opt.step()
        with torch.no_grad():
            return torch.cat([m(b).pred for b in doubles])

    p_full = fit("none")
    p_add = fit("additive")
    sel = (y_depth == 2).numpy()
    rho_full = _spearman(p_full.numpy()[sel], y.numpy()[sel])
    rho_add = _spearman(p_add.numpy()[sel], y.numpy()[sel])
    assert rho_full > rho_add, (rho_full, rho_add)
