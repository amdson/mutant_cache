"""
tests/test_models_ablations.py
------------------------------
Each of DATASETS.md's four ablations must *change the computation*, in the
specific way its "what a tie means" column assumes.  An ablation flag that
silently does nothing is worse than no ablation at all: it turns a tie into a
false negative for the whole architecture.

So each one is checked by its defining invariant rather than by "the number
moved":

  additive       predictions are exactly additive across mutations
  no_cross_attn  predictions do not depend on the reference states at all
  random_ref     predictions change when the pretrained states are replaced,
                 and are stable across calls (a fixed random feature map, not
                 resampled noise)
  position_only  predictions are invariant to the identity of the alt residue
"""
from __future__ import annotations

import pytest
import torch

from fixtures import AA, AA_TO_ID, make_stub_backbone, make_synthetic_dataset

from mutant_cache.models import (
    VariantEffectModel, VariantModelConfig, build_variant_batches,
)
from mutant_cache.models.data_compat import load_tables


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("abl")
    out, _ = make_synthetic_dataset(d, n_targets=1, seq_len=14, n_singles=40,
                                    n_doubles=40, n_triples=10, seed=3)
    t = load_tables(out)
    seqs = dict(zip(t["targets"]["target_id"], t["targets"]["sequence"]))
    bb = make_stub_backbone(d_model=32, n_layers=2, n_heads=4, seed=2)
    return t["variants"], seqs, bb


def _model(bb, n_assays=1, **kw):
    torch.manual_seed(0)
    m = VariantEffectModel(bb, VariantModelConfig(n_assays=n_assays, **kw))
    # a zero-initialised head makes every ablation look identical, so give the
    # additive projection and the interaction MLP real weights first.
    with torch.no_grad():
        for p in m.head_parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, std=0.3)
            else:
                torch.nn.init.normal_(p, std=0.1)
        m.assay_scale.weight.fill_(1.0)
    m.eval()
    return m


def _batch(seqs, tid, mutants, aa=AA_TO_ID):
    """A one-off VariantBatch from an explicit list of mutant strings.

    Note the rows are *reordered*: ``build_variant_batches`` sorts by position
    tuple so that variants sharing sites land together and K stays small.  Use
    ``_preds`` rather than positional unpacking.
    """
    import pandas as pd
    df = pd.DataFrame({
        "score_set_id": ["S"] * len(mutants), "target_id": [tid] * len(mutants),
        "mutant": mutants,
        "n_mut": [0 if m == "" else m.count(":") + 1 for m in mutants],
        "score": [0.0] * len(mutants), "score_rank": [0.5] * len(mutants)})
    b, _ = build_variant_batches(df, seqs, aa, batch_size=len(mutants))
    assert len(b) == 1, "the test batch must be a single forward"
    return b[0]


def _preds(m, seqs, tid, mutants, **kw):
    """{mutant: prediction}, resolved through row_index so row order is moot."""
    b = _batch(seqs, tid, mutants)
    with torch.no_grad():
        p = m(b, **kw).pred
    return {mutants[int(i)]: float(v) for i, v in zip(b.row_index, p)}, b


def _wt_and_singles(seq, positions):
    out = []
    for p in positions:
        ref = seq[p - 1]
        alt = next(a for a in AA if a != ref)
        out.append(f"{ref}{p}{alt}")
    return out


# ── additive ─────────────────────────────────────────────────────────────────

def test_additive_ablation_is_exactly_additive(world):
    """f(A:B) - f(wt) == (f(A) - f(wt)) + (f(B) - f(wt)), numerically."""
    variants, seqs, bb = world
    tid = sorted(seqs)[0]
    seq = seqs[tid]
    a, b = _wt_and_singles(seq, [3, 9])
    double = f"{a}:{b}"
    muts = ["", a, b, double]

    p, _ = _preds(_model(bb, ablation="additive"), seqs, tid, muts)
    wt = p[""]
    assert (p[double] - wt) == pytest.approx((p[a] - wt) + (p[b] - wt), abs=1e-5), p

    # and the unablated model is NOT additive — otherwise the ablation is vacuous
    q, _ = _preds(_model(bb, ablation="none"), seqs, tid, muts)
    wt2 = q[""]
    assert abs((q[double] - wt2) - ((q[a] - wt2) + (q[b] - wt2))) > 1e-3, q


def test_additive_ablation_is_additive_at_depth_3(world):
    variants, seqs, bb = world
    tid = sorted(seqs)[0]
    seq = seqs[tid]
    a, b, c = _wt_and_singles(seq, [2, 6, 11])
    triple = f"{a}:{b}:{c}"
    p, _ = _preds(_model(bb, ablation="additive"), seqs, tid,
                  ["", a, b, c, triple])
    wt = p[""]
    assert (p[triple] - wt) == pytest.approx(
        (p[a] - wt) + (p[b] - wt) + (p[c] - wt), abs=1e-5), p


def test_additive_ablation_masks_variant_variant_attention(world):
    """The hidden state of a variant token must not see its co-mutations.

    Both multi-mutants go through the *same* forward, so the position union
    (and therefore the set of withheld reference keys) is identical for the
    two rows: the only thing that can differ is variant-variant attention.
    """
    variants, seqs, bb = world
    tid = sorted(seqs)[0]
    seq = seqs[tid]
    a, b, c = _wt_and_singles(seq, [4, 7, 12])
    for ablation, same in (("additive", True), ("none", False)):
        m = _model(bb, ablation=ablation)
        batch = _batch(seqs, tid, [f"{a}:{b}", f"{a}:{c}"])
        slot = int((batch.positions == 4 + 1).nonzero()[0])    # the `a` token
        with torch.no_grad():
            h = m(batch).hidden
        row = {batch.row_index[i].item(): i for i in range(batch.n)}
        ha, hb = h[row[0], slot], h[row[1], slot]
        close = torch.allclose(ha, hb, atol=1e-5)
        assert close is same, (ablation, float((ha - hb).abs().max()))


def test_padded_slots_are_blocked_as_attention_keys(world):
    """A padded slot must not act as a key, or a variant's hidden state would
    depend on the *residues* its batch-mates substituted."""
    variants, seqs, bb = world
    tid = sorted(seqs)[0]
    seq = seqs[tid]
    a = _wt_and_singles(seq, [5])[0]
    p = 8
    ref = seq[p - 1]
    alt1, alt2 = [x for x in AA if x != ref][:2]
    m = _model(bb, ablation="none")
    with torch.no_grad():
        h1 = m(_batch(seqs, tid, [a, f"{ref}{p}{alt1}"])).hidden
        h2 = m(_batch(seqs, tid, [a, f"{ref}{p}{alt2}"])).hidden
    # the union is {5, 8} in both, so `a`'s token sees the same reference keys;
    # the co-variant's identity differs and must not reach it.
    assert torch.allclose(h1[0], h2[0], atol=1e-6), \
        float((h1[0] - h2[0]).abs().max())


def test_var_mask_makes_the_score_batch_independent(world):
    """A variant's score must not depend on who it was batched with.

    This is what passing ``var_mask`` through to the backbone buys: padded
    slots are clamped to the reference residual and stay visible as reference
    keys, so a single mutant sees the complete reference whatever the position
    union is.  Without it, padding withholds the reference key at every
    position a batch-mate mutates and the score drifts; ``exact_position_
    groups`` is the remedy for a backbone that has no ``var_mask``.
    """
    import pandas as pd

    variants, seqs, bb = world
    tid = sorted(seqs)[0]
    seq = seqs[tid]
    p = 5
    ref = seq[p - 1]
    muts = [f"{ref}{p}{x}" for x in [a for a in AA if a != ref][:3]]
    other = _wt_and_singles(seq, [11])[0]

    def run(m, mutants, **kw):
        df = pd.DataFrame({"score_set_id": ["S"] * len(mutants),
                           "target_id": [tid] * len(mutants),
                           "mutant": mutants,
                           "n_mut": [1] * len(mutants),
                           "score": 0.0, "score_rank": 0.5})
        bs, _ = build_variant_batches(df, seqs, AA_TO_ID,
                                      batch_size=len(mutants), **kw)
        out = {}
        with torch.no_grad():
            for b in bs:
                for i, v in zip(b.row_index, m(b).pred):
                    out[mutants[int(i)]] = float(v)
        return out

    m = _model(bb, ablation="none")
    assert m._var_mask_ok, "the stub must advertise var_mask like the real backbone"
    alone = run(m, [muts[0]])[muts[0]]
    padded = run(m, muts + [other])[muts[0]]
    assert padded == pytest.approx(alone, abs=1e-4), (alone, padded)

    # the fallback path, for a §1 backbone that does not take var_mask
    m2 = _model(bb, ablation="none", pass_var_mask="never")
    assert not m2._var_mask_ok
    alone2 = run(m2, [muts[0]])[muts[0]]
    padded2 = run(m2, muts + [other])[muts[0]]
    exact2 = run(m2, muts + [other], exact_position_groups=True)[muts[0]]
    assert abs(padded2 - alone2) > 1e-3, (alone2, padded2)
    assert exact2 == pytest.approx(alone2, abs=1e-6)


# ── no cross-attention ───────────────────────────────────────────────────────

def test_no_cross_attn_ignores_the_reference_entirely(world):
    variants, seqs, bb = world
    tid = sorted(seqs)[0]
    seq = seqs[tid]
    muts = _wt_and_singles(seq, [3, 8])
    batch = _batch(seqs, tid, muts + [":".join(muts)])

    for ablation, expect_same in (("no_cross_attn", True), ("none", False)):
        m = _model(bb, ablation=ablation)
        with torch.no_grad():
            ref = m.reference_states(batch)
            p0 = m(batch).pred.clone()
            # scramble the cached reference states; a model that reads them
            # must change its mind.
            for t in ref.layer_inputs:
                t.add_(torch.randn_like(t) * 3.0)
            ref.final.add_(torch.randn_like(ref.final) * 3.0)
            p1 = m(batch).pred
        same = torch.allclose(p0, p1, atol=1e-5)
        m.clear_reference_cache()
        assert same is expect_same, (ablation, float((p0 - p1).abs().max()))


# ── random reference states ──────────────────────────────────────────────────

def test_random_ref_changes_predictions_and_is_stable(world):
    variants, seqs, bb = world
    tid = sorted(seqs)[0]
    batch = _batch(seqs, tid, _wt_and_singles(seqs[tid], [2, 5, 10]))

    real = _model(bb, ablation="none")
    rand = _model(bb, ablation="random_ref")
    with torch.no_grad():
        p_real = real(batch).pred
        p_rand_1 = rand(batch).pred
        p_rand_2 = rand(batch).pred
    assert not torch.allclose(p_real, p_rand_1, atol=1e-3)
    # a *fixed* random feature map, not resampled noise: twice is the same
    assert torch.allclose(p_rand_1, p_rand_2, atol=1e-7)


def test_random_ref_matches_the_real_scale(world):
    variants, seqs, bb = world
    tid = sorted(seqs)[0]
    batch = _batch(seqs, tid, _wt_and_singles(seqs[tid], [2, 5]))
    m = _model(bb, ablation="random_ref")
    real = m.reference_states(batch)
    fake = m._randomize(real, tid)
    for a, b in zip(real.layer_inputs, fake.layer_inputs):
        assert a.shape == b.shape
        assert float(b.std()) == pytest.approx(float(a.std()), rel=0.4)


# ── position only ────────────────────────────────────────────────────────────

def test_position_only_is_invariant_to_the_alt_residue(world):
    variants, seqs, bb = world
    tid = sorted(seqs)[0]
    seq = seqs[tid]
    p = 6
    ref = seq[p - 1]
    alts = [a for a in AA if a != ref][:4]
    batch = _batch(seqs, tid, [f"{ref}{p}{a}" for a in alts])

    m = _model(bb, ablation="position_only")
    with torch.no_grad():
        pred = m(batch).pred
    assert float(pred.max() - pred.min()) < 1e-6, pred

    m2 = _model(bb, ablation="none")
    with torch.no_grad():
        pred2 = m2(batch).pred
    assert float(pred2.max() - pred2.min()) > 1e-3, pred2


def test_position_only_still_distinguishes_positions(world):
    variants, seqs, bb = world
    tid = sorted(seqs)[0]
    seq = seqs[tid]
    muts = _wt_and_singles(seq, [2, 7, 11])
    batch = _batch(seqs, tid, muts)
    m = _model(bb, ablation="position_only")
    with torch.no_grad():
        pred = m(batch).pred
    assert float(pred.max() - pred.min()) > 1e-3, pred


def test_position_only_drops_the_marginal_prior():
    cfg = VariantModelConfig(ablation="position_only")
    assert cfg.use_marginals is False and cfg.use_alt_residue is False
    assert VariantModelConfig(ablation="additive").interaction is False
    assert VariantModelConfig(ablation="no_cross_attn").cross_attention is False
    assert VariantModelConfig(ablation="random_ref").random_reference is True


# ── backends ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ablation", ["none", "additive", "no_cross_attn",
                                      "position_only"])
def test_both_backends_agree_under_every_ablation(world, ablation):
    """The recompute control must produce the same numbers as the cache path
    with the same weights — that is what makes the throughput ratio a free
    lunch rather than a trade."""
    variants, seqs, bb = world
    tid = sorted(seqs)[0]
    batch = _batch(seqs, tid, _wt_and_singles(seqs[tid], [3, 6, 9]) +
                   [":".join(_wt_and_singles(seqs[tid], [3, 6]))])
    m = _model(bb, ablation=ablation)
    with torch.no_grad():
        a = m(batch, backend="cache").pred
        b = m(batch, backend="recompute").pred
    assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())
