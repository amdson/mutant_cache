"""
tests/test_models_real_backbone.py
----------------------------------
The same contract checks as ``test_models_ablations.py``, but against the real
``mutant_cache.backbone.ESMC`` and ``mutant_cache.cache`` instead of the stub —
with random weights at a toy size, so no checkpoint download is needed and it
still runs on CPU in seconds.

This is the test that would have caught the two places where the stub and the
real backbone could silently drift apart: the ``variant_extra_embed`` calling
convention (the real cache does ``x = x + hook(x)``, so the hook returns a
*delta*), and ``ReferenceStates`` carrying fields the model layer must copy
rather than reconstruct.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("mutant_cache.backbone.esmc")

from mutant_cache.backbone.esmc import ESMC
from mutant_cache.baselines import build_llr_matrices
from mutant_cache.models import (
    VariantEffectModel, VariantModelConfig, build_variant_batches,
)

AA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture(scope="module")
def real():
    torch.manual_seed(0)
    bb = ESMC(64, 4, 2).eval()
    for p in bb.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list(AA), 18))
    seqs = {"T": seq}

    def mk(p):
        return f"{seq[p - 1]}{p}{[a for a in AA if a != seq[p - 1]][0]}"

    m1, m2, m3 = mk(3), mk(7), mk(12)
    muts = ["", m1, m2, m3, f"{m1}:{m2}", f"{m1}:{m2}:{m3}"]
    df = pd.DataFrame({"score_set_id": "S", "target_id": "T", "mutant": muts,
                       "n_mut": [0, 1, 1, 1, 2, 3], "score": 0.0,
                       "score_rank": 0.5})
    return bb, seqs, df, muts, (m1, m2, m3)


def _fit_ready(bb, **kw):
    torch.manual_seed(1)
    m = VariantEffectModel(bb, VariantModelConfig(n_assays=1, **kw))
    with torch.no_grad():
        for q in m.head_parameters():
            torch.nn.init.normal_(q, std=0.3 if q.dim() >= 2 else 0.1)
        m.assay_scale.weight.fill_(1.0)
    return m.eval()


def _scores(m, bb, seqs, df, muts, llr, backend=None):
    bs, _ = build_variant_batches(df, seqs, dict(bb.aa_to_id), tokenizer=bb.tokenizer, batch_size=len(muts),
                                  marginals=llr)
    assert len(bs) == 1
    b = bs[0]
    with torch.no_grad():
        p = m(b, backend=backend).pred
    return {muts[int(i)]: float(v) for i, v in zip(b.row_index, p)}, b


@pytest.fixture(scope="module")
def llr(real):
    bb, seqs, *_ = real
    return build_llr_matrices(bb, seqs)


def test_the_real_backbone_takes_var_mask(real):
    bb, *_ = real
    assert _fit_ready(bb)._var_mask_ok, \
        "the model relies on var_mask to keep padded slots at the reference"


def test_llr_matrix_is_zero_at_the_wildtype(real, llr):
    bb, seqs, *_ = real
    ids = bb.tokenizer.encode(seqs["T"])
    assert float(llr["T"].gather(1, ids[:, None]).abs().max()) < 1e-5


@pytest.mark.parametrize("ablation", ["none", "additive", "no_cross_attn",
                                      "random_ref", "position_only"])
def test_cache_and_recompute_agree(real, llr, ablation):
    bb, seqs, df, muts, _ = real
    m = _fit_ready(bb, ablation=ablation)
    c, b = _scores(m, bb, seqs, df, muts, llr, backend="cache")
    r, _ = _scores(m, bb, seqs, df, muts, llr, backend="recompute")
    worst = max(abs(c[k] - r[k]) for k in muts)
    assert worst < 1e-4, (ablation, worst)


def test_additive_ablation_is_exactly_additive_on_the_real_backbone(real, llr):
    bb, seqs, df, muts, (m1, m2, m3) = real
    o, _ = _scores(_fit_ready(bb, ablation="additive"), bb, seqs, df, muts, llr)
    wt = o[""]
    add2 = (o[m1] - wt) + (o[m2] - wt)
    add3 = add2 + (o[m3] - wt)
    assert (o[f"{m1}:{m2}"] - wt) == pytest.approx(add2, abs=1e-5)
    assert (o[f"{m1}:{m2}:{m3}"] - wt) == pytest.approx(add3, abs=1e-5)


@pytest.mark.parametrize("ablation", ["none", "no_cross_attn", "position_only"])
def test_other_ablations_are_not_additive(real, llr, ablation):
    bb, seqs, df, muts, (m1, m2, m3) = real
    o, _ = _scores(_fit_ready(bb, ablation=ablation), bb, seqs, df, muts, llr)
    wt = o[""]
    add2 = (o[m1] - wt) + (o[m2] - wt)
    assert abs((o[f"{m1}:{m2}"] - wt) - add2) > 1e-2, ablation


def test_position_only_is_invariant_to_alt_identity(real):
    bb, seqs, *_ = real
    seq = seqs["T"]
    p, ref = 6, seqs["T"][5]
    alts = [a for a in AA if a != ref][:4]
    muts = [f"{ref}{p}{a}" for a in alts]
    df = pd.DataFrame({"score_set_id": "S", "target_id": "T", "mutant": muts,
                       "n_mut": 1, "score": 0.0, "score_rank": 0.5})
    m = _fit_ready(bb, ablation="position_only")
    o, _ = _scores(m, bb, seqs, df, muts, None)
    assert max(o.values()) - min(o.values()) < 1e-6, o


def test_gradients_reach_the_head_through_the_real_cache(real, llr):
    bb, seqs, df, muts, _ = real
    m = _fit_ready(bb)
    bs, _ = build_variant_batches(df, seqs, dict(bb.aa_to_id), tokenizer=bb.tokenizer,
                                  batch_size=len(muts), marginals=llr)
    out = m(bs[0])
    out.pred.sum().backward()
    grads = {n: p.grad for n, p in m.named_parameters()
             if p.grad is not None and p.requires_grad}
    assert grads, "no parameter received a gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads.values())
    # the backbone stays frozen
    assert all(p.grad is None for p in bb.parameters())


@pytest.mark.parametrize("kw", [dict(ablation="none"), dict(ablation="none", ref_summary="attn"),
                                dict(ablation="additive")])
def test_bf16_backbone_with_fp32_head(real, llr, kw):
    """The harness default: frozen bf16 base, fp32 head. Backbone states must be
    cast at the head boundary, and predict() must hand back CPU indices even when
    batches live on the GPU."""
    bb, seqs, df, muts, _ = real
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb16 = ESMC(64, 4, 2).eval()
    bb16.load_state_dict(bb.state_dict())
    bb16 = bb16.to(device=dev, dtype=torch.bfloat16)
    m32 = _fit_ready(bb, **kw)
    m16 = _fit_ready(bb16, **kw).to(dev)
    assert m16.additive_proj.weight.dtype == torch.float32
    ref, _ = _scores(m32, bb, seqs, df, muts, llr)
    bs, _ = build_variant_batches(df, seqs, dict(bb.aa_to_id), tokenizer=bb.tokenizer, batch_size=len(muts),
                                  marginals=llr)
    idx, pred = m16.predict([b.to(dev) for b in bs])
    assert idx.device.type == "cpu" and pred.dtype == torch.float32
    got = {muts[int(i)]: float(v) for i, v in zip(idx, pred)}
    for k in muts:
        assert got[k] == pytest.approx(ref[k], abs=0.05 * (1 + abs(ref[k])))


def test_batches_substitute_the_named_residue(real):
    """`A3G` is token 3 (BOS at 0): the slot the model mutates must hold the
    wild-type residue named in the mutant string, and the reference must be the
    backbone's own encoding (ESM-C <eos> is 2, not the stub's 1)."""
    bb, seqs, df, muts, (m1, m2, m3) = real
    seq = seqs["T"]
    bs, _ = build_variant_batches(df[df.mutant == m2], seqs, dict(bb.aa_to_id),
                                  tokenizer=bb.tokenizer)
    b = bs[0]
    assert torch.equal(b.ref_ids, bb.tokenizer.encode(seq))
    assert int(b.ref_ids[-1]) == bb.tokenizer.eos_token_id
    p = int(m2[1:-1])
    assert b.positions.tolist() == [p]
    assert int(b.ref_ids[p]) == bb.aa_to_id[m2[0]] == bb.aa_to_id[seq[p - 1]]
    assert int(b.variant_ids[0, 0]) == bb.aa_to_id[m2[-1]]


def test_zero_shot_reads_the_mutated_row():
    from mutant_cache.baselines.zero_shot import score_variants
    aa = {"A": 5, "G": 6}
    M = torch.arange(10 * 64, dtype=torch.float).view(10, 64)   # entry = 64*row + col
    df = pd.DataFrame({"target_id": ["T"], "mutant": ["A3G"]})
    assert score_variants(df, {"T": M}, aa_to_id=aa)[0] == 64 * 3 + 6


def test_dual_lora_trains_through_the_model_and_never_serves_a_stale_reference(real, llr):
    """Shared adapters move the reference: in train mode it is recomputed with
    grad (so shared adapters learn through it), and an evaluation after the
    weights move must not reuse a reference cached before."""
    from mutant_cache.cache.lora import (LoRAConfig, attach_shared_lora, attach_variant_lora,
                                         detach_shared_lora, detach_variant_lora)
    bb, seqs, df, muts, _ = real
    attach_variant_lora(bb, LoRAConfig(r=4))
    attach_shared_lora(bb, LoRAConfig(r=4))
    try:
        m = _fit_ready(bb)
        bs, _ = build_variant_batches(df, seqs, dict(bb.aa_to_id), tokenizer=bb.tokenizer,
                                      batch_size=len(muts), marginals=llr)
        _, p0 = m.predict(bs)                      # caches the reference (eval)
        m.train()
        # in train mode the reference is live: shared adapters learn through it
        assert m.reference_states(bs[0]).final.requires_grad
        out = m(bs[0])
        assert out.pred.requires_grad
        out.pred.sum().backward()
        for ad in (bb.variant_adapters, bb.shared_adapters):   # lora_B is zero: its grad is the signal
            gB = [p.grad for n, p in ad.named_parameters() if n.endswith("lora_B")]
            assert all(g is not None for g in gB) and sum(g.abs().sum() for g in gB) > 0
        with torch.no_grad():
            for n, p in bb.shared_adapters.named_parameters():
                if n.endswith("lora_B"):
                    p.normal_(0, 0.05)
        _, p1 = m.predict(bs)
        m.clear_reference_cache()                  # force a fresh reference
        _, p2 = m.predict(bs)
        assert (p1 - p0).abs().max() > 1e-4
        # the post-update evaluation used the moved reference, not the cached one
        assert torch.allclose(p1, p2, atol=1e-6), (p1 - p2).abs().max()
    finally:
        detach_variant_lora(bb)
        detach_shared_lora(bb)
        for p in bb.parameters():
            p.requires_grad_(False)
            p.grad = None
