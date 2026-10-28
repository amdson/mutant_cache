"""
VariantEffectModel through the ragged cache vs the padded cache (cfg.ragged
False): same predictions and gradients for every ablation / pool / LoRA, on a
batch whose variants mutate different sites (so the padded path pads heavily).
Plus the exact backend (full forward of every mutant) against ``model(ids)``.

Real ESM-C architecture at a toy size, random weights; on CUDA the ragged path
runs flex, on CPU its dense-mask oracle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mutant_cache.backbone.esmc import ESMC
from mutant_cache.cache import LoRAConfig, attach_variant_lora, detach_variant_lora
from mutant_cache.cache.variant_cache import forward_reference, forward_variants_exact
from mutant_cache.models import (
    ABLATIONS, VariantEffectModel, VariantModelConfig, build_variant_batches,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture(scope="module")
def real():
    torch.manual_seed(0)
    bb = ESMC(64, 4, 2).to(DEV).eval()
    for p in bb.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list(AA), 40))
    muts = [""]
    for _ in range(40):
        k = int(rng.integers(1, 4))
        ps = sorted(rng.choice(np.arange(1, 41), k, replace=False))
        muts.append(":".join(f"{seq[p - 1]}{p}{rng.choice([a for a in AA if a != seq[p - 1]])}"
                             for p in ps))
    muts = list(dict.fromkeys(muts))
    df = pd.DataFrame({"score_set_id": "S", "target_id": "T", "mutant": muts,
                       "n_mut": [0 if m == "" else m.count(":") + 1 for m in muts],
                       "score": rng.normal(size=len(muts)), "score_rank": 0.5})
    return bb, {"T": seq}, df


def _model(bb, ragged, **kw):
    torch.manual_seed(1)
    m = VariantEffectModel(bb, VariantModelConfig(n_assays=1, ragged=ragged, **kw)).to(DEV)
    with torch.no_grad():
        for q in m.head_parameters():
            torch.nn.init.normal_(q, std=0.3 if q.dim() >= 2 else 0.1)
        m.assay_scale.weight.fill_(1.0)
    return m.eval()


def _batch(bb, seqs, df):
    bs, _ = build_variant_batches(df, seqs, dict(bb.aa_to_id), tokenizer=bb.tokenizer,
                                  batch_size=len(df))
    assert len(bs) == 1
    return bs[0].to(DEV)


def _close(a, b, tol=1e-5):
    return float((a - b).abs().max()) <= tol * max(1.0, float(b.abs().max()))


@pytest.mark.parametrize("ablation", ABLATIONS)
@pytest.mark.parametrize("pool", ["mean", "max", "sum"])
def test_ragged_matches_padded(real, ablation, pool):
    bb, seqs, df = real
    b = _batch(bb, seqs, df)
    with torch.no_grad():
        p_r = _model(bb, True, ablation=ablation, pool=pool)(b)
        p_p = _model(bb, False, ablation=ablation, pool=pool)(b)
    assert b.k > 10                          # the padded path really is padding
    for f in ("pred", "additive", "interaction", "token_score"):
        assert _close(getattr(p_r, f), getattr(p_p, f)), (f, ablation, pool)


@pytest.mark.parametrize("ablation", ["none", "additive"])
def test_ragged_matches_padded_with_variant_lora(real, ablation):
    bb, seqs, df = real
    attach_variant_lora(bb, LoRAConfig(r=4, targets=("q", "k", "v", "o", "gate", "up", "down")))
    try:
        g = torch.Generator(device=DEV).manual_seed(0)
        with torch.no_grad():
            for n, p in bb.variant_adapters.named_parameters():
                if "lora_B" in n:
                    p.copy_(torch.randn(p.shape, generator=g, device=DEV) * 0.05)
        b = _batch(bb, seqs, df)
        with torch.no_grad():
            p_r = _model(bb, True, ablation=ablation)(b).pred
            p_p = _model(bb, False, ablation=ablation)(b).pred
        assert _close(p_r, p_p)
    finally:
        detach_variant_lora(bb)


@pytest.mark.parametrize("ablation", ["none", "additive"])
def test_ragged_gradients_match_padded(real, ablation):
    bb, seqs, df = real
    b = _batch(bb, seqs, df)
    grads = []
    for ragged in (True, False):
        m = _model(bb, ragged, ablation=ablation).train()
        m(b).pred.square().sum().backward()
        grads.append({n: p.grad.clone() for n, p in m.named_parameters()
                      if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and grads[0]
    for n in grads[0]:
        assert _close(grads[0][n], grads[1][n], 1e-4), n


def test_ragged_prediction_independent_of_batchmates(real):
    """The property var_mask bought for the padded path holds for ragged."""
    bb, seqs, df = real
    m = _model(bb, True)
    sub = df.iloc[::2]
    with torch.no_grad():
        bf, bh = _batch(bb, seqs, df), _batch(bb, seqs, sub.reset_index(drop=True))
        full, half = m(bf).pred, m(bh).pred
    # batches are sorted by position tuple: align by source row
    by_row = dict(zip(bf.row_index.tolist(), full))
    expect = torch.stack([by_row[int(sub.index[i])] for i in bh.row_index.tolist()])
    assert _close(expect, half)


# ── exact backend ─────────────────────────────────────────────────────────────

def _mutant_ids(bb, seq, pos, vid):
    ids = bb.tokenizer.encode(seq).to(DEV)[None].repeat(vid.shape[0], 1)
    ids[:, pos] = vid
    return ids


def test_exact_is_the_full_forward(real):
    bb, seqs, _ = real
    seq = seqs["T"]
    ref = forward_reference(bb, bb.tokenizer.encode(seq))
    g = torch.Generator().manual_seed(0)
    pos = (torch.randperm(len(seq), generator=g)[:5] + 1).to(DEV)
    aa = torch.tensor([bb.aa_to_id[a] for a in AA])
    vid = aa[torch.randint(0, 20, (6, 5), generator=g)].to(DEV)
    with torch.no_grad():
        ex = forward_variants_exact(bb, ref, pos, vid)
        full = bb(_mutant_ids(bb, seq, pos, vid))
        cache = bb.forward_variants(ref, pos, vid)
    assert _close(ex.hidden, full.hidden[:, pos])
    assert _close(ex.logits, full.logits[:, pos], 1e-4)
    # and the cache is not (the approximation is real)
    assert not _close(cache.hidden, full.hidden[:, pos], 1e-3)


def test_exact_chunking_and_wildtype(real):
    bb, seqs, _ = real
    seq = seqs["T"]
    ref = forward_reference(bb, bb.tokenizer.encode(seq))
    pos = torch.tensor([3, 9, 20], device=DEV)
    aa = torch.tensor([bb.aa_to_id[a] for a in AA], device=DEV)
    vid = aa[torch.arange(30, device=DEV).reshape(10, 3) % 20]
    with torch.no_grad():
        a = forward_variants_exact(bb, ref, pos, vid)
        b = forward_variants_exact(bb, ref, pos, vid, max_tokens=3 * ref.seq_len)
        wt = forward_variants_exact(bb, ref, pos, ref.ids[pos][None])
    assert _close(a.hidden, b.hidden)
    assert _close(wt.hidden[0], ref.final[0, pos])


def test_model_exact_backend_runs_and_differs(real):
    bb, seqs, df = real
    b = _batch(bb, seqs, df)
    m = _model(bb, True)
    with torch.no_grad():
        p_c = m(b).pred
        p_e = m(b, backend="exact").pred
        p_r = m(b, backend="recompute").pred
    assert torch.isfinite(p_e).all()
    assert _close(p_c, p_r)                  # recompute = same approximation
    assert not _close(p_c, p_e, 1e-4)        # exact is a different model


def test_exact_chunking_with_batch_hooks(real):
    """Regression: chunking must not hand batch-shaped hooks a chunk. The
    model's padded extra-embed hook closes over all N variants, and an (N,K,K)
    attention mask is per variant; both crashed / misaligned when exact
    recursed on slices of N."""
    bb, seqs, _ = real
    seq = seqs["T"]
    ref = forward_reference(bb, bb.tokenizer.encode(seq))
    g = torch.Generator().manual_seed(3)
    N, K = 23, 4
    pos = (torch.randperm(len(seq), generator=g)[:K] + 1).to(DEV)
    aa = torch.tensor([bb.aa_to_id[a] for a in AA])
    vid = aa[torch.randint(0, 20, (N, K), generator=g)].to(DEV)
    vm = (torch.rand(N, K, generator=g) < 0.7).to(DEV)
    vm[:, 0] = True
    delta = torch.randn(N, K, bb.d_model, generator=g).to(DEV) * 0.1   # per variant
    am = (torch.rand(N, K, K, generator=g) < 0.6).to(DEV) | torch.eye(K, dtype=torch.bool, device=DEV)
    bb.variant_extra_embed = lambda emb: delta.to(emb.dtype)
    bb.variant_attn_mask = am
    try:
        with torch.no_grad():
            one = forward_variants_exact(bb, ref, pos, vid, vm)
            many = forward_variants_exact(bb, ref, pos, vid, vm, max_tokens=5 * ref.seq_len)
    finally:
        bb.variant_extra_embed = None
        bb.variant_attn_mask = None
    assert _close(one.hidden, many.hidden)
