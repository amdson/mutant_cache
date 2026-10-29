"""
The minimal model (cfg.minimal / harness --minimal): mutated tokens through the
cache, mean-pooled, one MLP. Checks it is exactly that -- no hook, no prior, only
MLP parameters trained -- and that its cache paths agree with their oracles.

Real ESM-C architecture at a toy size, random weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mutant_cache.backbone.esmc import ESMC
from mutant_cache.models import VariantEffectModel, VariantModelConfig, build_variant_batches

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture(scope="module")
def world():
    torch.manual_seed(0)
    bb = ESMC(64, 4, 2).to(DEV).eval()
    for p in bb.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list(AA), 40))
    muts = [""]
    for _ in range(30):
        ps = sorted(rng.choice(np.arange(1, 41), int(rng.integers(1, 4)), replace=False))
        muts.append(":".join(f"{seq[p - 1]}{p}{rng.choice([a for a in AA if a != seq[p - 1]])}"
                             for p in ps))
    muts = list(dict.fromkeys(muts))
    df = pd.DataFrame({"score_set_id": "S", "target_id": "T", "mutant": muts,
                       "n_mut": [0 if m == "" else m.count(":") + 1 for m in muts],
                       "score": rng.normal(size=len(muts)), "score_rank": 0.5})
    bs, _ = build_variant_batches(df, {"T": seq}, dict(bb.aa_to_id),
                                  tokenizer=bb.tokenizer, batch_size=len(df))
    return bb, bs[0].to(DEV)


def _model(bb, **kw):
    torch.manual_seed(1)
    return VariantEffectModel(bb, VariantModelConfig(n_assays=1, minimal=True, **kw)).to(DEV).eval()


def test_minimal_is_plain(world):
    bb, b = world
    m = _model(bb)
    c = m.cfg
    assert not (c.use_marginals or c.use_position_embedding or c.use_alt_residue
                or c.use_ref_residue or c.use_ref_summary or c.assay_in_token)
    assert m._extra_embed(b) is None
    assert {id(p) for p in m.head_parameters()} == {id(p) for p in m.head_mlp.parameters()}
    # predictions ignore marginals entirely
    with torch.no_grad():
        p0 = m(b).pred.clone()
        if b.marginals is not None:
            b.marginals.add_(100.0)
        assert torch.allclose(m(b).pred, p0)


def test_minimal_rejects_ablation():
    with pytest.raises(ValueError):
        VariantModelConfig(minimal=True, ablation="additive")


@pytest.mark.parametrize("pool", ["mean", "sum", "max"])
def test_minimal_ragged_matches_padded_and_recompute(world, pool):
    bb, b = world
    rag, pad = _model(bb, pool=pool, ragged=True), _model(bb, pool=pool, ragged=False)
    pad.load_state_dict(rag.state_dict())
    with torch.no_grad():
        pr, pp = rag(b).pred, pad(b).pred
        prc = rag(b, backend="recompute").pred
    tol = 1e-4 * max(1.0, float(pp.abs().max()))
    assert float((pr - pp).abs().max()) <= tol
    assert float((pr - prc).abs().max()) <= tol


def test_minimal_trains_head_and_lora(world):
    from mutant_cache.cache import LoRAConfig, attach_variant_lora, detach_variant_lora
    bb, b = world
    attach_variant_lora(bb, LoRAConfig(r=4))
    try:
        m = _model(bb).train()
        m(b).pred.sum().backward()
        assert all(p.grad is not None for p in m.head_mlp.parameters())
        from mutant_cache.cache.lora import lora_parameters
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
                   for p in lora_parameters(bb, "variant"))
    finally:
        detach_variant_lora(bb)
