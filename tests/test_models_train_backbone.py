"""
--train-backbone through the cache: the reference states must follow the
weights. Before the fix they were memoized at the first forward and never
recomputed, so a trainable base silently scored variants against a stale
reference (shared LoRA already handled this; a trainable base did not).

Real ESM-C architecture at a toy size, random weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from mutant_cache.backbone.esmc import ESMC
from mutant_cache.models import VariantEffectModel, VariantModelConfig, build_variant_batches

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AA = "ACDEFGHIKLMNPQRSTVWY"


def _setup(trainable: bool):
    torch.manual_seed(0)
    bb = ESMC(64, 4, 2).to(DEV).eval()
    for p in bb.parameters():
        p.requires_grad_(trainable)
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list(AA), 30))
    muts = ["", f"{seq[2]}3A", f"{seq[9]}10W", f"{seq[2]}3A:{seq[9]}10W"]
    df = pd.DataFrame({"score_set_id": "S", "target_id": "T", "mutant": muts,
                       "n_mut": [0, 1, 1, 2], "score": rng.normal(size=4),
                       "score_rank": 0.5})
    torch.manual_seed(1)
    m = VariantEffectModel(bb, VariantModelConfig(n_assays=1)).to(DEV)
    with torch.no_grad():
        for q in m.head_parameters():
            torch.nn.init.normal_(q, std=0.3 if q.dim() >= 2 else 0.1)
        m.assay_scale.weight.fill_(1.0)
    bs, _ = build_variant_batches(df, {"T": seq}, dict(bb.aa_to_id),
                                  tokenizer=bb.tokenizer, batch_size=len(df))
    return m.eval(), bs[0].to(DEV)


def _fresh(m, b):
    m.clear_reference_cache()
    with torch.no_grad():
        return m(b).pred.clone()


def test_trainable_base_reference_follows_weights():
    m, b = _setup(trainable=True)
    assert m._reference_moves()
    with torch.no_grad():
        m(b)                                       # would memoize, pre-fix
    m.train()                                      # an optimizer step happens in train mode
    with torch.no_grad():
        for p in m.backbone.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    m.eval()
    with torch.no_grad():
        got = m(b).pred.clone()
    assert torch.allclose(got, _fresh(m, b), atol=1e-5)


def test_trainable_base_gets_gradient_through_reference():
    m, b = _setup(trainable=True)
    m.train()
    m(b).pred.sum().backward()
    # attention projections only act on the variant tokens through the
    # reference keys/values at non-variant positions, so a nonzero grad on
    # every layer's qkv says the reference stream carries gradient
    g = [p.grad for n, p in m.backbone.named_parameters() if "qkv" in n or "layernorm_qkv" in n]
    assert g and all(x is not None and float(x.abs().sum()) > 0 for x in g)


def test_frozen_base_reference_still_memoized():
    m, b = _setup(trainable=False)
    assert not m._reference_moves()
    with torch.no_grad():
        m(b)
    assert m._ref_cache, "a frozen base should memoize the reference"
