"""
Zero-anchored readouts and additive composition (TODO_ESM_MSR_EXPERIMENT.md §2).

``zero_anchor``: d(x) = f(x) - f(WT), so the wild type predicts exactly 0.
``composition="additive"`` (arm A, and S-add when a joint model is evaluated
that way): every real substitution is scored as its own single and the
singles are summed, d(AB) = d(A) + d(B).

Checks: config guards; WT is exactly 0 and the anchor is the head's WT output;
additive == the sum of the same singles scored jointly, on every backend and
batch composition; singles are identical under both compositions; doubles are
not (the joint path is not secretly additive); gradients reach the head and
variant LoRA under additive training and never the frozen base.

Real ESM-C architecture at a toy size, random weights (fp32).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mutant_cache.backbone.esmc import ESMC
from mutant_cache.cache import (LoRAConfig, attach_variant_lora, detach_variant_lora,
                                lora_parameters)
from mutant_cache.data.mutations import parse_mutant
from mutant_cache.models import (VariantEffectModel, VariantModelConfig, build_variant_batches,
                                 singles_batch)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AA = "ACDEFGHIKLMNPQRSTVWY"
ANCHORED = ["mean_count", "sum_mean_count", "wt_relative", "mut_proj"]


def _muts(seq, n=30, seed=0):
    rng = np.random.default_rng(seed)
    out = [""]
    for _ in range(n):
        ps = sorted(rng.choice(np.arange(1, len(seq) + 1), int(rng.integers(1, 4)), replace=False))
        out.append(":".join(f"{seq[p - 1]}{p}{rng.choice([a for a in AA if a != seq[p - 1]])}"
                            for p in ps))
    return list(dict.fromkeys(out))


def _frame(muts, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"score_set_id": "S", "target_id": "T", "mutant": muts,
                         "n_mut": [0 if m == "" else m.count(":") + 1 for m in muts],
                         "score": rng.normal(size=len(muts)), "score_rank": 0.5})


@pytest.fixture(scope="module")
def world():
    torch.manual_seed(0)
    bb = ESMC(64, 4, 2).to(DEV).eval()
    bb.freeze_base(True)
    seq = "".join(np.random.default_rng(0).choice(list(AA), 40))
    return bb, seq, _frame(_muts(seq))


def _batches(bb, seq, df, bs=None):
    out, _ = build_variant_batches(df.reset_index(drop=True), {"T": seq}, dict(bb.aa_to_id),
                                   tokenizer=bb.tokenizer, batch_size=bs or len(df),
                                   target_col="score")
    return [b.to(DEV) for b in out]


def _model(bb, head, **kw):
    torch.manual_seed(1)
    return VariantEffectModel(bb, VariantModelConfig(n_assays=1, minimal=True, head=head,
                                                     **kw)).to(DEV).eval()


def _by_mutant(m, bb, seq, df, bs=None, **kw):
    df = df.reset_index(drop=True)
    out = {}
    with torch.no_grad():
        for b in _batches(bb, seq, df, bs):
            p = m(b, **kw).pred
            out.update(zip(df["mutant"].iloc[b.row_index.cpu().numpy()], p.tolist()))
    return out


def test_config_guards():
    with pytest.raises(ValueError):
        VariantModelConfig(minimal=True, head="sum_mean_count", composition="additive")
    with pytest.raises(ValueError):
        VariantModelConfig(minimal=True, head="mean", zero_anchor=True)
    with pytest.raises(ValueError):
        VariantModelConfig(minimal=True, head="sum_mean_count", zero_anchor=True,
                           composition="nope")
    VariantModelConfig(minimal=True, head="sum_mean_count", zero_anchor=True,
                       composition="additive")


@pytest.mark.parametrize("head", ANCHORED)
def test_zero_anchor(world, head):
    bb, seq, df = world
    plain, anch = _model(bb, head), _model(bb, head, zero_anchor=True)
    anch.load_state_dict(plain.state_dict())
    a, p = _by_mutant(anch, bb, seq, df), _by_mutant(plain, bb, seq, df)
    with torch.no_grad():
        f_wt = float(plain.head_mlp(torch.zeros(1, plain.head_mlp[0].in_features,
                                                device=DEV)).squeeze())
    assert a[""] == 0.0
    for mut in a:
        assert abs(a[mut] - (p[mut] - f_wt)) <= 1e-5 * max(1.0, abs(p[mut]))


@pytest.mark.parametrize("head", ANCHORED)
def test_additive_is_sum_of_singles(world, head):
    bb, seq, df = world
    m = _model(bb, head, zero_anchor=True)
    singles = sorted({f"{r}{p}{a}" for x in df["mutant"] if x for p, r, a in parse_mutant(x)})
    s_joint = _by_mutant(m, bb, seq, _frame(singles))
    ref = {x: sum(s_joint[f"{r}{p}{a}"] for p, r, a in parse_mutant(x)) if x else 0.0
           for x in df["mutant"]}
    runs = [_by_mutant(m, bb, seq, df, composition="additive"),
            _by_mutant(m, bb, seq, df, bs=4, composition="additive"),
            _by_mutant(m, bb, seq, df.iloc[::2], composition="additive"),
            _by_mutant(m, bb, seq, df, composition="additive", backend="recompute")]
    m.cfg.ragged = False
    runs.append(_by_mutant(m, bb, seq, df, composition="additive"))
    m.cfg.ragged = True
    for run in runs:
        for mut, v in run.items():
            assert abs(v - ref[mut]) <= 1e-4 * max(1.0, abs(ref[mut])), (head, mut)


def test_singles_agree_doubles_differ(world):
    bb, seq, df = world
    m = _model(bb, "sum_mean_count", zero_anchor=True)
    j = _by_mutant(m, bb, seq, df)
    a = _by_mutant(m, bb, seq, df, composition="additive")
    k = dict(zip(df["mutant"], df["n_mut"]))
    for mut in j:
        if k[mut] <= 1:
            assert abs(j[mut] - a[mut]) <= 1e-5 * max(1.0, abs(j[mut]))
    gap = max(abs(j[x] - a[x]) for x in j if k[x] >= 2)
    assert gap > 1e-4                       # the joint path sees the other mutation


def test_singles_batch_layout(world):
    bb, seq, df = world
    b = _batches(bb, seq, df)[0]
    s, owner = singles_batch(b)
    assert s.n == int(b.var_mask.sum()) and torch.equal(s.positions, b.positions)
    assert (s.var_mask.sum(1) == 1).all() and (s.n_mut == 1).all()
    assert torch.equal(owner, b.var_mask.nonzero(as_tuple=True)[0])
    ref_tok = b.ref_ids.to(DEV)[b.positions]
    for i in range(s.n):
        j = int(s.var_mask[i].nonzero())
        assert int(s.variant_ids[i, j]) == int(b.variant_ids[owner[i], j])
        keep = torch.ones(b.k, dtype=torch.bool, device=DEV)
        keep[j] = False
        assert torch.equal(s.variant_ids[i][keep], ref_tok[keep])


def test_all_wild_type_batch(world):
    bb, seq, _ = world
    m = _model(bb, "sum_mean_count", zero_anchor=True, composition="additive")
    b = _batches(bb, seq, _frame([""]))[0]
    with torch.no_grad():
        assert m(b).pred.tolist() == [0.0]


def test_additive_gradients_reach_head_and_variant_lora(world):
    bb, seq, df = world
    attach_variant_lora(bb, LoRAConfig(r=4, alpha=8))
    try:
        for p in lora_parameters(bb, "variant"):
            if p.dim() == 2 and p.shape[0] != 4:          # lora_B: make the path nonzero
                torch.nn.init.normal_(p, std=0.02)
        m = _model(bb, "sum_mean_count", zero_anchor=True, composition="additive").train()
        b = _batches(bb, seq, df)[0]
        (m(b).pred - b.target).square().mean().backward()
        # the head's output bias cancels in f(x) - f(WT): no gradient, by design
        last_bias = m.head_mlp[-1].bias
        assert last_bias.grad is None or float(last_bias.grad.abs().sum()) == 0.0
        assert all(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.head_parameters() if p is not last_bias)
        assert all(p.grad is not None and p.grad.abs().sum() > 0
                   for p in lora_parameters(bb, "variant"))
        lora = {id(p) for p in lora_parameters(bb, "variant")}
        assert all(p.grad is None for p in bb.parameters() if id(p) not in lora)
    finally:
        for p in bb.parameters():
            p.grad = None
        detach_variant_lora(bb)
