"""
The opt-in minimal readouts (cfg.head, TODO.md §2): mean_count,
sum_mean_count, wt_relative, and mut_proj (wt_relative's capacity control).

Checks: only the active head modules exist and train; mutation-order and
batch-composition invariance; ragged == padded == recompute predictions and
gradients; the WT state gathered at the right token; k = 0 (wild type) is the
zero feature; gradients through both trainable streams (shared + variant
LoRA) and none forced through a frozen reference; the exact backend runs the
same readout; the legacy mean head is untouched.

Real ESM-C architecture at a toy size, random weights (fp32).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mutant_cache.backbone.esmc import ESMC
from mutant_cache.cache import (LoRAConfig, attach_shared_lora, attach_variant_lora,
                                detach_shared_lora, detach_variant_lora, lora_parameters)
from mutant_cache.models import HEADS, VariantEffectModel, VariantModelConfig, build_variant_batches

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AA = "ACDEFGHIKLMNPQRSTVWY"
NEW = [h for h in HEADS if h != "mean"]


def _muts(seq, n=30, seed=0):
    rng = np.random.default_rng(seed)
    out = [""]
    for _ in range(n):
        ps = sorted(rng.choice(np.arange(1, len(seq) + 1), int(rng.integers(1, 5)), replace=False))
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
    out, _ = build_variant_batches(df, {"T": seq}, dict(bb.aa_to_id), tokenizer=bb.tokenizer,
                                   batch_size=bs or len(df), target_col="score")
    return [b.to(DEV) for b in out]


def _model(bb, head, **kw):
    torch.manual_seed(1)
    return VariantEffectModel(bb, VariantModelConfig(n_assays=1, minimal=True, head=head,
                                                     **kw)).to(DEV).eval()


def _by_row(m, batches, **kw):
    out = {}
    with torch.no_grad():
        for b in batches:
            p = m(b, **kw).pred
            out.update(zip(b.row_index.tolist(), p.tolist()))
    return out


def test_config_guards():
    with pytest.raises(ValueError):
        VariantModelConfig(head="wt_relative")                 # needs minimal
    with pytest.raises(ValueError):
        VariantModelConfig(minimal=True, head="mean_count", pool="sum")
    with pytest.raises(ValueError):
        VariantModelConfig(minimal=True, head="nope")


@pytest.mark.parametrize("head", NEW)
def test_only_active_modules(world, head):
    bb, _, _ = world
    m = _model(bb, head)
    names = {n.split(".")[0] for n, _ in m.named_parameters() if not n.startswith("backbone.")}
    expect = {"head_mlp"} | ({"token_proj", "residue_embed"}
                             if head in ("wt_relative", "mut_proj") else set())
    assert names == expect
    assert {id(p) for p in m.head_parameters()} == \
        {id(p) for n, p in m.named_parameters() if not n.startswith("backbone.")}
    d = bb.d_model
    assert m.head_mlp[0].in_features == {"mean_count": d + 1, "sum_mean_count": 2 * d + 1,
                                         "wt_relative": 2 * 128 + 1,
                                         "mut_proj": 2 * 128 + 1}[head]
    if head == "wt_relative":
        assert m.token_proj[0].in_features == 3 * d + 2 * 16
    if head == "mut_proj":
        assert m.token_proj[0].in_features == d + 2 * 16


def test_legacy_mean_head_unchanged(world):
    """head='mean' builds exactly the historical module set, in the historical
    order (so the same seed gives the same initialization)."""
    bb, _, _ = world
    m = _model(bb, "mean")
    keys = [k for k in m.state_dict() if not k.startswith("backbone.")]
    assert keys == ["extra_gain", "marginal_weight", "ref_query", "pos_embed.weight",
                    "alt_embed.weight", "ref_embed.weight", "assay_embed.weight",
                    "additive_proj.weight", "additive_proj.bias", "ref_proj.weight",
                    "ref_proj.bias", "inter_mlp.0.weight", "inter_mlp.0.bias",
                    "inter_mlp.2.weight", "inter_mlp.2.bias", "assay_bias.weight",
                    "assay_scale.weight", "head_mlp.0.weight", "head_mlp.0.bias",
                    "head_mlp.2.weight", "head_mlp.2.bias"]


@pytest.mark.parametrize("head", NEW)
def test_backends_agree_and_batch_invariant(world, head):
    bb, seq, df = world
    m = _model(bb, head)
    full = _batches(bb, seq, df)
    a = _by_row(m, full)
    others = [_by_row(m, _batches(bb, seq, df, bs=5)),
              _by_row(m, _batches(bb, seq, df.iloc[::3])),
              _by_row(m, full, backend="recompute")]
    m.cfg.ragged = False                                    # padded cache path
    others.append(_by_row(m, full))
    m.cfg.ragged = True
    for other in others:
        for r, v in other.items():
            assert abs(v - a[r]) <= 1e-4 * max(1.0, abs(a[r])), (head, r)


@pytest.mark.parametrize("head", NEW)
def test_mutation_order_invariance(world, head):
    bb, seq, df = world
    m = _model(bb, head)
    rev = df.copy()
    rev["mutant"] = [":".join(reversed(x.split(":"))) if x else x for x in df["mutant"]]
    a, b = _by_row(m, _batches(bb, seq, df)), _by_row(m, _batches(bb, seq, rev))
    for r in a:
        assert abs(a[r] - b[r]) <= 1e-5 * max(1.0, abs(a[r]))


@pytest.mark.parametrize("head", NEW)
def test_wild_type_is_zero_feature(world, head):
    bb, seq, df = world
    m = _model(bb, head)
    b = _batches(bb, seq, df)[0]
    wt = int((b.n_mut == 0).nonzero()[0, 0])
    with torch.no_grad():
        p = m(b).pred
        expect = m.head_mlp(torch.zeros(1, m.head_mlp[0].in_features, device=DEV)).squeeze()
    assert torch.isfinite(p).all()
    assert torch.allclose(p[wt], expect, atol=1e-6)


@pytest.mark.parametrize("head", NEW)
def test_ragged_padded_gradients(world, head):
    bb, seq, df = world
    b = _batches(bb, seq, df)[0]
    grads = []
    for ragged in (True, False):
        m = _model(bb, head, ragged=ragged).train()
        m(b).pred.square().sum().backward()
        grads.append({n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and grads[0]
    for n in grads[0]:
        assert float((grads[0][n] - grads[1][n]).abs().max()) <= \
            1e-4 * max(1.0, float(grads[1][n].abs().max())), n


def test_wt_state_gathered_at_the_same_token(world):
    """Rebuild the wt_relative prediction by hand for single mutants, from
    ref.final at the mutated token index (residue p, 1-based, is token p)."""
    bb, seq, _ = world
    m = _model(bb, "wt_relative")
    muts = [f"{seq[p - 1]}{p}{'A' if seq[p - 1] != 'A' else 'C'}" for p in (1, 7, 40)]
    b = _batches(bb, seq, _frame(muts))[0]
    assert sorted(b.positions.tolist()) == [1, 7, 40]
    ridx = m._residue_index
    with torch.no_grad():
        pred = m(b).pred
        ref = m.reference_states(b)
        for i in range(b.n):
            p_tok = int(b.positions[b.var_mask[i]][0])
            alt = int(b.variant_ids[i][b.var_mask[i]][0])
            h = bb.forward_variants(ref, torch.tensor([p_tok], device=DEV),
                                    torch.tensor([[alt]], device=DEV)).hidden[0, 0]
            hw = ref.final[0, p_tok]
            z = m.token_proj(torch.cat([h, hw, h - hw,
                                        m.residue_embed(ridx[b.ref_ids[p_tok]]),
                                        m.residue_embed(ridx[torch.tensor(alt, device=DEV)])]))
            x = torch.cat([z, z, torch.log1p(torch.ones(1, device=DEV))])
            assert torch.allclose(pred[i], m.head_mlp(x).squeeze(), atol=1e-5)


def test_residue_index_is_canonical(world):
    bb, _, _ = world
    m = _model(bb, "wt_relative")
    assert [int(m._residue_index[bb.aa_to_id[a]]) for a in AA] == list(range(20))
    others = set(range(bb.vocab_size)) - {bb.aa_to_id[a] for a in AA}
    assert all(int(m._residue_index[t]) == 20 for t in others)


def test_wt_relative_grad_through_both_streams(world):
    """With shared + variant LoRA in training, wt_relative sends gradient into
    the reference's final states (and so into the shared adapters through
    them); with a frozen reference nothing forces a graph."""
    bb, seq, df = world
    b = _batches(bb, seq, df)[0]
    attach_shared_lora(bb, LoRAConfig(r=4))
    attach_variant_lora(bb, LoRAConfig(r=4))
    try:
        g = torch.Generator(device=DEV).manual_seed(0)
        with torch.no_grad():
            for ad in (bb.shared_adapters, bb.variant_adapters):
                for n, p in ad.named_parameters():
                    if "lora_B" in n:
                        p.copy_(torch.randn(p.shape, generator=g, device=DEV) * 0.05)
        for head in ("wt_relative", "mean_count"):
            m = _model(bb, head).train()
            ref = m.reference_states(b)
            assert ref.final.requires_grad
            b.ref = ref
            try:
                pred = m(b).pred
                (gf,) = torch.autograd.grad(pred.square().sum(), ref.final,
                                            retain_graph=True, allow_unused=True)
                if head == "wt_relative":
                    assert gf is not None and float(gf.abs().sum()) > 0
                else:                                   # mutant states only
                    assert gf is None or float(gf.abs().sum()) == 0
                pred.square().sum().backward()
                for which in ("shared", "variant"):
                    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
                               for p in lora_parameters(bb, which)), (head, which)
                for p in lora_parameters(bb):
                    p.grad = None
            finally:
                b.ref = None
    finally:
        detach_shared_lora(bb)
        detach_variant_lora(bb)
        bb.freeze_base(True)
    m = _model(bb, "wt_relative").train()
    assert not m.reference_states(b).final.requires_grad


@pytest.mark.parametrize("head", NEW)
def test_exact_backend_runs_same_readout(world, head):
    """Exact is a different computation from the cache, but the readout is the
    same: finite, and the wild type (k = 0) identical across backends."""
    bb, seq, df = world
    m = _model(bb, head)
    b = _batches(bb, seq, df)[0]
    with torch.no_grad():
        pe, pc = m(b, backend="exact").pred, m(b).pred
    assert torch.isfinite(pe).all()
    wt = b.n_mut == 0
    assert wt.any() and torch.allclose(pe[wt], pc[wt])


def test_harness_records_head(tmp_path):
    import json
    from fixtures import make_synthetic_dataset
    from mutant_cache import harness
    d, _ = make_synthetic_dataset(tmp_path / "synth", n_targets=2, seq_len=20,
                                  n_singles=30, n_doubles=30, n_triples=5)
    out = tmp_path / "run"
    rc = harness.main([
        "--data-dir", str(d), "--dataset", "synth", "--split", "protein",
        "--eval-split", "test", "--backbone", "random:64:4:2", "--precision", "fp32",
        "--device", DEV, "--minimal", "--head", "wt_relative", "--max-steps", "5",
        "--eval-every", "2", "--batch-size", "16", "--baselines", "none",
        "--epsilon-depths", "--output", str(out), "--out-root", str(tmp_path)])
    assert rc == 0
    res = json.loads((out / "results.json").read_text())
    ex = res["extra"]
    assert ex["model_config"]["head"] == "wt_relative" and ex["head"] == "wt_relative"
    assert ex["active_head_params"] > 0
    assert ex["timing"]["steps"] == 5 and ex["timing"]["total_steps"] == 5
    steps = [h["step"] for h in res["history"] if h["tag"] != "init"]
    assert steps == [2, 4, 5]                         # cadence + final step only
    assert [g["name"] for g in ex["timing"]["optimizer_groups"]] == ["head"]
    assert ex["protocol"]["split_sha1"] and ex["protocol"]["n_train_unique"] > 0
    pred = pd.read_csv(out / "predictions.csv")
    assert set(pred["split"]) == {"test"}
