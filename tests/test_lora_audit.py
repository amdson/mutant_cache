"""
LoRA audit (TODO.md §1): every logical projection, alone and together, on the
variant / shared / both adapter sets, through the ragged, padded and clamped-
recompute (bruteforce oracle) paths -- forward values AND adapter gradients
with nonzero adapters, fp32. Plus: first-step gradient structure of a zero-B
adapter, fp32 adapter arithmetic, optimizer groups, and reference-cache
lifetime across optimizer steps and checkpoint restores.

Real ESM-C architecture at a toy size, random weights (fp32). The exact
backend is a different computation and is deliberately not used as an oracle.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest
import torch

from mutant_cache import harness
from mutant_cache.backbone.esmc import ESMC
from mutant_cache.cache import (LoRAConfig, attach_shared_lora, attach_variant_lora,
                                detach_shared_lora, detach_variant_lora, lora_parameters)
from mutant_cache.cache.lora import ALL_TARGETS
from mutant_cache.cache.variant_cache import forward_variants, ragged_from_padded
from mutant_cache.models import VariantEffectModel, VariantModelConfig, build_variant_batches

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AA = "ACDEFGHIKLMNPQRSTVWY"


def _toy(seed=0):
    torch.manual_seed(seed)
    bb = ESMC(64, 4, 2).to(DEV).eval()
    bb.freeze_base(True)
    return bb


def _seq(n=40, seed=0):
    return "".join(np.random.default_rng(seed).choice(list(AA), n))


def _padded(bb, seq, N=8, K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = bb.tokenizer.encode(seq).to(DEV)
    pos = (torch.randperm(len(seq), generator=g)[:K] + 1).to(DEV)
    aa = torch.tensor([bb.aa_to_id[a] for a in AA])
    vid = aa[torch.randint(0, 20, (N, K), generator=g)].to(DEV)
    vm = torch.rand(N, K, generator=g) < 0.6
    vm[:, 0] = True
    vm[-1] = False                                   # one wild type in the batch
    return ids, pos, vid, vm.to(DEV)


def _randomize(ad, std=0.05, seed=0, both=True):
    g = torch.Generator(device=DEV).manual_seed(seed)
    with torch.no_grad():
        for n, p in ad.named_parameters():
            if "lora_B" in n or both:
                p.copy_(torch.randn(p.shape, generator=g, device=DEV) * std)


def _attach(bb, which, targets, compute="activation"):
    cfg = LoRAConfig(r=4, alpha=8.0, targets=targets, compute=compute)
    if which in ("variant", "both"):
        attach_variant_lora(bb, cfg)
        _randomize(bb.variant_adapters, seed=1, both=False)
    if which in ("shared", "both"):
        attach_shared_lora(bb, cfg)
        _randomize(bb.shared_adapters, seed=2, both=False)


def _detach(bb):
    detach_variant_lora(bb)
    detach_shared_lora(bb)
    bb.freeze_base(True)


def _run_path(bb, path, ids, pos, vid, vm, w):
    """Hidden at real substitutions (T, d) and adapter grads for one path."""
    for p in lora_parameters(bb):
        p.grad = None
    ref = bb.forward_reference(ids, grad=True)
    if path == "ragged":
        h = bb.forward_variants_ragged(ref, ragged_from_padded(pos, vid, vm)).hidden
    elif path == "padded":
        h = forward_variants(bb, ref, pos, vid, var_mask=vm).hidden[vm]
    else:
        h = bb.forward_variants_bruteforce(ref, pos, vid, var_mask=vm).hidden[vm]
    (h * w).sum().backward()
    grads = {i: p.grad.detach().clone() for i, p in enumerate(lora_parameters(bb))
             if p.grad is not None}
    return h.detach(), grads


def _close(a, b, tol):
    return float((a - b).abs().max()) <= tol * max(1.0, float(b.abs().max()))


@pytest.mark.parametrize("which", ["variant", "shared", "both"])
@pytest.mark.parametrize("targets", [(t,) for t in ALL_TARGETS] + [ALL_TARGETS])
def test_paths_agree_forward_and_grad(which, targets):
    bb = _toy()
    ids, pos, vid, vm = _padded(bb, _seq())
    _attach(bb, which, targets)
    try:
        T = int(vm.sum())
        w = torch.randn(T, bb.d_model, generator=torch.Generator().manual_seed(3)).to(DEV)
        res = {p: _run_path(bb, p, ids, pos, vid, vm, w)
               for p in ("bruteforce", "padded", "ragged")}
        h0, g0 = res["bruteforce"]
        # with B != 0, A gets a gradient too, on every adapted projection
        assert len(g0) == len(lora_parameters(bb))
        assert all(float(g.abs().sum()) > 0 for g in g0.values())
        for p in ("padded", "ragged"):
            h, g = res[p]
            assert _close(h, h0, 1e-4), (p, which, targets)
            assert g.keys() == g0.keys()
            for k in g0:
                assert _close(g[k], g0[k], 1e-3), (p, which, targets, k)
    finally:
        _detach(bb)


def test_adapters_are_actually_in_the_path():
    """Nonzero adapters change the variant output on every target, and never
    the padded (reference) slots."""
    bb = _toy()
    ids, pos, vid, vm = _padded(bb, _seq())
    with torch.no_grad():
        ref = bb.forward_reference(ids)
        base = bb.forward_variants(ref, pos, vid, var_mask=vm).hidden
    for t in ALL_TARGETS:
        _attach(bb, "variant", (t,))
        try:
            with torch.no_grad():
                h = bb.forward_variants(ref, pos, vid, var_mask=vm).hidden
            assert float((h - base)[vm].abs().max()) > 1e-4, t
            assert torch.equal(h[~vm], base[~vm]), t
        finally:
            _detach(bb)


def test_zero_B_first_step_gradient_structure():
    """B = 0: dL/dA = 0 exactly on the first step, dL/dB != 0. After B moves,
    A receives gradient."""
    bb = _toy()
    ids, pos, vid, vm = _padded(bb, _seq())
    attach_variant_lora(bb, LoRAConfig(r=4, targets=ALL_TARGETS))
    try:
        ref = bb.forward_reference(ids)
        rv = ragged_from_padded(pos, vid, vm)
        bb.forward_variants_ragged(ref, rv).hidden.square().sum().backward()
        A = [p for n, p in bb.variant_adapters.named_parameters() if "lora_A" in n]
        B = [p for n, p in bb.variant_adapters.named_parameters() if "lora_B" in n]
        assert all(float(p.grad.abs().max()) == 0.0 for p in A)
        assert all(float(p.grad.abs().sum()) > 0 for p in B)
        torch.optim.SGD(B, lr=1e-2).step()
        for p in A + B:
            p.grad = None
        bb.forward_variants_ragged(ref, rv).hidden.square().sum().backward()
        assert all(float(p.grad.abs().sum()) > 0 for p in A)
    finally:
        _detach(bb)


def test_fp32_adapter_compute():
    """fp32 base: 'fp32' compute is bit-identical to the default. bf16 base:
    both run, outputs stay in the activation dtype, differ from fp32 only by
    bf16 rounding, and adapter grads stay fp32."""
    seq = _seq()
    outs = {}
    for dt in (torch.float32, torch.bfloat16):
        for compute in ("activation", "fp32"):
            bb = _toy().to(dtype=dt)
            ids, pos, vid, vm = _padded(bb, seq)
            _attach(bb, "variant", ALL_TARGETS, compute=compute)
            ref = bb.forward_reference(ids)
            h = bb.forward_variants_ragged(ref, ragged_from_padded(pos, vid, vm)).hidden
            assert h.dtype == dt
            h.float().square().sum().backward()
            gs = [p.grad for p in lora_parameters(bb)]
            assert all(g is not None and g.dtype == torch.float32 for g in gs)
            outs[(dt, compute)] = h.detach().float()
    assert torch.equal(outs[(torch.float32, "activation")], outs[(torch.float32, "fp32")])
    ref32 = outs[(torch.float32, "fp32")]
    for c in ("activation", "fp32"):
        assert _close(outs[(torch.bfloat16, c)], ref32, 0.1)


# ── optimizer groups ──────────────────────────────────────────────────────────

def _args(**kw):
    a = dict(lr=3e-3, lora_lr=None, backbone_lr=1e-5, weight_decay=0.01,
             train_backbone=False)
    a.update(kw)
    return argparse.Namespace(**a)


@pytest.mark.parametrize("cfg", [
    dict(variant=True), dict(shared=True), dict(variant=True, shared=True),
    dict(train_backbone=True), dict(train_backbone=True, variant=True),
    dict(), dict(variant=True, lora_lr=1e-4)])
@pytest.mark.parametrize("minimal", [True, False])
def test_param_groups_each_active_parameter_once(cfg, minimal):
    bb = _toy()
    tb = cfg.get("train_backbone", False)
    bb.freeze_base(not tb)
    if cfg.get("variant"):
        attach_variant_lora(bb, LoRAConfig(r=4), freeze_base=not tb)
    if cfg.get("shared"):
        attach_shared_lora(bb, LoRAConfig(r=4), freeze_base=not tb)
    m = VariantEffectModel(bb, VariantModelConfig(n_assays=1, minimal=minimal)).to(DEV)
    groups = harness.build_param_groups(m, _args(train_backbone=tb, lora_lr=cfg.get("lora_lr")))
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    byname = {g["name"]: g for g in groups}
    assert byname["head"]["lr"] == 3e-3
    assert all(g["weight_decay"] == 0.01 for g in groups)
    lora_lr = cfg.get("lora_lr") or 3e-3
    for which in ("variant", "shared"):
        name = f"{which}_lora"
        if cfg.get(which):
            assert byname[name]["lr"] == lora_lr
            assert {id(p) for p in byname[name]["params"]} == \
                {id(p) for p in lora_parameters(bb, which)}
        else:
            assert name not in byname
    base = {id(p) for _, p in bb.base_parameters()}
    if tb:
        assert byname["backbone"]["lr"] == 1e-5
        assert {id(p) for p in byname["backbone"]["params"]} == base
    else:
        assert "backbone" not in byname and not (set(ids) & base)
    if minimal:      # the unused legacy head modules never reach the optimizer
        assert {id(p) for p in byname["head"]["params"]} == \
            {id(p) for p in m.head_mlp.parameters()}
    else:            # every trainable parameter is in exactly one group
        assert set(ids) == {id(p) for p in m.parameters() if p.requires_grad}


# ── reference-cache lifetime ──────────────────────────────────────────────────

def _batch(bb, seq, n=24, seed=0):
    rng = np.random.default_rng(seed)
    muts = [""]
    for _ in range(n):
        ps = sorted(rng.choice(np.arange(1, len(seq) + 1), int(rng.integers(1, 4)), replace=False))
        muts.append(":".join(f"{seq[p - 1]}{p}{rng.choice([a for a in AA if a != seq[p - 1]])}"
                             for p in ps))
    muts = list(dict.fromkeys(muts))
    df = pd.DataFrame({"score_set_id": "S", "target_id": "T", "mutant": muts,
                       "n_mut": [0 if m == "" else m.count(":") + 1 for m in muts],
                       "score": rng.normal(size=len(muts)), "score_rank": 0.5})
    bs, _ = build_variant_batches(df, {"T": seq}, dict(bb.aa_to_id), tokenizer=bb.tokenizer,
                                  batch_size=len(df), target_col="score")
    return bs[0].to(DEV)


def _fresh_pred(m, b):
    m.clear_reference_cache()
    with torch.no_grad():
        return m(b).pred.clone()


@pytest.mark.parametrize("mode", ["variant", "shared", "both", "backbone"])
def test_reference_cache_lifetime(mode):
    """After optimizer steps, evaluation scores against references computed
    with the current weights; restoring an older checkpoint never reuses
    references of the weights it replaced -- even without a mode switch."""
    bb = _toy()
    tb = mode == "backbone"
    bb.freeze_base(not tb)
    if mode in ("variant", "both"):
        attach_variant_lora(bb, LoRAConfig(r=4))
    if mode in ("shared", "both"):
        attach_shared_lora(bb, LoRAConfig(r=4))
    b = _batch(bb, _seq(30, seed=1))
    m = VariantEffectModel(bb, VariantModelConfig(n_assays=1, minimal=True)).to(DEV)
    opt = torch.optim.AdamW(harness.build_param_groups(
        m, _args(train_backbone=tb, lr=1e-2, lora_lr=1e-2, backbone_lr=1e-2)))
    m.eval()
    with torch.no_grad():
        m(b)                                          # populate the cache
    moves = mode != "variant"
    assert m._reference_moves() == moves
    state0 = {k: v.detach().clone() for k, v in m.state_dict().items()}
    for _ in range(3):
        m.train()
        opt.zero_grad()
        m(b).pred.square().mean().backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        got = m(b).pred.clone()
    if not moves:                                     # frozen reference: still memoized
        assert m._ref_cache
    assert torch.allclose(got, _fresh_pred(m, b), atol=1e-6)
    # restore the old checkpoint while staying in eval mode (no mode switch)
    with torch.no_grad():
        m(b)
    m.load_state_dict(state0)
    with torch.no_grad():
        restored = m(b).pred.clone()
    assert torch.allclose(restored, _fresh_pred(m, b), atol=1e-6)
    if moves:
        assert not torch.allclose(restored, got)      # the steps did move it
    _detach(bb)


def test_frozen_variant_lora_reference_has_no_grad():
    bb = _toy()
    attach_variant_lora(bb, LoRAConfig(r=4))
    m = VariantEffectModel(bb, VariantModelConfig(n_assays=1, minimal=True)).to(DEV).train()
    ref = m.reference_states(_batch(bb, _seq(30, seed=2)))
    assert not ref.final.requires_grad and not ref.layer_inputs[0].requires_grad
    _detach(bb)
