"""Variant-only LoRA: reference stream untouched, zero-B == base, efficient ==
bruteforce with adapters, bf16 base + fp32 adapters. Shared LoRA (both
streams): zero-B == base, wild-type identity still exact, efficient ==
bruteforce with both sets, gradients through both streams."""
import pytest
import torch

from mutant_cache.backbone import load_esmc
from mutant_cache.cache import (LoRAConfig, attach_shared_lora, attach_variant_lora,
                                base_fingerprint, cast_base, detach_shared_lora,
                                detach_variant_lora, lora_parameters, lora_state_dict)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


@pytest.fixture(scope="module")
def model():
    return load_esmc("esmc_300m", device=DEV, dtype=torch.float32)


def _variants(model, K=4, N=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = len(SEQ) + 2
    pos = (torch.randperm(L - 2, generator=g)[:K] + 1).to(DEV)
    aa = torch.tensor(list(model.aa_to_id.values()))
    vid = aa[torch.randint(0, 20, (N, K), generator=g)].to(DEV)
    return pos, vid


def _randomize_B(model, std=0.02, seed=0, which="variant_adapters"):
    g = torch.Generator(device=DEV).manual_seed(seed)
    with torch.no_grad():
        for n, p in getattr(model, which).named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g, device=DEV) * std)


def test_reference_bit_identical_with_random_adapters(model):
    ids = model.tokenizer.encode(SEQ)
    detach_variant_lora(model)
    with torch.no_grad():
        ref0 = model.forward_reference(ids)
        full0 = model(ids[None])
    attach_variant_lora(model, LoRAConfig(r=8, targets=("q", "k", "v", "o", "gate", "up", "down")))
    _randomize_B(model, std=0.05)
    try:
        with torch.no_grad():
            ref1 = model.forward_reference(ids)
            full1 = model(ids[None])
        assert all(torch.equal(a, b) for a, b in zip(ref0.layer_inputs, ref1.layer_inputs))
        assert torch.equal(ref0.final, ref1.final)
        assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                   for a, b in zip(ref0.kv, ref1.kv))
        assert torch.equal(full0.logits, full1.logits)
        # ...while the variant stream does change
        pos, vid = _variants(model)
        with torch.no_grad():
            v1 = model.forward_variants(ref1, pos, vid)
        detach_variant_lora(model)
        with torch.no_grad():
            v0 = model.forward_variants(ref0, pos, vid)
        assert (v1.hidden - v0.hidden).abs().max() > 1e-3
    finally:
        detach_variant_lora(model)
        model.freeze_base(False)


def test_zero_B_reproduces_base(model):
    ids = model.tokenizer.encode(SEQ)
    ref = model.forward_reference(ids)
    pos, vid = _variants(model, K=5, N=8, seed=1)
    detach_variant_lora(model)
    with torch.no_grad():
        base = model.forward_variants(ref, pos, vid, output_layers=True)
    attach_variant_lora(model, LoRAConfig(r=16, targets=("q", "k", "v", "o", "gate", "up", "down")))
    try:
        with torch.no_grad():
            lora = model.forward_variants(ref, pos, vid, output_layers=True)
        assert torch.equal(base.hidden, lora.hidden)
        assert torch.equal(base.logits, lora.logits)
        assert all(torch.equal(a, b) for a, b in zip(base.layer_hidden, lora.layer_hidden))
    finally:
        detach_variant_lora(model)
        model.freeze_base(False)


@pytest.mark.parametrize("targets", [("q", "o", "gate", "up", "down"),
                                     ("q", "k", "v", "o", "gate", "up", "down")])
def test_lora_efficient_vs_bruteforce(model, targets):
    ids = model.tokenizer.encode(SEQ)
    ref = model.forward_reference(ids)
    pos, vid = _variants(model, K=6, N=8, seed=2)
    vm = torch.rand(8, 6, generator=torch.Generator().manual_seed(0)) < 0.7
    vm[:, 0] = True
    attach_variant_lora(model, LoRAConfig(r=8, targets=targets))
    _randomize_B(model, std=0.05, seed=3)
    try:
        for var_mask in (None, vm.to(DEV)):
            with torch.no_grad():
                a = model.forward_variants(ref, pos, vid, var_mask=var_mask)
                b = model.forward_variants_bruteforce(ref, pos, vid, var_mask=var_mask)
            dh = (a.hidden - b.hidden).abs().max().item()
            dl = (a.logits - b.logits).abs().max().item()
            print(f"\n  LoRA {targets} var_mask={var_mask is not None}: "
                  f"hidden {dh:.2e} logits {dl:.2e}")
            assert dh < 1e-4 and dl < 1e-3
    finally:
        detach_variant_lora(model)
        model.freeze_base(False)


def test_bf16_base_fp32_adapters_grads():
    m = load_esmc("esmc_300m", device=DEV, dtype=torch.bfloat16)
    attach_variant_lora(m)                       # freezes base
    assert all(p.dtype == torch.float32 for p in lora_parameters(m))
    cast_base(m, torch.bfloat16)                 # must keep adapters fp32
    assert all(p.dtype == torch.float32 for p in lora_parameters(m))
    assert m.embed.weight.dtype == torch.bfloat16
    ref = m.forward_reference(m.tokenizer.encode(SEQ))
    _randomize_B(m, std=0.01)
    pos, vid = _variants(m)
    out = m.forward_variants(ref, pos, vid)
    out.logits.float().logsumexp(-1).sum().backward()
    grads = [p.grad for p in lora_parameters(m)]
    assert all(g is not None and g.dtype == torch.float32 for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0
    assert all(p.grad is None for _, p in m.base_parameters())
    assert all(not t.requires_grad for t in ref.layer_inputs)
    sd = lora_state_dict(m)
    assert sd and all("lora_" in k for k in sd)


# ── shared adapters (both streams) ───────────────────────────────────────────

ALL = ("q", "k", "v", "o", "gate", "up", "down")


def _detach_all(model):
    detach_variant_lora(model)
    detach_shared_lora(model)
    model.freeze_base(False)


def test_shared_zero_B_is_the_base(model):
    ids = model.tokenizer.encode(SEQ)
    _detach_all(model)
    with torch.no_grad():
        full0 = model(ids[None])
    attach_shared_lora(model, LoRAConfig(r=8, targets=ALL))
    try:
        with torch.no_grad():
            full1 = model(ids[None])      # adapter code path, zero delta
        assert (full0.logits - full1.logits).abs().max() < 1e-4
        assert (full0.hidden - full1.hidden).abs().max() < 1e-5
    finally:
        _detach_all(model)


def test_shared_lora_moves_reference_and_identity_stays_exact(model):
    """Shared adapters change the base, so the reference moves — and a variant
    carrying the wild-type residues must still reproduce the (adapted) full
    forward exactly, through the shared K/V on the reference side."""
    ids = model.tokenizer.encode(SEQ)
    _detach_all(model)
    with torch.no_grad():
        ref0 = model.forward_reference(ids)
    attach_shared_lora(model, LoRAConfig(r=8, targets=ALL))
    _randomize_B(model, std=0.05, which="shared_adapters")
    try:
        with torch.no_grad():
            ref = model.forward_reference(ids)
            full = model(ids[None])
            pos, _ = _variants(model, K=6, N=1, seed=4)
            v = model.forward_variants(ref, pos, ids.to(DEV)[pos][None].repeat(3, 1),
                                       output_layers=True)
        assert (ref.final - ref0.final).abs().max() > 1e-2
        assert (v.logits - full.logits[:, pos]).abs().max() < 1e-3
        assert (v.hidden - full.hidden[:, pos]).abs().max() < 1e-4
        for i, x in enumerate(v.layer_hidden):
            tgt = (full.layer_inputs[i + 1] if i + 1 < model.n_layers else full.hidden_prenorm)[:, pos]
            assert ((x - tgt).abs().max() / tgt.abs().max().clamp_min(1)) < 1e-5
    finally:
        _detach_all(model)


@pytest.mark.parametrize("targets", [("q", "o", "gate", "up", "down"), ALL])
def test_shared_plus_variant_efficient_vs_bruteforce(model, targets):
    _detach_all(model)
    attach_shared_lora(model, LoRAConfig(r=8, targets=targets))
    attach_variant_lora(model, LoRAConfig(r=8, targets=targets))
    _randomize_B(model, std=0.05, seed=5, which="shared_adapters")
    _randomize_B(model, std=0.05, seed=6)
    pos, vid = _variants(model, K=6, N=8, seed=7)
    vm = torch.rand(8, 6, generator=torch.Generator().manual_seed(1)) < 0.7
    vm[:, 0] = True
    try:
        with torch.no_grad():
            ref = model.forward_reference(model.tokenizer.encode(SEQ))
            for var_mask in (None, vm.to(DEV)):
                a = model.forward_variants(ref, pos, vid, var_mask=var_mask)
                b = model.forward_variants_bruteforce(ref, pos, vid, var_mask=var_mask)
                dh = (a.hidden - b.hidden).abs().max().item()
                dl = (a.logits - b.logits).abs().max().item()
                print(f"\n  shared+variant {targets} var_mask={var_mask is not None}: "
                      f"hidden {dh:.2e} logits {dl:.2e}")
                assert dh < 1e-4 and dl < 1e-3
    finally:
        _detach_all(model)


def test_shared_lora_grads_through_both_streams(model):
    _detach_all(model)
    attach_shared_lora(model, LoRAConfig(r=4))
    attach_variant_lora(model, LoRAConfig(r=4))
    _randomize_B(model, std=0.01, seed=8, which="shared_adapters")
    _randomize_B(model, std=0.01, seed=9)
    try:
        fp0 = base_fingerprint(model)
        ref = model.forward_reference(model.tokenizer.encode(SEQ), grad=True)
        assert ref.layer_inputs[-1].requires_grad        # the reference now depends on them
        pos, vid = _variants(model, seed=10)
        model.forward_variants(ref, pos, vid).logits.logsumexp(-1).sum().backward()
        for which in ("shared", "variant"):
            g = [p.grad for p in lora_parameters(model, which)]
            assert all(x is not None for x in g) and sum(x.abs().sum().item() for x in g) > 0
        assert all(p.grad is None for _, p in model.base_parameters())
        # shared adapters change the reference, so they are part of its fingerprint
        with torch.no_grad():
            lora_parameters(model, "shared")[1].add_(0.1)
        assert base_fingerprint(model) != fp0
    finally:
        for p in lora_parameters(model):
            p.grad = None
        _detach_all(model)
