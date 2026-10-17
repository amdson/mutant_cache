"""Variant-only LoRA: reference stream untouched, zero-B == base, efficient ==
bruteforce with adapters, bf16 base + fp32 adapters."""
import pytest
import torch

from mutant_cache.backbone import load_esmc
from mutant_cache.cache import (LoRAConfig, attach_variant_lora, cast_base,
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


def _randomize_B(model, std=0.02, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    with torch.no_grad():
        for n, p in model.variant_adapters.named_parameters():
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
