"""Ragged variant cache (per-variant positions) vs the padded shared-position
path with var_mask (exact by definition), vs a per-variant bruteforce oracle,
flex vs dense-mask attention, adapters, gradients, bf16.

Run with ``pytest -s`` to see the numbers."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mutant_cache.backbone import load_esmc
from mutant_cache.cache import (LoRAConfig, attach_shared_lora, attach_variant_lora,
                                detach_shared_lora, detach_variant_lora)
from mutant_cache.cache.variant_cache import (forward_variants, forward_variants_ragged,
                                              ragged_from_lists, ragged_from_padded)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMPLS = ["flex", "reference"] if DEV == "cuda" else ["reference"]
SEQ = ("MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
       "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")


@pytest.fixture(scope="module")
def model():
    return load_esmc("esmc_300m", device=DEV, dtype=torch.float32).freeze_base()


@pytest.fixture(scope="module")
def ref(model):
    return model.forward_reference(model.tokenizer.encode(SEQ))


def _reset(model):
    model.variant_extra_embed = None
    model.variant_attn_mask = None
    model.disable_cross_attention = False


def _padded_batch(model, ref, K, N, seed, keep=0.5):
    """Shared union positions + a sparse var_mask, i.e. what make_batches builds
    for variants at mixed sites."""
    g = torch.Generator().manual_seed(seed)
    L = ref.seq_len
    pos = torch.randperm(L - 2, generator=g)[:K] + 1
    aa = torch.tensor(list(model.aa_to_id.values()))
    vid = aa[torch.randint(0, 20, (N, K), generator=g)]
    vm = torch.rand(N, K, generator=g) < keep
    vm[torch.arange(N), torch.randint(0, K, (N,), generator=g)] = True
    vid = torch.where(vm, vid, ref.ids.cpu()[pos])
    return pos.to(DEV), vid.to(DEV), vm.to(DEV)


def _vs_padded(model, ref, pos, vid, vm, impl, tag):
    rv = ragged_from_padded(pos, vid, vm)
    with torch.no_grad():
        a = forward_variants_ragged(model, ref, rv, output_layers=True, attn_impl=impl)
        b = forward_variants(model, ref, pos, vid, var_mask=vm, output_layers=True,
                             cross_impl="reference")
    d = {"hidden": (a.hidden - b.hidden[vm]).abs().max().item(),
         "logits": (a.logits - b.logits[vm]).abs().max().item(),
         "layers_rel": max(((x - y[vm]).abs().max() / y[vm].abs().max().clamp_min(1)).item()
                           for x, y in zip(a.layer_hidden, b.layer_hidden))}
    print(f"  ragged[{impl}] vs padded {tag:<22} T={rv.n_tokens:<4} N={rv.n_variants:<3} "
          + " ".join(f"{k}={v:.2e}" for k, v in d.items()))
    assert d["hidden"] < 1e-4 and d["logits"] < 1e-3 and d["layers_rel"] < 1e-5, d


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("K,N", [(1, 1), (4, 16), (12, 32), (40, 64)])
def test_ragged_vs_padded(model, ref, impl, K, N):
    _reset(model)
    pos, vid, vm = _padded_batch(model, ref, K, N, seed=K * 7 + N)
    _vs_padded(model, ref, pos, vid, vm, impl, "plain")


@pytest.mark.parametrize("impl", IMPLS)
def test_ragged_vs_padded_hooks(model, ref, impl):
    K, N = 8, 16
    pos, vid, vm = _padded_batch(model, ref, K, N, seed=3)
    g = torch.Generator().manual_seed(0)
    try:
        _reset(model)
        model.variant_attn_mask = torch.eye(K, dtype=torch.bool, device=DEV)
        _vs_padded(model, ref, pos, vid, vm, impl, "attn_mask=eye")
        m = torch.rand(N, K, K, generator=g) < 0.5
        m |= torch.eye(K, dtype=torch.bool)
        model.variant_attn_mask = m.to(DEV)
        _vs_padded(model, ref, pos, vid, vm, impl, "attn_mask=(N,K,K)")
        _reset(model)
        model.disable_cross_attention = True
        _vs_padded(model, ref, pos, vid, vm, impl, "disable_cross")
        _reset(model)
        torch.manual_seed(0)
        model.variant_extra_embed = nn.Linear(model.d_model, model.d_model).to(DEV)
        model.variant_attn_mask = m.to(DEV)
        _vs_padded(model, ref, pos, vid, vm, impl, "extra+attn_mask")
    finally:
        _reset(model)


@pytest.mark.parametrize("impl", IMPLS)
def test_ragged_vs_bruteforce_independent_positions(model, ref, impl):
    """Variants at unrelated sites, each checked against its own bruteforce call."""
    _reset(model)
    g = torch.Generator().manual_seed(1)
    L = ref.seq_len
    aa = list(model.aa_to_id.values())
    variants = []
    for n in range(12):
        k = int(torch.randint(1, 6, (1,), generator=g))
        ps = (torch.randperm(L - 2, generator=g)[:k] + 1).tolist()
        variants.append([(p, aa[int(torch.randint(0, 20, (1,), generator=g))]) for p in ps])
    rv = ragged_from_lists(variants)
    rv.validate(L)
    with torch.no_grad():
        a = forward_variants_ragged(model, ref, rv, attn_impl=impl)
        worst = 0.0
        for n, v in enumerate(variants):
            pos = torch.tensor([p for p, _ in v], device=DEV)
            vid = torch.tensor([[t for _, t in v]], device=DEV)
            b = model.forward_variants_bruteforce(ref, pos, vid)
            sl = slice(int(rv.offsets[n]), int(rv.offsets[n + 1]))
            worst = max(worst, (a.hidden[sl] - b.hidden[0]).abs().max().item())
    print(f"\n  ragged[{impl}] vs per-variant bruteforce (T={rv.n_tokens}): hidden {worst:.2e}")
    assert worst < 1e-4


@pytest.mark.parametrize("impl", IMPLS)
def test_ragged_vs_padded_lora(model, impl):
    _reset(model)
    targets = ("q", "k", "v", "o", "gate", "up", "down")
    attach_variant_lora(model, LoRAConfig(r=4, targets=targets))
    attach_shared_lora(model, LoRAConfig(r=4, targets=targets))
    try:
        g = torch.Generator(device=DEV).manual_seed(0)
        with torch.no_grad():
            for mod in (model.variant_adapters, model.shared_adapters):
                for n, p in mod.named_parameters():
                    if "lora_B" in n:
                        p.copy_(torch.randn(p.shape, generator=g, device=DEV) * 0.02)
        # shared adapters change the reference stream: it must be rebuilt
        ref = model.forward_reference(model.tokenizer.encode(SEQ))
        pos, vid, vm = _padded_batch(model, ref, 8, 16, seed=5)
        _vs_padded(model, ref, pos, vid, vm, impl, "variant+shared LoRA")
    finally:
        detach_variant_lora(model)
        detach_shared_lora(model)


@pytest.mark.skipif(DEV != "cuda", reason="flex is CUDA-only")
def test_ragged_flex_vs_dense_gradients(model, ref):
    _reset(model)
    torch.manual_seed(0)
    lin = nn.Linear(model.d_model, model.d_model).to(DEV)
    nn.init.normal_(lin.weight, std=1e-2)
    model.variant_extra_embed = lin
    try:
        pos, vid, vm = _padded_batch(model, ref, 10, 16, seed=9)
        rv = ragged_from_padded(pos, vid, vm)
        grads = []
        for impl in ("flex", "reference"):
            lin.zero_grad()
            out = forward_variants_ragged(model, ref, rv, attn_impl=impl)
            out.logits.logsumexp(-1).sum().backward()
            grads.append(lin.weight.grad.clone())
        rel = ((grads[0] - grads[1]).abs().max() / grads[1].abs().max()).item()
        print(f"\n  ragged flex vs dense grad(extra_embed) rel max diff {rel:.2e}")
        assert rel < 1e-4
    finally:
        _reset(model)


def test_ragged_empty_and_wildtype(model, ref):
    """Variants with no rows are allowed (offsets repeat); ref tokens reproduce ref."""
    _reset(model)
    L = ref.seq_len
    ps = list(range(1, L - 1, 9))
    rv = ragged_from_lists([[], [(p, int(ref.ids[p])) for p in ps], []])
    with torch.no_grad():
        out = forward_variants_ragged(model, ref, rv)
    d = (out.hidden - ref.final[0, torch.tensor(ps, device=DEV)]).abs().max().item()
    assert out.hidden.shape[0] == len(ps) and d < 1e-4


@pytest.mark.skipif(DEV != "cuda", reason="flex is CUDA-only")
def test_ragged_bf16_vs_padded():
    m = load_esmc("esmc_300m", device=DEV, dtype=torch.bfloat16).freeze_base()
    r = m.forward_reference(m.tokenizer.encode(SEQ))
    pos, vid, vm = _padded_batch(m, r, 12, 32, seed=1)
    with torch.no_grad():
        a = forward_variants_ragged(m, r, ragged_from_padded(pos, vid, vm))
        b = m.forward_variants_bruteforce(r, pos, vid, var_mask=vm)
    cos = F.cosine_similarity(a.hidden.float(), b.hidden[vm].float(), dim=-1).min().item()
    print(f"\n  [bf16] ragged flex vs bruteforce: min cos {cos:.5f}")
    assert cos > 0.999
