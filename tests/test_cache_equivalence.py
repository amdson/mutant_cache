"""Variant cache: efficient (log-sum-exp) vs bruteforce oracle, hooks, padding,
gradients, and the single-mutant approximation gap vs the exact forward.

Run with ``pytest -s`` to see the numbers."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mutant_cache.backbone import load_esmc

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# GB1 (Olson et al. 2014 construct) + ubiquitin: two short real proteins
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


def _random_variants(model, ref, K, N, seed):
    g = torch.Generator().manual_seed(seed)
    L = ref.seq_len
    pos = torch.randperm(L - 2, generator=g)[:K] + 1          # residues only
    aa = torch.tensor(list(model.aa_to_id.values()))
    vid = aa[torch.randint(0, 20, (N, K), generator=g)]
    return pos.to(DEV), vid.to(DEV)


def _maxdiff(a, b):
    return {
        "hidden": (a.hidden - b.hidden).abs().max().item(),
        "logits": (a.logits - b.logits).abs().max().item(),
        "layers_rel": max(((x - y).abs().max() / y.abs().max().clamp_min(1)).item()
                          for x, y in zip(a.layer_hidden, b.layer_hidden)),
    }


def _check(model, ref, pos, vid, var_mask=None, tag=""):
    with torch.no_grad():
        a = model.forward_variants(ref, pos, vid, var_mask=var_mask, output_layers=True)
        b = model.forward_variants_bruteforce(ref, pos, vid, var_mask=var_mask,
                                              output_layers=True)
    d = _maxdiff(a, b)
    print(f"  {tag:<38} K={pos.numel():<3} N={vid.shape[0]:<3} "
          + " ".join(f"{k}={v:.2e}" for k, v in d.items()))
    assert d["hidden"] < 1e-4, d
    assert d["logits"] < 1e-3, d
    assert d["layers_rel"] < 1e-5, d
    return d


@pytest.mark.parametrize("K,N", [(1, 1), (1, 32), (3, 8), (3, 32), (10, 4), (10, 32)])
def test_efficient_vs_bruteforce(model, ref, K, N):
    _reset(model)
    pos, vid = _random_variants(model, ref, K, N, seed=K * 100 + N)
    _check(model, ref, pos, vid, tag="plain")


@pytest.mark.parametrize("K,N", [(3, 8), (10, 32)])
def test_efficient_vs_bruteforce_hooks(model, ref, K, N):
    pos, vid = _random_variants(model, ref, K, N, seed=7 + K + N)
    g = torch.Generator().manual_seed(K * N)
    try:
        # additive ablation: identity variant->variant mask
        _reset(model)
        model.variant_attn_mask = torch.eye(K, dtype=torch.bool, device=DEV)
        _check(model, ref, pos, vid, tag="attn_mask=eye(K)")
        # per-variant random mask (diagonal kept)
        m = torch.rand(N, K, K, generator=g) < 0.5
        m |= torch.eye(K, dtype=torch.bool)
        model.variant_attn_mask = m.to(DEV)
        _check(model, ref, pos, vid, tag="attn_mask=(N,K,K) random")
        # no cross-attention
        _reset(model)
        model.disable_cross_attention = True
        _check(model, ref, pos, vid, tag="disable_cross_attention")
        # extra embed (e.g. assay / position embedding)
        _reset(model)
        torch.manual_seed(0)
        lin = nn.Linear(model.d_model, model.d_model).to(DEV)
        model.variant_extra_embed = lin
        _check(model, ref, pos, vid, tag="variant_extra_embed")
        # everything at once + padding mask
        model.variant_attn_mask = m.to(DEV)
        vm = torch.rand(N, K, generator=g) < 0.7
        vm[:, 0] = True
        _check(model, ref, pos, vid, var_mask=vm.to(DEV), tag="extra+mask+var_mask")
        model.disable_cross_attention = True
        _check(model, ref, pos, vid, var_mask=vm.to(DEV), tag="extra+mask+var_mask+nocross")
    finally:
        _reset(model)


def test_var_mask_padding_is_exact(model, ref):
    """A padded slot (var_mask False) == that position not being a variant."""
    _reset(model)
    pos, vid = _random_variants(model, ref, 6, 1, seed=3)
    real = torch.tensor([True, False, True, True, False, False], device=DEV)
    padded = vid.clone()
    padded[0, ~real] = ref.ids[pos[~real]]
    with torch.no_grad():
        a = model.forward_variants(ref, pos, padded, var_mask=real[None])
        b = model.forward_variants(ref, pos[real], vid[:, real])
    d = (a.hidden[0, real] - b.hidden[0]).abs().max().item()
    dpad = (a.hidden[0, ~real] - ref.final[0, pos[~real]]).abs().max().item()
    print(f"\n  var_mask padding vs real-only: {d:.2e}; padded slots vs ref: {dpad:.2e}")
    assert d < 1e-4 and dpad == 0


def test_reference_tokens_reproduce_reference(model, ref):
    """Substituting the reference's own tokens reproduces the reference states."""
    _reset(model)
    L = ref.seq_len
    pos = torch.arange(1, L - 1, 7, device=DEV)
    with torch.no_grad():
        out = model.forward_variants(ref, pos, ref.ids[pos][None])
    d = (out.hidden[0] - ref.final[0, pos]).abs().max().item()
    print(f"\n  ref tokens as variants (K={pos.numel()}): max|hidden - ref| = {d:.2e}")
    assert d < 1e-4


def test_gradients_flow_through_variant_path(model, ref):
    _reset(model)
    torch.manual_seed(0)
    lin = nn.Linear(model.d_model, model.d_model).to(DEV)
    nn.init.zeros_(lin.weight)
    model.variant_extra_embed = lin
    try:
        pos, vid = _random_variants(model, ref, 3, 4, seed=11)
        out = model.forward_variants(ref, pos, vid)
        out.logits.logsumexp(-1).sum().backward()
        assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0
        assert all(not t.requires_grad for t in ref.layer_inputs)
        assert ref.kv is not None and not ref.kv[0][0].requires_grad
        # base frozen -> no base grads
        assert all(p.grad is None for _, p in model.base_parameters())
    finally:
        _reset(model)


def test_reference_reused_across_calls(model, ref):
    """forward_variants never mutates the reference states (beyond memoizing kv)."""
    _reset(model)
    before = [t.clone() for t in ref.layer_inputs]
    pos, vid = _random_variants(model, ref, 5, 8, seed=5)
    with torch.no_grad():
        a = model.forward_variants(ref, pos, vid)
        b = model.forward_variants(ref, pos, vid)
    assert torch.equal(a.hidden, b.hidden)
    assert all(torch.equal(x, y) for x, y in zip(before, ref.layer_inputs))


def test_bf16_efficient_vs_bruteforce():
    m = load_esmc("esmc_300m", device=DEV, dtype=torch.bfloat16).freeze_base()
    r = m.forward_reference(m.tokenizer.encode(SEQ))
    pos, vid = _random_variants(m, r, 10, 32, seed=1)
    with torch.no_grad():
        a = m.forward_variants(r, pos, vid)
        b = m.forward_variants_bruteforce(r, pos, vid)
    dh = (a.hidden.float() - b.hidden.float()).abs().max().item()
    dl = (a.logits.float() - b.logits.float()).abs().max().item()
    cos = F.cosine_similarity(a.hidden.float(), b.hidden.float(), dim=-1).min().item()
    print(f"\n  [bf16] efficient vs bruteforce K=10 N=32: hidden {dh:.3g} logits {dl:.3g} "
          f"min cos {cos:.5f}")
    assert cos > 0.999


@torch.no_grad()
def test_single_mutant_vs_exact_forward(model, ref):
    """The approximation itself: cache (frozen wild-type context) vs the exact
    full forward of the mutant sequence, at the mutated position. Not expected
    to be zero; reported."""
    _reset(model)
    g = torch.Generator().manual_seed(0)
    L = ref.seq_len
    aas = list(model.aa_to_id)
    cos_all, dlogp_all, llr_gap = [], [], []
    for p in (torch.randperm(L - 2, generator=g)[:12] + 1).tolist():
        wt = model.tokenizer.id_to_aa[int(ref.ids[p])]
        alts = [a for a in aas if a != wt]
        vid = torch.tensor([[model.aa_to_id[a]] for a in alts], device=DEV)  # (19, 1)
        pos = torch.tensor([p], device=DEV)
        cache = model.forward_variants(ref, pos, vid)
        mut_ids = ref.ids[None].repeat(len(alts), 1)
        mut_ids[:, p] = vid[:, 0]
        exact = model(mut_ids)
        h_c, h_e = cache.hidden[:, 0], exact.hidden[:, p]
        lp_c = cache.logits[:, 0].float().log_softmax(-1)
        lp_e = exact.logits[:, p].float().log_softmax(-1)
        cos_all.append(F.cosine_similarity(h_c, h_e, dim=-1))
        dlogp_all.append((lp_c - lp_e).abs().amax(-1))
        # LLR of the mutant residue vs wild type, read at the mutated position
        a_id, w_id = vid[:, 0], model.aa_to_id[wt]
        llr_c = lp_c.gather(1, a_id[:, None])[:, 0] - lp_c[:, w_id]
        llr_e = lp_e.gather(1, a_id[:, None])[:, 0] - lp_e[:, w_id]
        llr_gap.append((llr_c - llr_e).abs())
    cos_all = torch.cat(cos_all)
    dlogp_all = torch.cat(dlogp_all)
    llr_gap = torch.cat(llr_gap)
    print(f"\n  single mutant, cache vs exact ({cos_all.numel()} mutants, L={L}):"
          f"\n    final-hidden cosine  mean {cos_all.mean():.5f}  min {cos_all.min():.5f}"
          f"\n    max|dlogp| per mut   mean {dlogp_all.mean():.4f}  max {dlogp_all.max():.4f}"
          f"\n    |LLR(alt vs wt) gap| mean {llr_gap.mean():.4f}  max {llr_gap.max():.4f}")
    assert cos_all.min() > 0.5   # sanity only: this is the approximation gap


# ---------------------------------------------------------------------------- #
# Cross-attention backends: fused flex vs the fp32 reference implementation     #
# ---------------------------------------------------------------------------- #
needs_cuda = pytest.mark.skipif(DEV != "cuda", reason="flex backend is CUDA-only")


def _both_impls(model, ref, pos, vid, **kw):
    return (model_fv(model, ref, pos, vid, cross_impl="flex", **kw),
            model_fv(model, ref, pos, vid, cross_impl="reference", **kw))


def model_fv(model, ref, pos, vid, **kw):
    from mutant_cache.cache.variant_cache import forward_variants
    return forward_variants(model, ref, pos, vid, **kw)


@needs_cuda
@pytest.mark.parametrize("K,N", [(1, 1), (3, 32), (10, 32)])
def test_flex_vs_reference_cross_attention(model, ref, K, N):
    _reset(model)
    pos, vid = _random_variants(model, ref, K, N, seed=K + 31 * N)
    g = torch.Generator().manual_seed(K)
    vm = torch.rand(N, K, generator=g) < 0.7
    vm[:, 0] = True
    for tag, var_mask in (("plain", None), ("var_mask", vm.to(DEV))):
        with torch.no_grad():
            a, b = _both_impls(model, ref, pos, vid, var_mask=var_mask, output_layers=True)
        d = _maxdiff(a, b)
        print(f"  flex vs reference {tag:<10} K={K:<3} N={N:<3} "
              + " ".join(f"{k}={v:.2e}" for k, v in d.items()))
        assert d["hidden"] < 1e-4 and d["logits"] < 1e-3 and d["layers_rel"] < 1e-5, d


@needs_cuda
def test_flex_vs_reference_gradients(model, ref):
    """Gradients through the cross branch, including its LSE (merge weights)."""
    _reset(model)
    torch.manual_seed(0)
    lin = nn.Linear(model.d_model, model.d_model).to(DEV)
    nn.init.normal_(lin.weight, std=1e-2)
    model.variant_extra_embed = lin
    try:
        pos, vid = _random_variants(model, ref, 5, 16, seed=13)
        grads = []
        for impl in ("flex", "reference"):
            lin.zero_grad()
            out = model_fv(model, ref, pos, vid, cross_impl=impl)
            out.logits.logsumexp(-1).sum().backward()
            grads.append(lin.weight.grad.clone())
        rel = ((grads[0] - grads[1]).abs().max() / grads[1].abs().max()).item()
        print(f"\n  flex vs reference grad(extra_embed) rel max diff {rel:.2e}")
        assert rel < 1e-4
    finally:
        _reset(model)


@needs_cuda
def test_flex_all_positions_variant(model, ref):
    """K == L: no reference keys left; flex skips the cross branch."""
    _reset(model)
    L = ref.seq_len
    pos = torch.arange(L, device=DEV)
    with torch.no_grad():
        a, b = _both_impls(model, ref, pos, ref.ids[None])
    assert (a.hidden - b.hidden).abs().max().item() < 1e-4


@needs_cuda
def test_bf16_flex_vs_reference():
    m = load_esmc("esmc_300m", device=DEV, dtype=torch.bfloat16).freeze_base()
    r = m.forward_reference(m.tokenizer.encode(SEQ))
    pos, vid = _random_variants(m, r, 10, 32, seed=1)
    with torch.no_grad():
        a, b = _both_impls(m, r, pos, vid)
        c = m.forward_variants_bruteforce(r, pos, vid)
    for tag, x in (("flex", a), ("reference", b)):
        cos = F.cosine_similarity(x.hidden.float(), c.hidden.float(), dim=-1).min().item()
        print(f"\n  [bf16] {tag} vs bruteforce: min cos {cos:.5f}")
        assert cos > 0.999
