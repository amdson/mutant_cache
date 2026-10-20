"""Local ESMC (mutant_cache.backbone) vs the official `esm` package.

Two official references, both from the installed esm 3.4.1:
  * legacy  — ``esm/models/esmc.py::ESMC`` (the class the published .pth was
    saved from; shadowed by the ``esm.models.esmc`` package, so loaded by path),
    SDPA attention (no flash-attn installed).
  * current — ``esm.models.esmc.EsmcForMaskedLM`` (sdpa), checkpoint keys
    translated to its layout.

Run with ``pytest -s`` to see the numbers.
"""
import importlib.util
import logging
import os
import re

import pytest
import torch

from mutant_cache.backbone import load_esmc, esmc_weights_path

# The official esm warns at import that fused kernels (Transformer Engine,
# xformers, flash-attn) are missing. Keep it that way: the pure-PyTorch path is
# the one our port matches bit-for-bit; fused kernels would make it drift.
logging.getLogger("esm.models.esmc.kernels").setLevel(logging.ERROR)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NAME = "esmc_300m"
SEQS = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDW",
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
]


def _legacy_cls():
    import esm
    path = os.path.join(os.path.dirname(esm.__file__), "models", "esmc.py")
    spec = importlib.util.spec_from_file_location("_esm_legacy_esmc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ESMC


def _sd():
    return torch.load(esmc_weights_path(NAME), map_location="cpu", weights_only=True)


def official_legacy(dtype):
    from esm.tokenization import EsmSequenceTokenizer
    m = _legacy_cls()(960, 15, 30, EsmSequenceTokenizer(), use_flash_attn=False)
    m.load_state_dict(_sd(), strict=True)
    return m.to(DEV, dtype).eval()


def _legacy_to_native(sd):
    out = {}
    for k, v in sd.items():
        k2 = k
        k2 = re.sub(r"attn\.layernorm_qkv\.0\.weight$", "attn.layernorm_qkv.layer_norm_weight", k2)
        k2 = re.sub(r"attn\.layernorm_qkv\.0\.bias$", "attn.layernorm_qkv.layer_norm_bias", k2)
        k2 = re.sub(r"attn\.layernorm_qkv\.1\.weight$", "attn.layernorm_qkv.weight", k2)
        k2 = re.sub(r"ffn\.0\.weight$", "ffn.layer_norm_weight", k2)
        k2 = re.sub(r"ffn\.0\.bias$", "ffn.layer_norm_bias", k2)
        k2 = re.sub(r"ffn\.1\.weight$", "ffn.fc1_weight", k2)
        k2 = re.sub(r"ffn\.3\.weight$", "ffn.fc2_weight", k2)
        if k2.startswith("sequence_head."):
            k2 = "lm_head." + k2[len("sequence_head."):]
        else:
            k2 = "esmc." + k2
        out[k2] = v
    return out


def official_current(dtype):
    from esm.models.esmc import EsmcConfig, EsmcForMaskedLM
    with torch.device("cpu"):
        m = EsmcForMaskedLM(EsmcConfig(hidden_size=960, num_attention_heads=15,
                                       num_hidden_layers=30, attn_implementation="sdpa"))
    m.load_state_dict(_legacy_to_native(_sd()), strict=True)
    return m.to(DEV, dtype).eval()


@pytest.fixture(scope="module")
def batch():
    from mutant_cache.backbone import ESMCTokenizer
    return ESMCTokenizer().encode_batch(SEQS).to(DEV)


def test_tokenizer_matches_official():
    from esm.models.esmc import EsmcTokenizer
    from esm.tokenization import EsmSequenceTokenizer
    from mutant_cache.backbone import ESMCTokenizer, CANONICAL_AA
    mine = ESMCTokenizer()
    seq = SEQS[1] + "XBUZO"
    for off in (EsmcTokenizer(), EsmSequenceTokenizer()):
        assert mine.encode(seq).tolist() == off(seq)["input_ids"]
        assert mine.pad_token_id == off.pad_token_id
        assert mine.mask_token_id == off.mask_token_id
        assert mine.eos_token_id == off.eos_token_id
    assert len(mine.aa_to_id) == 20 and set(mine.aa_to_id) == set(CANONICAL_AA)
    assert mine.decode(mine.encode(SEQS[1])) == SEQS[1]


def _diffs(mine, legacy_out, current_out, mask):
    """max abs diff over real (non-pad) positions."""
    m = mask[..., None]
    md = lambda a, b: ((a.float() - b.float()).abs() * m).max().item()  # noqa: E731
    r = {}
    r["legacy.logits"] = md(mine.logits, legacy_out.sequence_logits)
    r["legacy.hidden"] = md(mine.hidden, legacy_out.embeddings)
    # legacy hidden_states[i] = output of block i (pre final norm)
    hs = legacy_out.hidden_states
    r["legacy.layers"] = max(md(mine.layer_inputs[i + 1], hs[i]) for i in range(len(hs) - 1))
    r["legacy.prenorm"] = md(mine.hidden_prenorm, hs[-1])
    r["current.logits"] = md(mine.logits, current_out.logits)
    r["current.hidden"] = md(mine.hidden, current_out.last_hidden_state)
    # current hidden_states = (block inputs ..., final normed)
    chs = current_out.hidden_states
    r["current.layer_inputs"] = max(md(mine.layer_inputs[i], chs[i]) for i in range(len(chs) - 1))
    return r


@torch.no_grad()
def test_fp32_equivalence(batch):
    mine = load_esmc(NAME, device=DEV, dtype=torch.float32)
    mask = batch != 1
    out = mine(batch)
    leg = official_legacy(torch.float32)(sequence_tokens=batch)
    cur = official_current(torch.float32)(input_ids=batch, output_hidden_states=True)
    r = _diffs(out, leg, cur, mask)
    print("\n[fp32] max abs diff vs official:", {k: f"{v:.3g}" for k, v in r.items()})
    print(f"[fp32] residual-stream scale: max|x| = {out.hidden_prenorm.abs().max().item():.3g}")
    assert r["legacy.logits"] < 1e-3 and r["current.logits"] < 1e-3
    assert r["legacy.hidden"] < 1e-4 and r["current.hidden"] < 1e-4
    # unnormalized residual stream: relative to its magnitude
    scale = out.hidden_prenorm.abs().max().item()
    assert r["legacy.layers"] < 1e-5 * max(scale, 1.0) * 10
    assert r["current.layer_inputs"] < 1e-5 * max(scale, 1.0) * 10


@torch.no_grad()
def test_bf16_deviation(batch):
    """Report bf16 deviation (mine-bf16 vs official-bf16, and both vs fp32)."""
    mask = batch != 1
    ref32 = official_current(torch.float32)(input_ids=batch, output_hidden_states=True)
    mine = load_esmc(NAME, device=DEV, dtype=torch.bfloat16)
    out = mine(batch)
    leg = official_legacy(torch.bfloat16)(sequence_tokens=batch)
    cur = official_current(torch.bfloat16)(input_ids=batch, output_hidden_states=True)
    r = _diffs(out, leg, cur, mask)
    m = mask[..., None]
    md = lambda a, b: ((a.float() - b.float()).abs() * m).max().item()  # noqa: E731
    r["mine_bf16_vs_fp32.logits"] = md(out.logits, ref32.logits)
    r["official_bf16_vs_fp32.logits"] = md(cur.logits, ref32.logits)
    r["mine_bf16_vs_fp32.hidden"] = md(out.hidden, ref32.last_hidden_state)
    r["official_bf16_vs_fp32.hidden"] = md(cur.last_hidden_state, ref32.last_hidden_state)
    print("\n[bf16] max abs diff:", {k: f"{v:.3g}" for k, v in r.items()})
    am = (out.logits.argmax(-1) == cur.logits.argmax(-1))[mask].float().mean().item()
    print(f"[bf16] argmax agreement mine vs official-current: {am:.4f}")
    # mine in bf16 is no further from fp32 than the official bf16 model is (+slack)
    assert r["mine_bf16_vs_fp32.logits"] < 2 * r["official_bf16_vs_fp32.logits"] + 0.1
    assert am > 0.97
