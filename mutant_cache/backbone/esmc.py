"""
mutant_cache/backbone/esmc.py
-----------------------------
Local, hackable re-implementation of ESM-C (EvolutionaryScale), in the spirit of
~/svar/CARBON_modules/carbon_layers.py: the same math as the official ``esm``
package, as a flat pure-PyTorch stack that is easy to fork for the variant cache.

Architecture (read off ``esm/layers/{blocks,attention,rotary}.py`` and
``esm/models/esmc/layers.py``, esm 3.4.1):

  embed: nn.Embedding(64, d)                       (no positional embedding)
  n_layers x pre-norm block, residual scaled by 1/sqrt(n_layers/36):
      x = x + attn(x) / s
      x = x + ffn(x)  / s
    attn: LayerNorm(d, bias) -> Linear(d, 3d, no bias) -> chunk q,k,v
          q_ln / k_ln: LayerNorm(d, NO bias) over the full d (not per head)
          RoPE (GPT-NeoX half-split, base 1e4, head_dim) at absolute token index
          softmax(q k^T / sqrt(dh)) v, bidirectional, pad keys masked
          out_proj: Linear(d, d, no bias)
    ffn:  LayerNorm(d, bias) -> Linear(d, 2*hid, no bias) -> silu(x1) * x2
          -> Linear(hid, d, no bias);  hid = ceil256(8/3 * d)
  final norm: LayerNorm(d, NO bias)
  sequence_head: Linear(d,d) -> GELU -> LayerNorm(d) -> Linear(d, 64)

  esmc_300m: d=960,  15 heads (dh 64), 30 layers, hid 2560
  esmc_600m: d=1152, 18 heads (dh 64), 36 layers, hid 3072

Parity
------
Parameter names match the published ``esmc_*_2024_12_v0.pth`` state dicts (the
legacy ``esm.models.esmc.ESMC`` layout), so they load with ``strict=True``.
Outputs match the official model to fp32 numerics; see
``tests/test_backbone_equivalence.py``.

RoPE detail worth knowing: ``inv_freq`` is kept in fp32 and never cast with the
model. The legacy official class registers it as a float buffer, so
``model.to(torch.bfloat16)`` rounds the *frequencies* to bf16; the current
``esm.models.esmc`` recomputes them in fp32 after every cast. We follow the
current one.

Per-layer residual-stream inputs (``out.layer_inputs[i]`` = what enters block i)
are exactly the cache the variant path cross-attends to: K/V of block i are
projections of LayerNorm(layer_inputs[i]).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import ESMCTokenizer

ESMC_CONFIGS = {
    "esmc_300m": dict(d_model=960, n_heads=15, n_layers=30,
                      hf_repo="EvolutionaryScale/esmc-300m-2024-12",
                      weights="data/weights/esmc_300m_2024_12_v0.pth"),
    "esmc_600m": dict(d_model=1152, n_heads=18, n_layers=36,
                      hf_repo="EvolutionaryScale/esmc-600m-2024-12",
                      weights="data/weights/esmc_600m_2024_12_v0.pth"),
}
VOCAB_SIZE = 64


def swiglu_hidden_dim(d_model: int, expansion_ratio: float = 8 / 3) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


# ---------------------------------------------------------------------------- #
# RoPE                                                                          #
# ---------------------------------------------------------------------------- #
def rope_inv_freq(head_dim: int, device=None, base: float = 10000.0) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device,
                                        dtype=torch.float32) / head_dim))


def rope_cos_sin(positions: torch.Tensor, head_dim: int, dtype: torch.dtype,
                 inv_freq: Optional[torch.Tensor] = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for arbitrary absolute ``positions`` (P,), shaped
    ``(P, 1, head_dim)`` so they broadcast over ``(..., P, heads, head_dim)``.

    Matches the official tables: positions and frequencies in fp32, the outer
    product in fp32, then cast to the activation dtype; halves repeated
    ``[c, c]`` (non-interleaved / GPT-NeoX layout)."""
    if inv_freq is None:
        inv_freq = rope_inv_freq(head_dim, device=positions.device)
    freqs = torch.outer(positions.to(torch.float32), inv_freq.to(positions.device))
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(-2)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(-2)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: ``(..., P, heads, head_dim)``; cos/sin: ``(P, 1, head_dim)``."""
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------- #
# Modules (attribute names == published checkpoint keys)                        #
# ---------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class ESMCAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 3 * d_model, bias=False))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_ln = nn.LayerNorm(d_model, bias=False)
        self.k_ln = nn.LayerNorm(d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, L, d). attn_mask: bool, broadcastable to (B, h, L, L), True =
        attend; None = full bidirectional attention."""
        B, L, _ = x.shape
        q, k, v = self.layernorm_qkv(x).chunk(3, dim=-1)
        q = self.q_ln(q).to(q.dtype)
        k = self.k_ln(k).to(q.dtype)
        q = apply_rope(q.view(B, L, self.n_heads, self.d_head), cos, sin)
        k = apply_rope(k.view(B, L, self.n_heads, self.d_head), cos, sin)
        v = v.view(B, L, self.n_heads, self.d_head)
        ctx = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask)
        return self.out_proj(ctx.transpose(1, 2).reshape(B, L, self.d_model))


class ESMCBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        hid = swiglu_hidden_dim(d_model)
        self.attn = ESMCAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * hid, bias=False),
            SwiGLU(),
            nn.Linear(hid, d_model, bias=False),
        )
        self.scaling_factor = math.sqrt(n_layers / 36)

    def forward(self, x, cos, sin, attn_mask=None):
        x = x + self.attn(x, cos, sin, attn_mask) / self.scaling_factor
        x = x + self.ffn(x) / self.scaling_factor
        return x


class ESMCTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ESMCBlock(d_model, n_heads, n_layers) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model, bias=False)


@dataclass
class ESMCOutput:
    logits: torch.Tensor                 # (B, L, V)
    hidden: torch.Tensor                 # (B, L, d) post final norm (head input)
    layer_inputs: list[torch.Tensor]     # n_layers x (B, L, d): residual entering block i
    hidden_prenorm: torch.Tensor         # (B, L, d) last block output, pre final norm


class ESMC(nn.Module):
    """ESM-C masked LM.

    Variant-cache API (implemented in ``mutant_cache.cache.variant_cache``):
    ``forward_reference``, ``forward_variants``, ``forward_variants_bruteforce``.

    Hooks read only by the variant path (never by ``forward`` /
    ``forward_reference``):
      * ``variant_extra_embed``: callable ``(N, K, d) -> (N, K, d)``, called on
        the variant token embeddings and *added* to them before block 0.
      * ``variant_attn_mask``: bool ``(K, K)`` or ``(N, K, K)``; False blocks
        variant query i -> variant key j.
      * ``disable_cross_attention``: variant queries see only variant keys.
      * ``variant_adapters``: set by ``mutant_cache.cache.lora.attach_variant_lora``.

    ``shared_adapters`` (``attach_shared_lora``) are not a variant-path hook:
    they are a low-rank change to the base weights, so ``forward`` applies them
    too (and with them, ``forward_reference``).
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 name: str = "esmc"):
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = d_model // n_heads
        self.vocab_size = VOCAB_SIZE
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.transformer = ESMCTransformer(d_model, n_heads, n_layers)
        self.sequence_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, VOCAB_SIZE))
        self.tokenizer = ESMCTokenizer()
        self.aa_to_id: dict[str, int] = dict(self.tokenizer.aa_to_id)

        # Variant-path hooks (see class docstring).
        self.variant_extra_embed: Optional[Callable] = None
        self.variant_attn_mask: Optional[torch.Tensor] = None
        self.disable_cross_attention: bool = False
        self.variant_adapters: Optional[nn.ModuleList] = None
        self.shared_adapters: Optional[nn.ModuleList] = None

        self._inv_freq_cache: dict = {}

    # -- small helpers ------------------------------------------------------- #
    @property
    def blocks(self) -> nn.ModuleList:
        return self.transformer.blocks

    @property
    def device(self) -> torch.device:
        return self.embed.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed.weight.dtype

    def inv_freq(self, device) -> torch.Tensor:
        key = str(device)
        if key not in self._inv_freq_cache:
            self._inv_freq_cache[key] = rope_inv_freq(self.head_dim, device=device)
        return self._inv_freq_cache[key]

    def rope(self, positions: torch.Tensor, dtype: Optional[torch.dtype] = None):
        dtype = dtype or self.dtype
        return rope_cos_sin(positions, self.head_dim, dtype,
                            self.inv_freq(positions.device))

    def base_parameters(self):
        """Backbone parameters (everything except adapters / hooks)."""
        for n, p in self.named_parameters():
            if not n.startswith(("variant_adapters.", "shared_adapters.",
                                 "variant_extra_embed.")):
                yield n, p

    def freeze_base(self, freeze: bool = True) -> "ESMC":
        for _, p in self.base_parameters():
            p.requires_grad_(not freeze)
        return self

    # -- full forward (the exact oracle) ------------------------------------ #
    def forward(self, ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> ESMCOutput:
        if ids.dim() == 1:
            ids = ids[None]
        ids = ids.to(self.device)
        B, L = ids.shape
        if attention_mask is None:
            attention_mask = ids != self.tokenizer.pad_token_id
        attention_mask = attention_mask.to(self.device).bool()
        mask = None if bool(attention_mask.all()) else attention_mask[:, None, None, :]

        x = self.embed(ids)
        cos, sin = self.rope(torch.arange(L, device=self.device), x.dtype)
        layer_inputs = []
        if self.shared_adapters is None:
            for blk in self.blocks:            # stock path: parity with official esm
                layer_inputs.append(x)
                x = blk(x, cos, sin, mask)
        else:
            from mutant_cache.cache.variant_cache import _full_block
            for blk, sh in zip(self.blocks, self.shared_adapters):
                layer_inputs.append(x)
                x = _full_block(self, blk, x, cos, sin, mask, (sh, None), None)
        hidden = self.transformer.norm(x)
        logits = self.sequence_head(hidden)
        return ESMCOutput(logits=logits, hidden=hidden, layer_inputs=layer_inputs,
                          hidden_prenorm=x)

    # -- variant cache (thin wrappers; logic lives in mutant_cache.cache) ---- #
    def forward_reference(self, ids: torch.Tensor, grad: bool = False,
                          cache_kv: bool = True):
        from mutant_cache.cache.variant_cache import forward_reference
        return forward_reference(self, ids, grad=grad, cache_kv=cache_kv)

    def forward_variants(self, ref, positions, variant_ids, var_mask=None,
                         output_logits: bool = True, output_layers: bool = False):
        from mutant_cache.cache.variant_cache import forward_variants
        return forward_variants(self, ref, positions, variant_ids, var_mask=var_mask,
                                output_logits=output_logits,
                                output_layers=output_layers)

    def forward_variants_bruteforce(self, ref, positions, variant_ids, var_mask=None,
                                    output_logits: bool = True,
                                    output_layers: bool = False):
        from mutant_cache.cache.variant_cache import forward_variants_bruteforce
        return forward_variants_bruteforce(self, ref, positions, variant_ids,
                                           var_mask=var_mask,
                                           output_logits=output_logits,
                                           output_layers=output_layers)

    def forward_variants_ragged(self, ref, rv, output_logits: bool = True,
                                output_layers: bool = False):
        from mutant_cache.cache.variant_cache import forward_variants_ragged
        return forward_variants_ragged(self, ref, rv, output_logits=output_logits,
                                       output_layers=output_layers)

    def forward_variants_exact(self, ref, positions, variant_ids, var_mask=None,
                               output_logits: bool = True, output_layers: bool = False):
        from mutant_cache.cache.variant_cache import forward_variants_exact
        return forward_variants_exact(self, ref, positions, variant_ids,
                                      var_mask=var_mask, output_logits=output_logits,
                                      output_layers=output_layers)


# ---------------------------------------------------------------------------- #
# Loading                                                                       #
# ---------------------------------------------------------------------------- #
def esmc_weights_path(name: str) -> str:
    """Local path of the official ``.pth`` (HF cache; offline-first)."""
    cfg = ESMC_CONFIGS[name]
    from huggingface_hub import snapshot_download
    try:
        root = snapshot_download(cfg["hf_repo"], local_files_only=True)
    except Exception:
        root = snapshot_download(cfg["hf_repo"])
    return os.path.join(root, cfg["weights"])


def load_esmc(name: str = "esmc_300m", device="cuda",
              dtype: torch.dtype = torch.float32,
              weights_path: Optional[str] = None) -> ESMC:
    """Build the local ESMC and load the official checkpoint (strict).

    The whole model (LayerNorm weights included) is cast to ``dtype``, as the
    official ``ESMC.from_pretrained`` does for bf16. Returned in eval mode;
    parameters keep ``requires_grad=True`` (call ``model.freeze_base()``)."""
    if name not in ESMC_CONFIGS:
        raise ValueError(f"unknown ESMC model {name!r}; pick from {list(ESMC_CONFIGS)}")
    cfg = ESMC_CONFIGS[name]
    model = ESMC(cfg["d_model"], cfg["n_heads"], cfg["n_layers"], name=name)
    sd = torch.load(weights_path or esmc_weights_path(name), map_location="cpu",
                    weights_only=True)
    model.load_state_dict(sd, strict=True)
    return model.to(device=device, dtype=dtype).eval()
