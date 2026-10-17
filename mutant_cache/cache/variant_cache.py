"""
mutant_cache/cache/variant_cache.py
-----------------------------------
Bidirectional sparse variant KV-cache for ESM-C. Port of the DNA-LM versions in
~/svar (`DNABERT2_modules/variant_cache_layers.py`: bidirectional + ALiBi;
`CARBON_modules/variant_cache_layers.py`: causal + RoPE + batch-1 folding).

The idea
--------
Run one *reference* sequence once and cache the residual stream entering every
block (`forward_reference`). For N variants that each substitute tokens at the
same K positions, recompute **only those K positions**, layer by layer, with
the rest of the sequence frozen to the reference. A variant query at token
position p attends to

    {reference keys at non-variant positions}  ∪  {variant keys (all K slots)}

with no causal mask (ESM-C is bidirectional). We never concatenate the two key
sets (that would expand the reference to batch N); instead two attentions are
run and merged with the log-sum-exp trick, which is exactly one softmax over
the union. The reference K/V stay at batch 1 and the N scenarios are folded
into the cross branch's *query* axis so autograd never saves an N-expanded copy
of them (see `_cross_attention_with_lse`).

RoPE is applied at absolute token positions: variant Q/K rotated by
``positions``, reference K by ``0..L-1``. QK-LayerNorm is applied to both
streams (it is per-token, so the reference keys' norm is cached with them).

The approximation: reference positions never attend to the variants, so every
non-variant token's state is the wild-type one. Everything else is exact.

Backends
--------
* `forward_variants`            — efficient: O(N·K·L) attention per layer.
* `forward_variants_bruteforce` — oracle: ordinary full-sequence blocks on the
  (N, L) mutant sequences, with non-variant positions clamped back to the
  frozen reference after every block and an attention mask implementing the
  same key sets / hooks. Matches the efficient path to fp32 numerics.

Padding (``var_mask``)
----------------------
All N variants share the K positions. A variant with fewer real mutations pads
its unused slots with the reference token and passes ``var_mask[n, j] = False``.
A masked slot is treated *exactly* as a non-variant position: its residual is
clamped to the reference at every layer (so its key/value are the reference's),
adapters are not applied to it, it is seen as a reference key (subject to
``disable_cross_attention``, not to ``variant_attn_mask``), and its outputs are
the reference's. Without ``var_mask`` a pad slot is still exact in the sense
that it carries the reference token, but it is *recomputed* rather than frozen.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from mutant_cache.backbone.esmc import ESMC, apply_rope

_NEG = torch.finfo(torch.float32).min


# ---------------------------------------------------------------------------- #
# Data containers                                                               #
# ---------------------------------------------------------------------------- #
@dataclass
class ReferenceStates:
    """Frozen per-layer states of one reference sequence (batch 1).

    Valid only for the base weights they were computed with. With variant-only
    LoRA (`mutant_cache.cache.lora`) the reference stream never sees adapters,
    so these stay valid while adapters train; if base weights change they are
    stale (see `ReferenceCache`).
    """
    ids: torch.Tensor                    # (L,)
    layer_inputs: list[torch.Tensor]     # n_layers x (1, L, d): residual entering block i
    final: torch.Tensor                  # (1, L, d) post final norm
    final_prenorm: torch.Tensor          # (1, L, d) last block output
    # Per layer (k, v), each (1, h, L, dh): k already q/k-normed and rotated.
    # Pure function of layer_inputs + base weights; memoized lazily.
    kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None

    @property
    def seq_len(self) -> int:
        return self.ids.shape[0]

    def tensors(self) -> list[torch.Tensor]:
        out = [self.ids, *self.layer_inputs, self.final, self.final_prenorm]
        if self.kv is not None:
            out += [t for kv in self.kv for t in kv]
        return out

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())

    def to(self, device=None, dtype=None) -> "ReferenceStates":
        f = lambda t: t.to(device=device, dtype=dtype)  # noqa: E731
        return ReferenceStates(
            ids=self.ids.to(device=device),
            layer_inputs=[f(t) for t in self.layer_inputs],
            final=f(self.final), final_prenorm=f(self.final_prenorm),
            kv=None if self.kv is None else [(f(k), f(v)) for k, v in self.kv])

    def detach(self) -> "ReferenceStates":
        d = lambda t: t.detach()  # noqa: E731
        return ReferenceStates(
            ids=self.ids, layer_inputs=[d(t) for t in self.layer_inputs],
            final=d(self.final), final_prenorm=d(self.final_prenorm),
            kv=None if self.kv is None else [(d(k), d(v)) for k, v in self.kv])


@dataclass
class VariantOutput:
    hidden: torch.Tensor                       # (N, K, d) post final norm
    logits: Optional[torch.Tensor] = None      # (N, K, V)
    layer_hidden: Optional[list[torch.Tensor]] = None  # n_layers x (N, K, d) block outputs (pre-norm)
    hidden_prenorm: Optional[torch.Tensor] = None      # (N, K, d)


# ---------------------------------------------------------------------------- #
# Reference pass                                                                #
# ---------------------------------------------------------------------------- #
def _reference_kv_layer(model: ESMC, layer: int, x: torch.Tensor, cos, sin):
    """Rotated, normed reference K and V for block ``layer`` from its input
    ``x`` (1, L, d). Base weights only — never adapters."""
    attn = model.blocks[layer].attn
    d, h, dh = model.d_model, model.n_heads, model.head_dim
    L = x.shape[1]
    xn = attn.layernorm_qkv[0](x)
    w = attn.layernorm_qkv[1].weight
    k = F.linear(xn, w[d:2 * d])
    v = F.linear(xn, w[2 * d:])
    k = attn.k_ln(k).to(xn.dtype)
    k = apply_rope(k.view(1, L, h, dh), cos, sin).transpose(1, 2)
    v = v.view(1, L, h, dh).transpose(1, 2)
    return k, v


def reference_kv(model: ESMC, ref: ReferenceStates):
    """Per-layer reference (k, v); memoized on ``ref`` when it is constant
    (no autograd graph), recomputed otherwise so gradients can flow."""
    if ref.kv is not None:
        return ref.kv
    L = ref.seq_len
    cos, sin = model.rope(torch.arange(L, device=ref.layer_inputs[0].device),
                          ref.layer_inputs[0].dtype)
    kv = [_reference_kv_layer(model, i, x, cos, sin)
          for i, x in enumerate(ref.layer_inputs)]
    if not any(t.requires_grad for t in kv[0]):
        ref.kv = kv
    return kv


def forward_reference(model: ESMC, ids: torch.Tensor, grad: bool = False,
                      cache_kv: bool = True) -> ReferenceStates:
    """One batch-1 pass over the reference, caching every block's input.

    ``grad=False`` (default) runs under ``no_grad``: with a frozen base the
    reference is a constant and can be reused across steps and variants.
    ``grad=True`` keeps the graph (e.g. to fine-tune the base through the
    reference stream; then the states must not be cached across steps).
    Uses only the stock blocks, so adapters can never reach it."""
    ids = ids.reshape(-1).to(model.device)
    with (nullcontext() if grad else torch.no_grad()):
        out = model(ids[None])
        ref = ReferenceStates(ids=ids, layer_inputs=out.layer_inputs,
                              final=out.hidden, final_prenorm=out.hidden_prenorm)
        if cache_kv:
            reference_kv(model, ref)
    return ref


# ---------------------------------------------------------------------------- #
# Shared helpers                                                                #
# ---------------------------------------------------------------------------- #
def _prep(model: ESMC, ref: ReferenceStates, positions, variant_ids, var_mask):
    dev = model.device
    positions = torch.as_tensor(positions, device=dev).long().reshape(-1)
    variant_ids = torch.as_tensor(variant_ids, device=dev).long()
    if variant_ids.dim() == 1:
        variant_ids = variant_ids[None]
    K = positions.shape[0]
    if variant_ids.shape[1] != K:
        raise ValueError(f"variant_ids {tuple(variant_ids.shape)} vs {K} positions")
    if torch.unique(positions).numel() != K:
        raise ValueError("positions must be unique")
    L = ref.seq_len
    if K and (int(positions.min()) < 0 or int(positions.max()) >= L):
        raise ValueError(f"positions out of range [0, {L})")
    if var_mask is not None:
        var_mask = torch.as_tensor(var_mask, device=dev).bool().reshape(variant_ids.shape)
    return positions, variant_ids, var_mask


def _slot_key_mask(model: ESMC, var_mask, N: int, K: int, device):
    """(N|1, K, K) bool for variant query slot i -> variant key slot j, or
    None for all-True. Encodes `variant_attn_mask` and padded slots."""
    sm = model.variant_attn_mask
    if sm is not None:
        sm = torch.as_tensor(sm, device=device).bool()
        if sm.dim() == 2:
            sm = sm[None]
        if sm.shape[-2:] != (K, K) or sm.shape[0] not in (1, N):
            raise ValueError(f"variant_attn_mask {tuple(sm.shape)} vs N={N}, K={K}")
    if var_mask is not None:
        # A padded slot is a reference key: visible iff cross-attention is on.
        base = sm if sm is not None else torch.ones(1, K, K, dtype=torch.bool, device=device)
        ref_ok = torch.tensor(not model.disable_cross_attention, device=device)
        sm = torch.where(var_mask[:, None, :], base, ref_ok)
    return sm


def _embed_variants(model: ESMC, ref: ReferenceStates, positions, variant_ids, var_mask):
    x = model.embed(variant_ids)
    if model.variant_extra_embed is not None:
        x = x + model.variant_extra_embed(x)
    return _clamp(x, ref.layer_inputs[0], positions, var_mask)


def _clamp(x, ref_layer, positions, var_mask):
    """Padded slots take the reference residual at their position."""
    if var_mask is None:
        return x
    return torch.where(var_mask[..., None], x, ref_layer[0, positions].to(x.dtype))


def _adapter(model: ESMC, i: int):
    ad = model.variant_adapters
    return None if ad is None else ad[i]


def _delta(ad, name, x, rowmask):
    if ad is None:
        return None
    dlt = ad.delta(name, x)
    if dlt is None or rowmask is None:
        return dlt
    return dlt * rowmask.to(dlt.dtype)


def _add(base, dlt):
    return base if dlt is None else base + dlt


def _ffn(blk, x, ad, rowmask):
    """blk.ffn unrolled so gate/up/down can each take an adapter delta."""
    ln, fc1, _, fc2 = blk.ffn
    xn = ln(x)
    hid = fc2.in_features
    h = F.linear(xn, fc1.weight)
    g, u = h[..., :hid], h[..., hid:]
    g = _add(g, _delta(ad, "gate", xn, rowmask))
    u = _add(u, _delta(ad, "up", xn, rowmask))
    inner = F.silu(g) * u
    return _add(F.linear(inner, fc2.weight), _delta(ad, "down", inner, rowmask))


def _qkv(attn, x, ad, rowmask, d):
    xn = attn.layernorm_qkv[0](x)
    qkv = F.linear(xn, attn.layernorm_qkv[1].weight)
    q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
    q = _add(q, _delta(ad, "q", xn, rowmask))
    k = _add(k, _delta(ad, "k", xn, rowmask))
    v = _add(v, _delta(ad, "v", xn, rowmask))
    q = attn.q_ln(q).to(q.dtype)
    k = attn.k_ln(k).to(q.dtype)
    return q, k, v


# ---------------------------------------------------------------------------- #
# Attention with log-sum-exp (fp32 core)                                        #
# ---------------------------------------------------------------------------- #
def _self_attention_with_lse(q, k, v, mask, scale):
    """q,k,v: (N, h, K, dh). mask: bool broadcastable to (N, 1, K, K) or None.
    Returns out (N, h, K, dh) fp32 and lse (N, h, K, 1) fp32."""
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if mask is not None:
        s = s.masked_fill(~mask, _NEG)
    lse = torch.logsumexp(s, dim=-1, keepdim=True)
    out = torch.matmul(torch.exp(s - lse), v.float())
    return out, lse


def _cross_attention_with_lse(q, k, v, keep, scale):
    """Variant queries (N, h, K, dh) against batch-1 reference k/v (1, h, L, dh).

    The N scenarios are folded into the query axis — (1, h, N*K, dh) — so both
    matmul operands are batch-1 and autograd saves no N-expanded copy of the
    reference K/V (with plain broadcasting it saves 2·N·h·L·dh per layer;
    see the Carbon port's docstring). ``keep``: (L,) bool of allowed keys,
    shared by every query. Returns (N, h, K, dh), (N, h, K, 1), fp32."""
    N, h, K, dh = q.shape
    L = k.shape[-2]
    qf = q.float().transpose(0, 1).reshape(h, N * K, dh)          # (h, N*K, dh)
    s = torch.matmul(qf, k[0].float().transpose(-1, -2)) * scale   # (h, N*K, L)
    s = s.masked_fill(~keep, _NEG)
    lse = torch.logsumexp(s, dim=-1, keepdim=True)
    out = torch.matmul(torch.exp(s - lse), v[0].float())           # (h, N*K, dh)
    out = out.view(h, N, K, dh).transpose(0, 1)
    return out, lse.view(h, N, K, 1).transpose(0, 1)


# ---------------------------------------------------------------------------- #
# Efficient variant forward                                                     #
# ---------------------------------------------------------------------------- #
def forward_variants(model: ESMC, ref: ReferenceStates, positions, variant_ids,
                     var_mask=None, output_logits: bool = True,
                     output_layers: bool = False) -> VariantOutput:
    """N variants x K shared token positions against one frozen reference.

    positions: (K,) token indices into ``ref.ids`` (residue i, 1-based -> i).
    variant_ids: (N, K) token ids at those positions.
    var_mask: optional (N, K) bool, False = padded slot (see module docstring).
    Differentiable w.r.t. everything on the variant path (adapters, hooks,
    base weights as used by the variant stream); ``ref`` is treated as given.
    """
    positions, variant_ids, var_mask = _prep(model, ref, positions, variant_ids, var_mask)
    N, K = variant_ids.shape
    L = ref.seq_len
    d, h, dh = model.d_model, model.n_heads, model.head_dim
    scale = dh ** -0.5
    dev = model.device

    x = _embed_variants(model, ref, positions, variant_ids, var_mask)
    cos_q, sin_q = model.rope(positions, x.dtype)
    rowmask = None if var_mask is None else var_mask[..., None]
    sm = _slot_key_mask(model, var_mask, N, K, dev)
    sm4 = None if sm is None else sm[:, None]                  # (N|1, 1, K, K)
    cross = not model.disable_cross_attention
    if cross:
        kv = reference_kv(model, ref)
        keep = torch.ones(L, dtype=torch.bool, device=dev)
        keep[positions] = False

    layer_hidden = []
    for i, blk in enumerate(model.blocks):
        ad = _adapter(model, i)
        attn = blk.attn
        q, k, v = _qkv(attn, x, ad, rowmask, d)
        q = apply_rope(q.view(N, K, h, dh), cos_q, sin_q).transpose(1, 2)
        k = apply_rope(k.view(N, K, h, dh), cos_q, sin_q).transpose(1, 2)
        v = v.view(N, K, h, dh).transpose(1, 2)

        out_s, lse_s = _self_attention_with_lse(q, k, v, sm4, scale)
        if cross:
            k_r, v_r = kv[i]
            out_c, lse_c = _cross_attention_with_lse(q, k_r, v_r, keep, scale)
            lse = torch.logaddexp(lse_c, lse_s)
            o = torch.exp(lse_c - lse) * out_c + torch.exp(lse_s - lse) * out_s
        else:
            o = out_s
        o = o.to(x.dtype).transpose(1, 2).reshape(N, K, d)
        o = _add(attn.out_proj(o), _delta(ad, "o", o, rowmask))
        x = x + o / blk.scaling_factor
        x = x + _ffn(blk, x, ad, rowmask) / blk.scaling_factor

        nxt = ref.layer_inputs[i + 1] if i + 1 < model.n_layers else ref.final_prenorm
        x = _clamp(x, nxt, positions, var_mask)
        if output_layers:
            layer_hidden.append(x)

    hidden = model.transformer.norm(x)
    hidden = _clamp(hidden, ref.final, positions, var_mask)
    logits = model.sequence_head(hidden) if output_logits else None
    return VariantOutput(hidden=hidden, logits=logits,
                         layer_hidden=layer_hidden if output_layers else None,
                         hidden_prenorm=x)


# ---------------------------------------------------------------------------- #
# Bruteforce oracle                                                             #
# ---------------------------------------------------------------------------- #
def _full_block(model: ESMC, blk, x, cos, sin, mask, ad, rowmask):
    """One ordinary full-sequence block (SDPA, boolean mask), with adapter
    deltas applied only on ``rowmask`` rows. Without adapters this is the same
    computation as ``ESMCBlock.forward``."""
    B, L, d = x.shape
    h, dh = model.n_heads, model.head_dim
    attn = blk.attn
    q, k, v = _qkv(attn, x, ad, rowmask, d)
    q = apply_rope(q.view(B, L, h, dh), cos, sin).transpose(1, 2)
    k = apply_rope(k.view(B, L, h, dh), cos, sin).transpose(1, 2)
    v = v.view(B, L, h, dh).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, mask)
    o = o.transpose(1, 2).reshape(B, L, d)
    o = _add(attn.out_proj(o), _delta(ad, "o", o, rowmask))
    x = x + o / blk.scaling_factor
    return x + _ffn(blk, x, ad, rowmask) / blk.scaling_factor


def forward_variants_bruteforce(model: ESMC, ref: ReferenceStates, positions,
                                variant_ids, var_mask=None,
                                output_logits: bool = True,
                                output_layers: bool = False) -> VariantOutput:
    """Correctness oracle for `forward_variants` (same signature/semantics).

    Builds the N full mutant sequences' residual streams as reference states
    with the variant values spliced in, runs ordinary full-sequence blocks,
    and keeps only the variant slots (= clamping every other position back to
    the frozen reference). The key sets of the efficient path are realised as
    an (N, 1, L, L) boolean attention mask on the variant rows. Materializes
    O(N·L·d) activations and O(N·L²) masks: validation / small N only."""
    positions, variant_ids, var_mask = _prep(model, ref, positions, variant_ids, var_mask)
    N, K = variant_ids.shape
    L = ref.seq_len
    dev = model.device

    x = _embed_variants(model, ref, positions, variant_ids, var_mask)
    cos, sin = model.rope(torch.arange(L, device=dev), x.dtype)

    # Key sets for the variant rows: (N, K, L).
    allowed = torch.zeros(N, K, L, dtype=torch.bool, device=dev)
    if not model.disable_cross_attention:
        allowed[:] = True
    sm = _slot_key_mask(model, var_mask, N, K, dev)
    allowed[:, :, positions] = (sm.expand(N, K, K) if sm is not None
                                else torch.ones(N, K, K, dtype=torch.bool, device=dev))
    mask = torch.ones(N, 1, L, L, dtype=torch.bool, device=dev)
    mask[:, 0, positions, :] = allowed   # non-variant rows: anything (discarded)

    rows = torch.zeros(N, L, 1, dtype=torch.bool, device=dev)
    rows[:, positions, 0] = True if var_mask is None else var_mask
    layer_hidden = []
    for i, blk in enumerate(model.blocks):
        full = ref.layer_inputs[i].to(x.dtype).expand(N, L, model.d_model).clone()
        full[:, positions] = x
        full = _full_block(model, blk, full, cos, sin, mask, _adapter(model, i), rows)
        x = full[:, positions]
        nxt = ref.layer_inputs[i + 1] if i + 1 < model.n_layers else ref.final_prenorm
        x = _clamp(x, nxt, positions, var_mask)
        if output_layers:
            layer_hidden.append(x)

    hidden = model.transformer.norm(x)
    hidden = _clamp(hidden, ref.final, positions, var_mask)
    logits = model.sequence_head(hidden) if output_logits else None
    return VariantOutput(hidden=hidden, logits=logits,
                         layer_hidden=layer_hidden if output_layers else None,
                         hidden_prenorm=x)
