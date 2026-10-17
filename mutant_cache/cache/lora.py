"""
mutant_cache/cache/lora.py
--------------------------
Variant-only LoRA for the ESM-C variant cache. Port of
~/svar/CARBON_modules/variant_lora_layers.py.

The idea
--------
`forward_variants` computes two streams with the same weights:

  * the **reference** stream (`forward_reference`) — one batch-1 pass over the
    wild type, producing the per-layer states the variant queries
    cross-attend to;
  * the **variant** stream — the N mutants' tokens at the K mutated positions,
    recomputed layer by layer.

Only the second depends on the mutant, so LoRA deltas are attached to the
variant call sites *only*. The adapters live in ``model.variant_adapters``, a
sibling ``ModuleList`` of ``model.transformer.blocks``. `ESMC.forward` and
`forward_reference` call the stock ``ESMCBlock.forward`` — which has no
reference to the adapters — and the reference K/V (`reference_kv`) use the base
projections explicitly. So the reference stream is base-weights-only by
construction, not by convention; `tests/test_cache_lora.py` asserts it is
bit-identical with random adapters attached.

What that buys
--------------
1. **The reference stream is a constant** while the base is frozen: compute it
   once per target (`ReferenceCache`), no reference backward at all.
2. **bf16 base with fp32 adapters.** The frozen base can sit in bf16; `LoRADelta`
   keeps its parameters in fp32 and casts at use, so gradients and AdamW
   moments live in fp32 (the master-weight scheme for free). Full fine-tuning
   in bf16 loses ~1e-2 relative updates under bf16's 3.9e-3 epsilon.

What it does NOT buy
--------------------
It cannot make the reference positions attend to the mutations: in this
approximation non-variant tokens are always computed as wild type.

Targets
-------
ESM-C fuses q/k/v into one ``layernorm_qkv`` Linear and gate/up into ``ffn[1]``;
adapters are per *logical* projection and their delta is added to the matching
slice: ``q, k, v`` (before q_ln/k_ln), ``o`` (out_proj), ``gate, up`` (fc1
halves, before SiLU / product), ``down`` (fc2). Default: q, o, gate, up, down —
the projections applied only to the variant stream. Adapting k/v means variant
keys and reference keys reach the log-sum-exp merge through different
projections (still a valid softmax over the union, but the two key sets are no
longer commensurate) — off by default so a first result is not confounded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mutant_cache.backbone.esmc import ESMC, swiglu_hidden_dim

ATTN_TARGETS = ("q", "k", "v", "o")
MLP_TARGETS = ("gate", "up", "down")
ALL_TARGETS = ATTN_TARGETS + MLP_TARGETS
DEFAULT_TARGETS = ("q", "o", "gate", "up", "down")


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = DEFAULT_TARGETS

    def __post_init__(self):
        self.targets = tuple(self.targets)
        bad = [t for t in self.targets if t not in ALL_TARGETS]
        if bad:
            raise ValueError(f"unknown LoRA target(s) {bad}; pick from {ALL_TARGETS}")
        if self.r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.r}")

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


class LoRADelta(nn.Module):
    """``B @ A @ x * (alpha/r)`` with fp32 parameters, cast to the activation
    dtype at use (differentiable cast -> fp32 grads / optimizer state).
    ``lora_B`` starts at zero, so a fresh adapter is exactly the base."""

    def __init__(self, in_features: int, out_features: int, cfg: LoRAConfig):
        super().__init__()
        self.scaling = cfg.scaling
        self.lora_A = nn.Parameter(torch.empty(cfg.r, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, cfg.r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(self.dropout(x), self.lora_A.to(x.dtype)),
                        self.lora_B.to(x.dtype)) * self.scaling


class VariantLayerAdapters(nn.Module):
    """Adapters for one block, keyed by logical projection name."""

    def __init__(self, d_model: int, cfg: LoRAConfig):
        super().__init__()
        hid = swiglu_hidden_dim(d_model)
        shapes = {"q": (d_model, d_model), "k": (d_model, d_model),
                  "v": (d_model, d_model), "o": (d_model, d_model),
                  "gate": (d_model, hid), "up": (d_model, hid), "down": (hid, d_model)}
        self.deltas = nn.ModuleDict({t: LoRADelta(*shapes[t], cfg) for t in cfg.targets})

    def delta(self, name: str, x: torch.Tensor):
        return self.deltas[name](x) if name in self.deltas else None


def attach_variant_lora(model: ESMC, cfg: LoRAConfig | None = None,
                        freeze_base: bool = True) -> ESMC:
    """Attach fresh variant-only adapters (fp32, on the model's device) to
    ``model.variant_adapters`` and, by default, freeze the base so only
    ``lora_*`` (and any hook modules) train."""
    cfg = cfg or LoRAConfig()
    model.variant_adapters = nn.ModuleList(
        [VariantLayerAdapters(model.d_model, cfg) for _ in range(model.n_layers)]
    ).to(device=model.device)
    model.lora_config = cfg
    if freeze_base:
        model.freeze_base(True)
    return model


def detach_variant_lora(model: ESMC) -> ESMC:
    model.variant_adapters = None
    return model


def lora_parameters(model: ESMC) -> list[nn.Parameter]:
    return [] if model.variant_adapters is None else list(model.variant_adapters.parameters())


def lora_state_dict(model: ESMC) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def cast_base(model: ESMC, dtype: torch.dtype) -> ESMC:
    """Cast the model to ``dtype`` but keep adapters fp32 (``module.to``
    would drag them along and lose the fp32 master copy)."""
    model.to(dtype=dtype)
    if model.variant_adapters is not None:
        model.variant_adapters.to(dtype=torch.float32)
    return model
