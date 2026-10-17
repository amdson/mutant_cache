"""
mutant_cache/baselines/zero_shot.py
-----------------------------------
ESM-C masked-marginal scoring (PLAN.md 21), which is two things at once:

* the zero-shot baseline — and DATASETS.md is emphatic that it belongs in a
  *context* row, never beside a supervised number;
* the initialisation of the model's additive component (PLAN.md 18), which is
  why this returns a full ``(L+2, V)`` log-probability matrix rather than just
  a score: ``VariantEffectModel`` indexes it per variant token.

Scoring is the standard masked-marginal LLR (Meier et al. 2021):

    score(mutant) = Σ_k [ log p(alt_k | x_masked at k) − log p(ref_k | x_masked at k) ]

one forward per mutated position, the position masked.  ``mode="wt"`` instead
takes a single unmasked forward (cheaper, slightly worse); ``mode="mutant"``
masks on the mutant sequence, which is what ESMRank's ``esm_score`` feature
uses and is not the same quantity.

Only INTERFACES.md §1 is assumed: ``model(ids)`` returning ``.logits``,
``model.tokenizer.encode``, ``model.aa_to_id``.
"""
from __future__ import annotations

import numpy as np
import torch

from mutant_cache.models.data_compat import parse_mutant


def _mask_id(model) -> int:
    for obj, attr in ((model, "mask_token_id"), (model, "mask_id"),
                      (getattr(model, "tokenizer", None), "mask_id"),
                      (getattr(model, "tokenizer", None), "mask_token_id")):
        v = getattr(obj, attr, None) if obj is not None else None
        if v is not None:
            return int(v)
    raise ValueError("backbone exposes no mask token id; pass mask_id=")


@torch.no_grad()
def masked_marginal_matrix(
    model,
    sequence: str,
    *,
    positions: list[int] | None = None,
    mode: str = "masked",
    mask_id: int | None = None,
    batch_size: int = 32,
    device=None,
) -> torch.Tensor:
    """``(L+2, V)`` log-probabilities, one row per token position.

    ``positions`` are token indices (1-based residue + BOS offset); ``None``
    scores every residue.  Rows not scored are left at ``-inf``-free zeros, so
    index only what you asked for.
    """
    ids = model.tokenizer.encode(sequence)
    device = device or next(model.parameters()).device
    ids = ids.to(device)
    seq_len = ids.shape[0]
    out = torch.zeros(seq_len, int(model(ids[None]).logits.shape[-1]),
                      device="cpu", dtype=torch.float32)

    if mode == "wt":
        lp = torch.log_softmax(model(ids[None]).logits[0].float(), -1)
        return lp.cpu()

    pos = list(range(1, seq_len - 1)) if positions is None else list(positions)
    mid = mask_id if mask_id is not None else _mask_id(model)
    for s in range(0, len(pos), batch_size):
        chunk = pos[s:s + batch_size]
        batch = ids[None].repeat(len(chunk), 1).clone()
        for i, p in enumerate(chunk):
            batch[i, p] = mid
        lg = model(batch).logits.float()
        lp = torch.log_softmax(lg, -1)
        for i, p in enumerate(chunk):
            out[p] = lp[i, p].cpu()
    return out


def llr_matrix(model, sequence: str, **kw) -> torch.Tensor:
    """``(L+2, V)`` log-likelihood *ratios* against the wild-type residue.

    This is the tensor ``VariantEffectModel`` wants for
    ``init_additive_from_marginals``: entry ``[p, a]`` is
    ``log p(a | masked) − log p(wt_p | masked)``, so the wild-type column is 0
    and a padded (non-substituted) variant slot contributes nothing.
    """
    lp = masked_marginal_matrix(model, sequence, **kw)
    ids = model.tokenizer.encode(sequence)
    wt = lp.gather(1, ids[:, None].cpu()).squeeze(1)
    return lp - wt[:, None]


def build_llr_matrices(model, sequences: dict[str, str], **kw) -> dict[str, torch.Tensor]:
    return {tid: llr_matrix(model, seq, **kw) for tid, seq in sequences.items()}


def score_variants(
    variants,
    llr: dict[str, torch.Tensor],
    *,
    aa_to_id: dict[str, int],
    bos_offset: int = 1,
    target_col: str = "target_id",
    mutant_col: str = "mutant",
) -> np.ndarray:
    """Summed masked-marginal LLR per row of a §2 ``variants`` frame."""
    out = np.zeros(len(variants), dtype=float)
    for i, r in enumerate(variants.itertuples()):
        M = llr.get(getattr(r, target_col))
        if M is None:
            out[i] = np.nan
            continue
        tot = 0.0
        for (p, _ref, alt) in parse_mutant(getattr(r, mutant_col)):
            tot += float(M[p + bos_offset, aa_to_id[alt]])
        out[i] = tot
    return out


class MaskedMarginalZeroShot:
    """Sklearn-ish wrapper so the harness can treat it like the other baselines.

    ``fit`` is a no-op *by design* — it sees no labels, which is the whole
    reason its row in the results table is labelled context-only.
    """

    name = "esmc_masked_marginal"
    kind = "zero_shot"

    def __init__(self, model, sequences: dict[str, str], *, mode="masked", **kw):
        self.model = model
        self.sequences = sequences
        self.aa_to_id = dict(model.aa_to_id)
        self.llr = build_llr_matrices(model, sequences, mode=mode, **kw)

    def fit(self, *a, **kw):
        return self

    def predict(self, variants) -> np.ndarray:
        return score_variants(variants, self.llr, aa_to_id=self.aa_to_id)
