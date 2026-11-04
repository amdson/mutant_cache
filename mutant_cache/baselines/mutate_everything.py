"""Supplied-library decoding through the original Mutate Everything modules.

Accepts the upstream model, not our backbone port. The same trained modules
are used; only requested mutation sets are gathered instead of an L² landscape.
The dense original decoder is the correctness oracle.
"""
from __future__ import annotations
import torch
from mutant_cache.data.mutations import parse_mutant

AA = "ACDEFGHIKLMNPQRSTVWY"


@torch.no_grad()
def score_requested(model, tokens, sequence, mutants, batch_size=512):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if model.training:
        raise ValueError("requested decoding is inference-only")
    batch = {"seqs": [sequence]}
    pred = model.backbone(tokens, batch)
    pred.update(model.aa_expansion(tokens, batch, pred))
    pred.update(model.single_decoder(tokens, batch, pred))
    single = pred["mut1_ddg"][0]
    multi = model.multi_decoder
    features = multi.multi_adapter(pred["mut1_feat"])[0] if hasattr(multi, "multi_adapter") else None
    output = []
    for start in range(0, len(mutants), batch_size):
        sets = [parse_mutant(m) for m in mutants[start:start+batch_size]]
        if any(not s for s in sets):
            raise ValueError("empty mutation set")
        width = max(map(len, sets))
        positions = torch.zeros(len(sets), width, dtype=torch.long, device=single.device)
        amino = torch.zeros_like(positions)
        keep = torch.zeros_like(positions, dtype=torch.bool)
        for i, subs in enumerate(sets):
            if len({p for p, _, _ in subs}) != len(subs):
                raise ValueError("duplicate mutation position")
            for j, (p, wt, alt) in enumerate(subs):
                if p > len(sequence) or sequence[p-1] != wt:
                    raise ValueError(f"WT mismatch in {mutants[start+i]}")
                positions[i, j], amino[i, j], keep[i, j] = p-1, AA.index(alt), True
        singles = (single[positions, amino] * keep).sum(-1)
        if features is not None:
            summed = (features[positions, amino] * keep[..., None]).sum(1)
            interaction = multi.head(summed).squeeze(-1)
            values = torch.where(keep.sum(-1) == 1, singles, singles+interaction)
        else:
            values = singles
        output.append(values)
    return torch.cat(output) if output else single.new_empty(0)
