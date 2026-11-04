"""Controlled stability-training objectives and measured-background augmentation."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mutant_cache.data.mutations import mutate, diff_sequences


def stability_loss(pred, target, n_mut, *, kind="mse_dg", delta=1.,
                   single_weight=1., double_weight=1., positive_weight=1., reduction="rows"):
    """Canonical positive-stabilizing scores; positive double weight uses >= 0.

    ``depth`` matches the upstream strategy of separate single/double means;
    ``rows`` uses one normalized weighted mean. Higher orders join the double
    task when background augmentation creates measured relative labels.
    """
    if min(single_weight, double_weight, positive_weight, delta) <= 0:
        raise ValueError("loss weights and Huber delta must be positive")
    if kind == "mse_dg":
        point = (pred-target).square()
    elif kind == "huber_dg":
        point = F.huber_loss(pred, target, reduction="none", delta=delta)
    else:
        raise ValueError("weighted stability loss requires mse_dg or huber_dg")
    multi = n_mut > 1
    weights = torch.where(multi & (target >= 0), positive_weight, 1.).to(pred.dtype)
    if reduction == "rows":
        weights = weights * torch.where(multi, double_weight, single_weight)
        return (point*weights).sum()/weights.sum()
    if reduction != "depth":
        raise ValueError("reduction must be rows or depth")
    loss = pred.sum()*0
    for mask, factor in ((~multi, single_weight), (multi, double_weight)):
        if mask.any():
            loss = loss + factor*(point[mask]*weights[mask]).sum()/weights[mask].sum()
    return loss


class ReferenceAugmentor:
    """A bounded bank of *training-only* measured single-mutant references.

    Reverse mode replaces a fraction of rows with inverse mutations. Background
    mode re-references a whole microbatch, using exact differences of measured
    labels, and includes the inverse-to-WT example. It can create 3-mutant
    examples from original doubles without assuming additive effects.
    """
    def __init__(self, frame, sequences, *, mode, fraction=.25, bank_size=4, seed=0):
        if mode not in ("reverse", "background") or not 0 < fraction <= 1 or bank_size < 1:
            raise ValueError("invalid reference augmentation settings")
        self.mode, self.fraction = mode, fraction
        self.rng = np.random.default_rng(seed+1907)
        self.original_sequences = dict(sequences)
        self.sequences, self.banks, records = {}, {}, []
        for tid, group in frame[frame.n_mut == 1].groupby("target_id", sort=True):
            unique = group.drop_duplicates("mutant")
            selected = self.rng.choice(len(unique), min(bank_size, len(unique)), replace=False)
            bank = []
            for i in selected:
                row = unique.iloc[int(i)]
                key = f"{tid}::measured:{row.mutant}"
                seq = mutate(sequences[tid], row.mutant)
                self.sequences[key] = seq
                item = dict(target_id=tid, reference_id=key, mutant=row.mutant, score=float(row.score))
                records.append(item)
                bank.append(item)
            self.banks[tid] = bank
        self.audit = pd.DataFrame(records)
        self.transformed_rows = 0

    def apply(self, batch):
        if batch.target_id.nunique() != 1:
            raise ValueError("augmentation expects one protein per microbatch")
        tid = batch.target_id.iloc[0]
        bank = self.banks.get(tid, [])
        if not bank:
            return batch
        out = batch.copy().reset_index(drop=True)
        if self.mode == "background":
            if self.rng.random() >= self.fraction:
                return out
            anchor = bank[int(self.rng.integers(len(bank)))]
            ref = self.sequences[anchor["reference_id"]]
            seqs = [mutate(self.original_sequences[tid], m) for m in out.mutant]
            # Include the WT target, which is an experimentally defined zero.
            seqs[0] = self.original_sequences[tid]
            scores = out.score.to_numpy(copy=True)
            scores[0] = 0.
            out["mutant"] = [diff_sequences(ref, s) for s in seqs]
            out["score"] = scores-anchor["score"]
            out["target_id"] = anchor["reference_id"]
            out["score_set_id"] = anchor["reference_id"]
            out["n_mut"] = out.mutant.map(lambda m: m.count(":")+1 if m else 0)
            self.transformed_rows += len(out)
            return out
        count = max(1, round(len(out)*self.fraction))
        for index in self.rng.choice(len(out), count, replace=False):
            anchor = bank[int(self.rng.integers(len(bank)))]
            ref = self.sequences[anchor["reference_id"]]
            out.loc[index, ["target_id", "score_set_id", "mutant", "n_mut", "score"]] = [
                anchor["reference_id"], anchor["reference_id"],
                diff_sequences(ref, self.original_sequences[tid]), 1, -anchor["score"]]
        self.transformed_rows += count
        return out
