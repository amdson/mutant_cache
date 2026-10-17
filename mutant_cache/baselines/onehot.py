"""
mutant_cache/baselines/onehot.py
--------------------------------
Ridge on one-hot substitutions, fit **per assay** (PLAN.md 21, the FLIP
baseline).

Deliberately the dumbest thing that can work: a variant is a (L x 20) indicator
over its substitutions, a ridge maps that to the score, and each score set gets
its own fit because the scales are not comparable.  It is a purely additive
per-(position, residue) model, which makes it the honest floor for the whole
project — DATASETS.md notes FLIP2's headline finding is that "simple models
often beat fine-tuned pLMs", and CombinGym's leaderboard carries Ridge for the
same reason.

It also doubles as a diagnostic: a one-hot ridge *cannot* represent epistasis,
so the gap between it and the model on depth 2+ is the clearest statement of
what the interaction module bought.  ``interaction_order=2`` optionally adds
pairwise terms for the sites a landscape actually varies (tractable only on
combinatorial sets like GB1 or CR9114, which is why it is off by default).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge, RidgeCV

from mutant_cache.models.data_compat import parse_mutant

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}


def onehot_features(mutants, seq_len: int, *, interaction_order: int = 1,
                    pair_index: dict | None = None):
    """(N, F) CSR matrix. F = seq_len*20 (+ pair block if order 2)."""
    n = len(mutants)
    base = seq_len * len(AA)
    rows, cols = [], []
    for i, m in enumerate(mutants):
        muts = parse_mutant(m)
        idx = []
        for (p, _ref, alt) in muts:
            j = (p - 1) * len(AA) + AA_INDEX[alt]
            idx.append(j)
            rows.append(i)
            cols.append(j)
        if interaction_order >= 2 and pair_index is not None:
            for a, b in combinations(sorted(idx), 2):
                k = pair_index.get((a, b))
                if k is not None:
                    rows.append(i)
                    cols.append(base + k)
    f = base + (len(pair_index) if (interaction_order >= 2 and pair_index) else 0)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, f))


def build_pair_index(mutants) -> dict:
    """Pair features for exactly the (pos, alt) combinations that occur."""
    seen = set()
    for m in mutants:
        idx = sorted((p - 1) * len(AA) + AA_INDEX[a] for p, _, a in parse_mutant(m))
        seen.update(combinations(idx, 2))
    return {pair: k for k, pair in enumerate(sorted(seen))}


class OneHotRidgePerAssay:
    """One ridge per ``score_set_id``, on one-hot substitution indicators."""

    name = "onehot_ridge"
    kind = "baseline"

    def __init__(self, sequences: dict[str, str], *, alpha=1.0, alphas=None,
                 interaction_order: int = 1, fit_intercept: bool = True):
        self.sequences = sequences
        self.alpha = alpha
        self.alphas = alphas
        self.interaction_order = interaction_order
        self.fit_intercept = fit_intercept
        self.models: dict[str, object] = {}
        self.pair_index: dict[str, dict] = {}
        self.target_of: dict[str, str] = {}

    def _X(self, ssid, mutants):
        L = len(self.sequences[self.target_of[ssid]])
        return onehot_features(mutants, L,
                               interaction_order=self.interaction_order,
                               pair_index=self.pair_index.get(ssid))

    def fit(self, variants, target_col="score_rank"):
        for ssid, sub in variants.groupby("score_set_id", observed=True):
            self.target_of[ssid] = sub["target_id"].iloc[0]
            muts = sub["mutant"].fillna("").tolist()
            if self.interaction_order >= 2:
                self.pair_index[ssid] = build_pair_index(muts)
            X = self._X(ssid, muts)
            y = sub[target_col].to_numpy(dtype=float)
            est = (RidgeCV(alphas=self.alphas, fit_intercept=self.fit_intercept)
                   if self.alphas is not None
                   else Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept))
            self.models[ssid] = est.fit(X, y)
        return self

    def predict(self, variants) -> np.ndarray:
        out = np.full(len(variants), np.nan)
        pos = {ix: i for i, ix in enumerate(variants.index)}
        for ssid, sub in variants.groupby("score_set_id", observed=True):
            est = self.models.get(ssid)
            if est is None:                      # unseen assay: no per-assay fit
                continue
            X = self._X(ssid, sub["mutant"].fillna("").tolist())
            out[[pos[i] for i in sub.index]] = est.predict(X)
        return out
