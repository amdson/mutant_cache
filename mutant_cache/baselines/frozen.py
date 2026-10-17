"""
mutant_cache/baselines/frozen.py
--------------------------------
Ridge and a small MLP on frozen mean-pooled embeddings (PLAN.md 21).

DATASETS.md calls this "the real supervised bar, given TabPFN3 at 0.767 mean
Spearman on ESM-C 300M features" — i.e. the number the whole architecture has
to beat, and the one most likely to embarrass it.

**These take the embedding matrix as an argument.**  They never touch a
backbone, so they are runnable the moment the data layer lands and do not wait
on ``mutant_cache/backbone``.  Produce ``X`` however you like — mean-pooled
mutant embeddings, WT-minus-mutant differences, or the variant-cache hidden
states — and pass it in aligned row-for-row with the variants frame.
``embed_fn`` exists only as a convenience for when a backbone *is* available.

Both are fit per score set by default, for the same reason as the one-hot
ridge: across-assay scores are not comparable, so a single pooled fit measures
mostly assay identity.  ``per_assay=False`` gives the pooled variant, which is
the right control for "does a single model across MaveDB help at all".
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler


def mean_pooled_embeddings(model, sequences, mutants, *, device=None,
                           batch_size: int = 16) -> np.ndarray:
    """Convenience only — mean-pool ``model(ids).hidden`` over residues.

    Used when a §1 backbone happens to be around; every estimator below works
    from a plain matrix and never calls this.
    """
    device = device or next(model.parameters()).device
    out = []
    with torch.no_grad():
        for s in range(0, len(sequences), batch_size):
            for seq in sequences[s:s + batch_size]:
                ids = model.tokenizer.encode(seq).to(device)
                h = model(ids[None]).hidden[0, 1:-1]
                out.append(h.mean(0).float().cpu().numpy())
    return np.stack(out)


class _PerAssay:
    """Shared plumbing: fit one estimator per score set (or one pooled)."""

    kind = "baseline"

    def __init__(self, *, per_assay: bool = True, standardize: bool = True):
        self.per_assay = per_assay
        self.standardize = standardize
        self.models: dict[object, object] = {}
        self.scalers: dict[object, StandardScaler] = {}

    def _new(self):                              # pragma: no cover - abstract
        raise NotImplementedError

    def _groups(self, variants):
        if not self.per_assay or "score_set_id" not in getattr(variants, "columns", []):
            yield "__all__", np.arange(len(variants))
            return
        ss = variants["score_set_id"].to_numpy()
        for g in np.unique(ss):
            yield g, np.flatnonzero(ss == g)

    def fit(self, X, variants, target_col="score_rank", y=None):
        X = np.asarray(X, dtype=np.float64)
        y = (np.asarray(y, dtype=float) if y is not None
             else variants[target_col].to_numpy(dtype=float))
        for g, idx in self._groups(variants):
            xi = X[idx]
            if self.standardize:
                sc = StandardScaler().fit(xi)
                self.scalers[g] = sc
                xi = sc.transform(xi)
            self.models[g] = self._fit_one(xi, y[idx])
        return self

    def predict(self, X, variants) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.full(len(X), np.nan)
        for g, idx in self._groups(variants):
            m = self.models.get(g) or self.models.get("__all__")
            if m is None:
                continue
            xi = X[idx]
            sc = self.scalers.get(g) or self.scalers.get("__all__")
            if sc is not None:
                xi = sc.transform(xi)
            out[idx] = self._predict_one(m, xi)
        return out


class EmbeddingRidge(_PerAssay):
    """Ridge on a frozen embedding matrix."""

    name = "embedding_ridge"

    def __init__(self, *, alpha=1.0, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0),
                 per_assay=True, standardize=True):
        super().__init__(per_assay=per_assay, standardize=standardize)
        self.alpha, self.alphas = alpha, alphas

    def _fit_one(self, X, y):
        est = (RidgeCV(alphas=self.alphas) if self.alphas is not None
               else Ridge(alpha=self.alpha))
        return est.fit(X, y)

    def _predict_one(self, m, X):
        return m.predict(X)


class EmbeddingMLP(_PerAssay):
    """A two-hidden-layer MLP on a frozen embedding matrix, in torch.

    Small on purpose: this is a baseline, not a second model.  Early stopping
    on a random 10% holdout, because with a few hundred variants per assay an
    MLP will otherwise memorise the set and report a baseline that is really a
    measure of training-set size.
    """

    name = "embedding_mlp"

    def __init__(self, *, hidden=(256, 64), lr=1e-3, epochs=300, weight_decay=1e-4,
                 batch_size=256, dropout=0.1, val_frac=0.1, patience=30,
                 per_assay=True, standardize=True, seed=0, device="cpu"):
        super().__init__(per_assay=per_assay, standardize=standardize)
        self.hidden, self.lr, self.epochs = hidden, lr, epochs
        self.weight_decay, self.batch_size, self.dropout = weight_decay, batch_size, dropout
        self.val_frac, self.patience, self.seed, self.device = val_frac, patience, seed, device

    def _net(self, d_in):
        layers, d = [], d_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def _fit_one(self, X, y):
        torch.manual_seed(self.seed)
        dev = torch.device(self.device)
        X = torch.tensor(X, dtype=torch.float32, device=dev)
        y = torch.tensor(y, dtype=torch.float32, device=dev)
        n = len(X)
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g)
        nv = max(1, int(self.val_frac * n)) if n > 10 else 0
        vi, ti = perm[:nv].to(dev), perm[nv:].to(dev)
        net = self._net(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        best, best_state, bad = float("inf"), None, 0
        for ep in range(self.epochs):
            net.train()
            order = ti[torch.randperm(len(ti), generator=g).to(dev)]
            for s in range(0, len(order), self.batch_size):
                b = order[s:s + self.batch_size]
                opt.zero_grad(set_to_none=True)
                nn.functional.mse_loss(net(X[b]).squeeze(-1), y[b]).backward()
                opt.step()
            if nv:
                net.eval()
                with torch.no_grad():
                    v = float(nn.functional.mse_loss(net(X[vi]).squeeze(-1), y[vi]))
                if v < best - 1e-6:
                    best, bad = v, 0
                    best_state = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def _predict_one(self, m, X):
        with torch.no_grad():
            return m(torch.tensor(X, dtype=torch.float32,
                                  device=next(m.parameters()).device)
                     ).squeeze(-1).cpu().numpy()
