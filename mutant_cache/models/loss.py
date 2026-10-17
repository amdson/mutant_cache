"""
mutant_cache/models/loss.py
---------------------------
Objectives for supervised variant-effect training (PLAN.md 19).

Three families, because the three dataset flavours in DATASETS.md want
different things:

* **Within-score-set ranking** — the MaveDB-wide objective.  Each score set has
  its own arbitrary monotone link between the latent trait and the reported
  number (DATASETS.md: "a model trained across score sets will spend capacity
  learning those links"), so the only part of the label that transfers is the
  *order* within a set.  Comparing scores across sets is meaningless, so every
  pair, list and softmax here is confined to one score set — enforced by
  ``ScoreSetBatchSampler`` upstream and re-checked by ``group`` here.

* **MSE on rank-normalized targets** — the cheap, stable fallback.  §2's
  ``score_rank`` is already uniform on (0,1) per score set, which absorbs the
  link well enough for a first run.

* **MSE on ΔG** — Megascale only, where the units are physical and additive, so
  an additive model is the right null and ε = ΔΔG_double − Σ ΔΔG_singles is
  meaningful.  Optionally inverse-variance weighted by the reported
  uncertainty, which is also the reliability ceiling eval uses.

What ESMRank actually does
--------------------------
DATASETS.md cites ESMRank (Arnese & Gambardella 2026, bioRxiv 2026.02.26.708185)
as the closest existing work and says to read it before building.  Its v1 has no
Methods section; the objective is pinned by the released artifact
(``models/esm_xgb_rank_rrf.json``, Zenodo 18773439), and it is worth being
precise about because it is *not* what the name suggests:

* It is **XGBoost LambdaMART on 19 frozen features**, three of them from a
  frozen ESM-2 650M (masked-marginal LLR, mean-pooled embedding distance,
  contact-map distance).  No PLM weights are ever updated.
* ``objective = "rank:pairwise"``: the LambdaRank kernel with the metric delta
  fixed at ΔZ ≡ 1, i.e. plain **RankNet logistic with σ hard-coded to 1**,
  **no margin**, **no |ΔNDCG| weighting**, **no auxiliary MSE**.
* ``lambdarank_num_pair_per_sample = 20``, ``pair_method = "mean"``: 20 partners
  sampled uniformly per item per round, not all pairs.
* **Groups are proteins, not score sets** — they can do that only because the
  labels are already an RRF fusion (``1/(60+rank)``, min-max'd) of the
  overlapping score sets, which erases per-assay scale *before* the loss.
* ``lambdarank_normalization = 0``: no per-group normalization, so a
  40,805-variant protein contributes ~2900x the gradient of a 14-variant one.
* Numbers to quote: 5-fold CV mean per-assay Spearman **0.55** (vs 0.50 for
  regression on identical features, 0.32 for the ESM-2 zero-shot feature
  alone); Human Domainome median ρ **0.62** (ThermoMPNN 0.46).

Three of those choices are ones to *not* copy here.  We group by score set
(``ScoreSetBatchSampler``) because we have no RRF fusion and raw cross-assay
scores are incomparable; we normalize per group, because otherwise MaveDB's
handful of mega-DMS sets own the objective; and we expose pair weighting by
label gap and by inverse measurement variance, which ESMRank does not use at
all despite the standard errors being right there in MaveDB.

The pairwise form itself is RankNet's: for a pair (i, j) with y_i > y_j, the
model is penalised by ``softplus(-σ · (ŷ_i - ŷ_j - margin))``, the logistic
log-likelihood of ordering the pair correctly.  σ = 1 matches ESMRank; margin
> 0 demands a gap rather than just the right sign; ``min_gap`` drops pairs whose
labels differ by less than the assay's noise, which is the cheap version of
uncertainty weighting and matters a lot on the 30-52% of human score sets whose
reliability is below 0.9.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSSES = ("rank", "mse_rank", "mse_dg", "listnet", "listmle", "soft_spearman")


def _group_eq(group, n, device):
    """(n, n) bool: entry (i, j) True iff i and j are comparable."""
    if group is None:
        return torch.ones(n, n, dtype=torch.bool, device=device)
    if not torch.is_tensor(group):
        uniq = {g: i for i, g in enumerate(sorted(set(group)))}
        group = torch.tensor([uniq[g] for g in group], device=device)
    return group[:, None] == group[None, :]


# ── ranking ──────────────────────────────────────────────────────────────────

def pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    group=None,
    *,
    sigma: float = 1.0,
    margin: float = 0.0,
    min_gap: float = 0.0,
    weight: torch.Tensor | None = None,
    se: torch.Tensor | None = None,
    pair_weight: str = "uniform",
    num_pair_per_sample: int = 0,
    generator: torch.Generator | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """RankNet logistic pairwise loss, restricted to within-group pairs.

    ``pred``/``target``: (N,).  ``group``: (N,) ids (ints, strings or a tensor);
    ``None`` means the whole batch is one score set, which is what
    ``ScoreSetBatchSampler`` guarantees.

    ``pair_weight``
        ``"uniform"``   every ordered pair counts the same (ESMRank's setting).
        ``"gap"``       weight by ``|y_i - y_j|``.  This is the cheap difference
                        between a Kendall-τ surrogate and a Spearman-ish one:
                        getting a large-gap pair backwards should cost more.
        ``"inv_var"``   weight by ``1/(se_i² + se_j²)``, so pairs the assay
                        cannot actually resolve stop driving the gradient.
                        Needs ``se``.
        ``"gap_inv_var"`` both.
    ``num_pair_per_sample``
        >0 samples that many partners per item instead of using all O(N²)
        pairs (ESMRank uses 20).  Only worth it for large batches; the full
        pair set is cheap at the batch sizes the cache path can afford.
    ``weight``
        per-item weight folded in as ``sqrt(w_i · w_j)`` — use it for per-assay
        reweighting when a batch does span score sets.
    """
    n = pred.shape[0]
    same = _group_eq(group, n, pred.device)
    tdiff = target[:, None] - target[None, :]
    valid = same & (tdiff > min_gap)
    if num_pair_per_sample and num_pair_per_sample < n:
        keep = torch.zeros(n, n, dtype=torch.bool, device=pred.device)
        idx = torch.randint(0, n, (n, int(num_pair_per_sample)),
                            device=pred.device, generator=generator)
        keep.scatter_(1, idx, True)
        valid = valid & (keep | keep.t())
    if not bool(valid.any()):
        return pred.sum() * 0.0

    w = valid.to(pred.dtype)
    if "gap" in pair_weight:
        w = w * tdiff.abs()
    if "inv_var" in pair_weight:
        if se is None:
            raise ValueError("pair_weight='inv_var' needs per-item `se`")
        v = se.clamp_min(1e-6) ** 2
        w = w / (v[:, None] + v[None, :])
    if weight is not None:
        w = w * torch.sqrt(weight[:, None] * weight[None, :])

    sdiff = pred[:, None] - pred[None, :]
    l = F.softplus(-sigma * (sdiff - margin)) * w
    if reduction == "sum":
        return l.sum()
    return l.sum() / w.sum().clamp_min(1e-8)


def listnet_loss(pred, target, group=None, *, temperature: float = 1.0):
    """ListNet top-1: cross-entropy between softmax(ŷ/τ) and softmax(y/τ),
    per group.  Listwise, so it sees the whole score set's shape at once."""
    n = pred.shape[0]
    same = _group_eq(group, n, pred.device)
    tot, cnt = pred.sum() * 0.0, 0
    for g in _groups(same):
        if g.numel() < 2:
            continue
        p = torch.log_softmax(pred[g] / temperature, 0)
        q = torch.softmax(target[g] / temperature, 0)
        tot = tot - (q * p).sum()
        cnt += 1
    return tot / max(cnt, 1)


def listmle_loss(pred, target, group=None):
    """ListMLE: negative log Plackett-Luce likelihood of the true ordering."""
    n = pred.shape[0]
    same = _group_eq(group, n, pred.device)
    tot, cnt = pred.sum() * 0.0, 0
    for g in _groups(same):
        if g.numel() < 2:
            continue
        order = torch.argsort(target[g], descending=True)
        s = pred[g][order]
        # log Σ_{j>=i} exp(s_j), computed right-to-left
        rev_lse = torch.logcumsumexp(s.flip(0), 0).flip(0)
        tot = tot + (rev_lse - s).sum() / g.numel()
        cnt += 1
    return tot / max(cnt, 1)


def soft_spearman_loss(pred, target, group=None, *, tau: float = 0.05):
    """1 − Pearson(softrank(ŷ), rank(y)), a differentiable Spearman surrogate.

    Soft ranks are the O(n²) sigmoid form ``r_i = Σ_j σ((ŷ_i − ŷ_j)/τ)``; exact
    in the τ→0 limit and smooth for τ > 0.  Directly optimises the reported
    metric, at the cost of being badly scaled when a batch has ties.
    """
    n = pred.shape[0]
    same = _group_eq(group, n, pred.device)
    tot, cnt = pred.sum() * 0.0, 0
    for g in _groups(same):
        if g.numel() < 3:
            continue
        p, t = pred[g], target[g]
        sr = torch.sigmoid((p[:, None] - p[None, :]) / tau).sum(1)
        tr = (t[:, None] > t[None, :]).float().sum(1)
        tot = tot + (1.0 - _pearson(sr, tr))
        cnt += 1
    return tot / max(cnt, 1)


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-8)


def _groups(same: torch.Tensor):
    """Index tensors for each distinct group, from the (N, N) equality mask."""
    n = same.shape[0]
    seen = torch.zeros(n, dtype=torch.bool, device=same.device)
    out = []
    for i in range(n):
        if seen[i]:
            continue
        g = torch.nonzero(same[i], as_tuple=False).squeeze(-1)
        seen[g] = True
        out.append(g)
    return out


# ── regression ───────────────────────────────────────────────────────────────

def mse_rank_loss(pred, target, weight=None):
    """Plain MSE against ``score_rank`` (§2: uniform on (0,1) per score set)."""
    l = (pred - target) ** 2
    if weight is None:
        return l.mean()
    return (l * weight).sum() / weight.sum().clamp_min(1e-8)


def mse_dg_loss(pred, dg, se=None, *, min_se: float = 0.05,
                inverse_variance: bool = True):
    """MSE on ΔG for Megascale, optionally inverse-variance weighted.

    Weighting by 1/σ² is the same information the reliability ceiling is built
    from, so a run that reports against the ceiling should usually also train
    against it — otherwise the loss spends capacity on the least trustworthy
    measurements in the set.
    """
    l = (pred - dg) ** 2
    if se is None or not inverse_variance:
        return l.mean()
    w = 1.0 / se.clamp_min(min_se) ** 2
    return (l * w).sum() / w.sum().clamp_min(1e-8)


# ── the combination actually used ────────────────────────────────────────────

@dataclass
class LossConfig:
    kind: str = "rank"                 # any of LOSSES, or "rank+mse_rank"
    rank_weight: float = 1.0
    mse_weight: float = 1.0
    sigma: float = 1.0                 # ESMRank's value, and XGBoost's hard-coded one
    margin: float = 0.0
    min_gap: float = 0.0
    pair_weight: str = "uniform"       # uniform | gap | inv_var | gap_inv_var
    num_pair_per_sample: int = 0       # 0 = all pairs; ESMRank samples 20
    temperature: float = 1.0
    tau: float = 0.05
    inverse_variance: bool = False
    min_se: float = 0.05

    def to_dict(self):
        return asdict(self)


class VariantLoss(nn.Module):
    """Dispatch over the objectives above; ``kind`` may be a ``+``-joined mix.

    Call with ``(pred, target, group=..., se=..., weight=...)``.  Returns
    ``(total, parts)`` so the harness can log each term.
    """

    def __init__(self, cfg: LossConfig | None = None, **kw):
        super().__init__()
        self.cfg = cfg or LossConfig(**kw)
        for part in self.cfg.kind.split("+"):
            if part not in LOSSES:
                raise ValueError(f"unknown loss {part!r}; choose from {LOSSES}")

    def forward(self, pred, target, *, group=None, se=None, weight=None):
        c = self.cfg
        parts: dict[str, torch.Tensor] = {}
        for part in c.kind.split("+"):
            if part == "rank":
                v = pairwise_rank_loss(
                    pred, target, group, sigma=c.sigma, margin=c.margin,
                    min_gap=c.min_gap, weight=weight, se=se,
                    pair_weight=c.pair_weight,
                    num_pair_per_sample=c.num_pair_per_sample)
                parts[part] = c.rank_weight * v
            elif part == "listnet":
                parts[part] = c.rank_weight * listnet_loss(
                    pred, target, group, temperature=c.temperature)
            elif part == "listmle":
                parts[part] = c.rank_weight * listmle_loss(pred, target, group)
            elif part == "soft_spearman":
                parts[part] = c.rank_weight * soft_spearman_loss(
                    pred, target, group, tau=c.tau)
            elif part == "mse_rank":
                parts[part] = c.mse_weight * mse_rank_loss(pred, target, weight)
            elif part == "mse_dg":
                parts[part] = c.mse_weight * mse_dg_loss(
                    pred, target, se, min_se=c.min_se,
                    inverse_variance=c.inverse_variance)
        total = sum(parts.values())
        return total, {k: float(v.detach()) for k, v in parts.items()}


# ── batching ─────────────────────────────────────────────────────────────────

class ScoreSetBatchSampler(torch.utils.data.Sampler):
    """Yield index batches that never cross a score set.

    The ranking losses above are only meaningful within one assay, so this is
    not a nicety: a batch mixing score sets makes most of its pairs noise, and
    the model can lower the loss by learning per-assay offsets instead of
    within-assay order.  ``drop_singletons`` removes batches of size 1, which
    contribute no pairs at all.
    """

    def __init__(self, score_set_ids, batch_size: int, *, shuffle: bool = True,
                 drop_last: bool = False, drop_singletons: bool = True,
                 seed: int = 0, min_group: int = 2):
        import numpy as np
        self.ids = list(score_set_ids)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.drop_singletons = drop_singletons
        self.seed = seed
        self.min_group = min_group
        self.epoch = 0
        self._np = np
        groups: dict[object, list[int]] = {}
        for i, s in enumerate(self.ids):
            groups.setdefault(s, []).append(i)
        self.groups = {k: v for k, v in groups.items() if len(v) >= min_group}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self):
        np = self._np
        rng = np.random.default_rng(self.seed + self.epoch)
        out = []
        for _, idx in self.groups.items():
            idx = np.array(idx)
            if self.shuffle:
                rng.shuffle(idx)
            for s in range(0, len(idx), self.batch_size):
                b = idx[s:s + self.batch_size].tolist()
                if self.drop_last and len(b) < self.batch_size:
                    continue
                if self.drop_singletons and len(b) < 2:
                    continue
                out.append(b)
        if self.shuffle:
            rng.shuffle(out)
        return out

    def __iter__(self):
        yield from self._batches()

    def __len__(self):
        return len(self._batches())
