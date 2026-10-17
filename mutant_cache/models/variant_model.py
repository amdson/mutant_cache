"""
mutant_cache/models/variant_model.py
------------------------------------
The supervised head over the sparse variant cache (PLAN.md items 16, 18, 20).

Shape of the thing:

    variant tokens  ──(cross-attend cached reference states, every layer)──┐
    (position, alt residue, ref residue, [assay])                          │
                                                                           ▼
                      backbone.forward_variants(ref, positions, ids)  ->  h (N, K, d)
                                                                           │
            ┌──────────────────────────────────────────────────────────────┤
            ▼                                                              ▼
    additive branch: per-token scalar + ESM-C masked           interaction branch:
    marginal prior, summed over real substitutions             MLP over pooled variant
                                                               tokens ⊕ reference summary
                                                               ⊕ assay embedding
            └──────────────────────────► pred = b(assay) + s(assay)·(A + I)

Two design decisions worth stating, because the tests depend on them:

* **Exact additivity is a property of the head, not just the attention mask.**
  DATASETS.md defines the additive ablation as "variant-variant attention
  masked", but masking attention alone does not make the *prediction* additive:
  a pooled-then-MLP head mixes independently computed tokens and reintroduces
  interactions. So ``ablation="additive"`` both sets ``variant_attn_mask`` to
  the identity and drops the interaction branch, and
  ``test_models_ablations.py`` asserts numerically that
  ``f(A:B) - f(wt) == (f(A) - f(wt)) + (f(B) - f(wt))``.

* **The additive component is initialisable from ESM-C masked marginals.**
  With ``zero_init_additive`` the learned per-token scalar starts at exactly
  zero, so an untrained model predicts
  ``bias(assay) + scale(assay) · w · Σ_k LLR(pos_k, alt_k)`` — the masked-
  marginal zero-shot baseline, as the starting point rather than a separate
  model.

Everything is config switches on one class, per the task: four ablations, two
backends (cache / full recompute, same weights), two reference summaries.

Contract notes (INTERFACES.md §1), the assumptions this file leans on:
  - ``variant_extra_embed`` returns a *delta* that the backbone adds to the
    variant input embeddings ("...(N,K,d) -> (N,K,d) added to the variant input
    embeddings").  If the real backbone instead treats the hook as a transform,
    flip ``VariantModelConfig.extra_embed_returns_delta``.
  - ``forward_variants(ref, positions, variant_ids)`` is called positionally
    with exactly those three arguments; padded slots carry the *reference*
    residue and are excluded by our own ``var_mask`` (through the (N,K,K)
    attention mask and the pooling), so no ``var_mask=`` kwarg is required.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn

from mutant_cache.models.data_compat import parse_mutant

ABLATIONS = ("none", "additive", "no_cross_attn", "random_ref", "position_only")
BACKENDS = ("cache", "recompute")


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class VariantModelConfig:
    """Every knob that changes a number. ``ablation`` is a preset over the
    individual switches below it; set those directly to mix and match."""

    d_model: int = 0                    # filled from the backbone if 0
    n_assays: int = 1
    max_position: int = 2048
    vocab_size: int = 0                 # filled from the backbone if 0

    ablation: str = "none"
    backend: str = "cache"              # "cache" | "recompute" (same weights)

    # token composition
    use_position_embedding: bool = True
    use_alt_residue: bool = True
    use_ref_residue: bool = True
    assay_in_token: bool = False

    # head
    interaction: bool = True
    use_ref_summary: bool = True
    pool: str = "mean"                  # mean | sum | max
    ref_summary: str = "mean"           # mean | attn   (PLAN.md open question)
    head_hidden: int = 128
    head_layers: int = 2
    dropout: float = 0.0
    per_assay_affine: bool = True

    # additive branch
    additive_decompose: bool = True     # additive ablation: one forward per substitution
    use_marginals: bool = True
    marginal_weight_init: float = 1.0
    learn_marginal_weight: bool = True
    zero_init_additive: bool = True

    # backbone-side switches
    cross_attention: bool = True
    random_reference: bool = False
    random_ref_match_scale: bool = True
    random_ref_seed: int = 0
    position_only: bool = False
    position_only_token: str = "mask"   # mask | ref
    extra_embed_returns_delta: bool = True
    pass_var_mask: str = "auto"         # auto | always | never

    def __post_init__(self):
        if self.ablation not in ABLATIONS:
            raise ValueError(f"ablation must be one of {ABLATIONS}")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if self.ablation == "additive":
            # attention mask -> identity (set at forward time) AND no pooled
            # interaction term, or the head would smuggle epistasis back in.
            self.interaction = False
        elif self.ablation == "no_cross_attn":
            # The ablation's meaning is "variant tokens cannot read cached
            # reference states", and a tie is supposed to say "ESM-C context
            # contributes nothing".  That reading only survives if the *head*
            # is cut off from the reference too — leaving the ref summary in
            # would keep feeding ESM-C activations through a side door and
            # make the tie uninterpretable.
            self.cross_attention = False
            self.use_ref_summary = False
        elif self.ablation == "random_ref":
            self.random_reference = True
        elif self.ablation == "position_only":
            self.position_only = True
            self.use_alt_residue = False
            self.use_marginals = False      # the marginal LLR *is* alt identity

    def to_dict(self) -> dict:
        return asdict(self)


# ── batch ────────────────────────────────────────────────────────────────────

@dataclass
class VariantBatch:
    """N variants of one target, sharing one reference and one position set.

    That grouping is forced by INTERFACES.md §1: ``forward_variants`` takes a
    single ``(K,)`` position vector for the whole call. K is the *union* of the
    mutated positions in the group, and variants with fewer mutations are padded
    with the reference residue at the slots they do not mutate.

    **Padding is only free if the backbone takes ``var_mask``.**  §1 says
    variant queries attend to "reference keys at non-variant positions", and
    "non-variant" is defined by the shared ``positions`` vector — so a padded
    slot would withhold that reference key from *every* variant in the group,
    and a variant's score would depend on who it was batched with.  Passing
    ``var_mask`` fixes it completely: the backbone clamps padded slots to the
    reference residual at every layer and keeps them visible as reference
    keys, so each variant sees the whole reference regardless of the union.
    ``VariantEffectModel`` detects and uses it automatically.  For a backbone
    without it, ``exact_position_groups=True`` batches only variants that
    mutate exactly the same sites, which removes padding altogether at the
    cost of smaller (and more correlated) lists for the ranking loss.
    """
    target_id: str
    score_set_id: str
    ref_ids: torch.Tensor               # (L+2,) long
    positions: torch.Tensor             # (K,) long, token indices
    variant_ids: torch.Tensor           # (N, K) long
    var_mask: torch.Tensor              # (N, K) bool, True = real substitution
    assay_ids: torch.Tensor             # (N,) long
    n_mut: torch.Tensor                 # (N,) long
    target: torch.Tensor | None = None  # (N,) float, the label
    marginals: torch.Tensor | None = None   # (N, K) float, ESM-C LLR
    se: torch.Tensor | None = None      # (N,) float, reported uncertainty
    weight: torch.Tensor | None = None  # (N,) float, per-variant loss weight
    row_index: torch.Tensor | None = None   # (N,) long, back into the source frame
    ref: object | None = None           # ReferenceStates, filled/cached by the model

    @property
    def n(self) -> int:
        return int(self.variant_ids.shape[0])

    @property
    def k(self) -> int:
        return int(self.variant_ids.shape[1])

    def to(self, device) -> "VariantBatch":
        def mv(x):
            return x.to(device) if torch.is_tensor(x) else x
        return VariantBatch(
            target_id=self.target_id, score_set_id=self.score_set_id,
            ref_ids=mv(self.ref_ids), positions=mv(self.positions),
            variant_ids=mv(self.variant_ids), var_mask=mv(self.var_mask),
            assay_ids=mv(self.assay_ids), n_mut=mv(self.n_mut),
            target=mv(self.target), marginals=mv(self.marginals),
            se=mv(self.se), weight=mv(self.weight),
            row_index=mv(self.row_index), ref=self.ref)


def build_variant_batches(
    variants,
    sequences: dict[str, str],
    aa_to_id: dict[str, int],
    *,
    assay_index: dict[str, int] | None = None,
    batch_size: int = 32,
    marginals: dict[str, torch.Tensor] | None = None,
    target_col: str = "score_rank",
    se_col: str | None = "score_se",
    weight_col: str | None = None,
    bos_offset: int = 1,
    max_union_positions: int | None = None,
    exact_position_groups: bool = False,
    shuffle_seed: int | None = None,
) -> tuple[list[VariantBatch], dict[str, int]]:
    """Group a §2 ``variants`` frame into ``VariantBatch`` objects.

    Batches never cross a score set, so a within-score-set ranking loss over one
    batch is well defined by construction (see ``loss.ScoreSetBatchSampler`` for
    the index-level equivalent).  Within a score set, variants are ordered by
    their position tuple before chunking, which keeps the union K small —
    variants that touch the same sites land in the same call.

    Returns ``(batches, assay_index)``.
    """
    import numpy as np
    import pandas as pd

    if assay_index is None:
        assay_index = {s: i for i, s in
                       enumerate(sorted(pd.unique(variants["score_set_id"])))}

    rows = []
    for r in variants.itertuples(index=True):
        muts = parse_mutant(getattr(r, "mutant"))
        rows.append((r.Index, r.score_set_id, r.target_id,
                     tuple(p + bos_offset for p, _, _ in muts),
                     tuple(a for _, _, a in muts), muts))

    batches: list[VariantBatch] = []
    byset: dict[tuple[str, str], list] = {}
    for item in rows:
        byset.setdefault((item[1], item[2]), []).append(item)

    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    tvals = variants[target_col].to_numpy(dtype=float) if target_col in variants else None
    wvals = (variants[weight_col].to_numpy(dtype=float)
             if weight_col and weight_col in variants else None)
    svals = (variants[se_col].to_numpy(dtype=float)
             if se_col and se_col in variants else None)
    pos_of_row = {i: j for j, i in enumerate(variants.index)}

    for (ssid, tid), items in byset.items():
        seq = sequences[tid]
        items.sort(key=lambda it: (it[3], it[4]))
        chunks = []
        cur: list = []
        cur_pos: set[int] = set()
        for it in items:
            nxt = cur_pos | set(it[3])
            too_wide = (max_union_positions is not None
                        and len(nxt) > max_union_positions and cur)
            if exact_position_groups and cur and set(it[3]) != cur_pos:
                too_wide = True
            if cur and (len(cur) >= batch_size or too_wide):
                chunks.append(cur)
                cur, cur_pos = [], set()
                nxt = set(it[3])
            cur.append(it)
            cur_pos = nxt
        if cur:
            chunks.append(cur)

        ref_ids = torch.tensor(
            [0] + [aa_to_id[c] for c in seq] + [1], dtype=torch.long)

        for chunk in chunks:
            pos = sorted({p for it in chunk for p in it[3]})
            if not pos:                      # a wild-type-only chunk
                pos = [bos_offset]
            pidx = {p: j for j, p in enumerate(pos)}
            kk = len(pos)
            vids = ref_ids[torch.tensor(pos)][None].repeat(len(chunk), 1).clone()
            vmask = torch.zeros(len(chunk), kk, dtype=torch.bool)
            for i, it in enumerate(chunk):
                for (p, ref_aa, alt_aa) in it[5]:
                    j = pidx[p + bos_offset]
                    if seq[p - 1] != ref_aa:
                        raise ValueError(
                            f"{ssid}/{tid}: mutant says {ref_aa}{p} but sequence "
                            f"has {seq[p - 1]}{p}")
                    vids[i, j] = aa_to_id[alt_aa]
                    vmask[i, j] = True
            idx = [it[0] for it in chunk]
            rpos = torch.tensor([pos_of_row[i] for i in idx], dtype=torch.long)
            mg = None
            if marginals is not None and tid in marginals:
                M = marginals[tid]                       # (L+2, V)
                mg = M[torch.tensor(pos)[None, :], vids] * vmask
            batches.append(VariantBatch(
                target_id=tid, score_set_id=ssid, ref_ids=ref_ids,
                positions=torch.tensor(pos, dtype=torch.long),
                variant_ids=vids, var_mask=vmask,
                assay_ids=torch.full((len(chunk),), assay_index[ssid], dtype=torch.long),
                n_mut=vmask.sum(-1),
                target=(torch.tensor(tvals[rpos.numpy()], dtype=torch.float)
                        if tvals is not None else None),
                marginals=mg,
                se=(torch.tensor(svals[rpos.numpy()], dtype=torch.float)
                    if svals is not None else None),
                weight=(torch.tensor(wvals[rpos.numpy()], dtype=torch.float)
                        if wvals is not None else None),
                row_index=torch.tensor(idx, dtype=torch.long)))

    if rng is not None:
        order = rng.permutation(len(batches))
        batches = [batches[i] for i in order]
    return batches, assay_index


# ── model ────────────────────────────────────────────────────────────────────

@dataclass
class VariantPrediction:
    pred: torch.Tensor                  # (N,)
    additive: torch.Tensor              # (N,)
    interaction: torch.Tensor           # (N,)
    token_score: torch.Tensor           # (N, K)
    hidden: torch.Tensor                # (N, K, d)


def _mlp(d_in, d_hidden, n_layers, d_out, dropout):
    if n_layers <= 1:
        return nn.Sequential(nn.Linear(d_in, d_out))
    layers: list[nn.Module] = []
    d = d_in
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, d_hidden), nn.GELU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class VariantEffectModel(nn.Module):
    """Variant tokens over a cached reference, pooled into a scalar per variant.

    The backbone is held, not owned: it is whatever satisfies INTERFACES.md §1
    (``mutant_cache.backbone.load_esmc`` in production, ``tests.fixtures.
    StubESMC`` under test).  Freeze or train it from outside; this module only
    sets the three documented hooks for the duration of its own forward.
    """

    def __init__(self, backbone, cfg: VariantModelConfig):
        super().__init__()
        self.backbone = backbone
        d = cfg.d_model or int(getattr(backbone, "d_model"))
        v = cfg.vocab_size or int(getattr(backbone, "vocab_size",
                                          getattr(backbone, "n_vocab", 64)))
        cfg.d_model, cfg.vocab_size = d, v
        self.cfg = cfg

        self.pos_embed = nn.Embedding(cfg.max_position + 2, d)
        self.alt_embed = nn.Embedding(v, d)
        self.ref_embed = nn.Embedding(v, d)
        self.assay_embed = nn.Embedding(cfg.n_assays, d)
        self.extra_gain = nn.Parameter(torch.ones(()))

        self.additive_proj = nn.Linear(d, 1)
        self.marginal_weight = nn.Parameter(
            torch.tensor(float(cfg.marginal_weight_init)),
            requires_grad=cfg.learn_marginal_weight)

        self.ref_proj = nn.Linear(d, d)
        self.ref_query = nn.Parameter(torch.randn(d) / d ** 0.5)
        self.inter_mlp = _mlp(3 * d, cfg.head_hidden, cfg.head_layers, 1, cfg.dropout)

        self.assay_bias = nn.Embedding(cfg.n_assays, 1)
        self.assay_scale = nn.Embedding(cfg.n_assays, 1)

        self.reset_head()
        self._random_ref_cache: dict = {}
        self._ref_cache: dict = {}
        self._var_mask_ok = self._backbone_takes_var_mask()

    def _backbone_takes_var_mask(self) -> bool:
        """Does ``forward_variants`` accept ``var_mask=``?

        §1 mentions it only parenthetically ("pass var_mask to mark real
        substitutions if needed"), so it is treated as optional.  When it is
        there it is strictly better: the backbone clamps padded slots to the
        reference residual at every layer and exposes them as reference keys,
        which makes a variant's prediction independent of which other variants
        shared its forward pass.  Without it, padding withholds the reference
        key at every position any batch-mate mutates.
        """
        if self.cfg.pass_var_mask == "never":
            return False
        if self.cfg.pass_var_mask == "always":
            return True
        import inspect
        try:
            sig = inspect.signature(self.backbone.forward_variants)
        except (TypeError, ValueError):
            return False
        return "var_mask" in sig.parameters

    # -- initialisation -------------------------------------------------------
    def reset_head(self) -> None:
        nn.init.zeros_(self.assay_bias.weight)
        nn.init.ones_(self.assay_scale.weight)
        if self.cfg.zero_init_additive:
            nn.init.zeros_(self.additive_proj.weight)
            nn.init.zeros_(self.additive_proj.bias)
        last = [m for m in self.inter_mlp if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def init_additive_from_marginals(self, weight: float = 1.0) -> None:
        """PLAN.md 18: start the additive component at the ESM-C masked
        marginals.  With ``zero_init_additive`` this is already the state after
        ``reset_head``; call this to re-set the scale explicitly."""
        with torch.no_grad():
            self.marginal_weight.fill_(float(weight))
            nn.init.zeros_(self.additive_proj.weight)
            nn.init.zeros_(self.additive_proj.bias)

    def head_parameters(self):
        bb = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in bb]

    # -- reference ------------------------------------------------------------
    def reference_states(self, batch: VariantBatch, use_cache: bool = True):
        """``ReferenceStates`` for this batch's target, cached by target_id.

        PLAN.md 9 wants a disk-backed ``ReferenceCache`` keyed by (target_id,
        seq hash); this is the in-process version of the same idea, and the only
        reason the recompute control is a fair comparison (both backends pay for
        the reference exactly once per target).
        """
        if batch.ref is not None:
            return batch.ref
        key = (batch.target_id, int(batch.ref_ids.shape[0]))
        if use_cache and key in self._ref_cache:
            return self._ref_cache[key]
        ref = self.backbone.forward_reference(batch.ref_ids)
        if getattr(ref, "ids", None) is None:
            try:
                ref.ids = batch.ref_ids
            except Exception:
                pass
        if use_cache:
            self._ref_cache[key] = ref
        return ref

    def clear_reference_cache(self) -> None:
        self._ref_cache.clear()
        self._random_ref_cache.clear()

    def _randomize(self, ref, key: str):
        """The 'random reference states' ablation: ESM-C activations replaced by
        random vectors of matching shape.

        Drawn once per target and cached, deliberately.  Resampling every step
        would be noise injection, which tests something else entirely; a fixed
        random feature map is what isolates "the architecture, not the
        pretrained features".
        """
        if key in self._random_ref_cache:
            return self._random_ref_cache[key]
        g = torch.Generator(device="cpu")
        g.manual_seed(abs(hash((self.cfg.random_ref_seed, key))) % (2 ** 31))

        def rnd(t):
            z = torch.randn(t.shape, generator=g, dtype=torch.float32).to(
                device=t.device, dtype=t.dtype)
            if self.cfg.random_ref_match_scale:
                z = z * t.std().clamp_min(1e-6) + t.mean()
            return z

        # Copy rather than reconstruct: ReferenceStates carries fields this
        # module has no business knowing about (final_prenorm, a memoized kv
        # cache, ...) and §1 is free to add more.
        import copy
        out = copy.copy(ref)
        out.layer_inputs = [rnd(t) for t in ref.layer_inputs]
        out.final = rnd(ref.final)
        if getattr(ref, "final_prenorm", None) is not None:
            out.final_prenorm = rnd(ref.final_prenorm)
        if hasattr(out, "kv"):
            out.kv = None          # memoized from the real states; now stale
        self._random_ref_cache[key] = out
        return out

    def _ref_summary(self, ref) -> torch.Tensor:
        h = ref.final                                  # (1, L+2, d)
        body = h[:, 1:-1] if h.shape[1] > 2 else h
        if self.cfg.ref_summary == "attn":
            w = (body @ self.ref_query) / self.cfg.d_model ** 0.5
            s = (w.softmax(-1)[..., None] * body).sum(1)
        else:
            s = body.mean(1)
        return self.ref_proj(s)                        # (1, d)

    # -- hooks ----------------------------------------------------------------
    def _extra_embed(self, batch: VariantBatch, variant_ids=None, assay_ids=None):
        cfg = self.cfg
        positions = batch.positions.clamp(max=cfg.max_position + 1)
        ref_at = batch.ref_ids.to(batch.positions.device)[batch.positions]
        variant_ids = batch.variant_ids if variant_ids is None else variant_ids
        assay_ids = batch.assay_ids if assay_ids is None else assay_ids

        def hook(emb: torch.Tensor) -> torch.Tensor:
            d = torch.zeros_like(emb)
            if cfg.use_position_embedding:
                d = d + self.pos_embed(positions)[None].to(d.dtype)
            if cfg.use_alt_residue and not cfg.position_only:
                d = d + self.alt_embed(variant_ids).to(d.dtype)
            if cfg.use_ref_residue:
                d = d + self.ref_embed(ref_at)[None].to(d.dtype)
            if cfg.assay_in_token:
                d = d + self.assay_embed(assay_ids)[:, None].to(d.dtype)
            d = d * self.extra_gain
            return d if cfg.extra_embed_returns_delta else emb + d

        return hook

    def _attn_mask(self, batch: VariantBatch) -> torch.Tensor:
        """(N, K, K) bool, True = may attend.

        Two jobs: implement the additive ablation (identity), and in every mode
        stop a padded slot (a position some *other* variant in the group
        mutates) from acting as a key — otherwise a variant's prediction would
        depend on which variants it was batched with.
        """
        n, kk = batch.variant_ids.shape
        eye = torch.eye(kk, dtype=torch.bool, device=batch.variant_ids.device)
        if self.cfg.ablation == "additive":
            return eye[None].expand(n, -1, -1)
        return batch.var_mask[:, None, :] | eye[None]

    def _position_only_ids(self, batch: VariantBatch) -> torch.Tensor:
        """Drop alt-residue identity from what the *backbone* sees.

        Zeroing our own alt embedding is not enough: the backbone builds its own
        token embedding from ``variant_ids``.  Substituted slots become the mask
        token (falling back to the reference residue if the backbone exposes
        none), so position and the fact of substitution survive and the identity
        of the substitution does not.
        """
        ref_at = batch.ref_ids.to(batch.positions.device)[batch.positions]
        base = ref_at[None].expand(batch.n, -1).clone()
        if self.cfg.position_only_token == "mask":
            mid = getattr(self.backbone, "mask_token_id",
                          getattr(getattr(self.backbone, "tokenizer", None),
                                  "mask_id", None))
            if mid is not None:
                base = torch.where(batch.var_mask,
                                   torch.full_like(base, int(mid)), base)
        return base

    # -- forward --------------------------------------------------------------
    def _backbone_forward(self, ref, positions, var_ids, var_mask, attn_mask,
                          hook, backend):
        bb = self.backbone
        saved = (bb.variant_extra_embed, bb.variant_attn_mask,
                 bb.disable_cross_attention)
        try:
            bb.variant_extra_embed = hook
            bb.variant_attn_mask = attn_mask
            bb.disable_cross_attention = not self.cfg.cross_attention
            fn = (bb.forward_variants if backend == "cache"
                  else bb.forward_variants_bruteforce)
            return (fn(ref, positions, var_ids, var_mask=var_mask)
                    if self._var_mask_ok else fn(ref, positions, var_ids))
        finally:
            (bb.variant_extra_embed, bb.variant_attn_mask,
             bb.disable_cross_attention) = saved

    def _decompose(self, cfg) -> bool:
        return (cfg.ablation == "additive" and cfg.additive_decompose
                and self._var_mask_ok)

    def _forward_decomposed(self, batch, ref, var_ids, backend):
        """The additive ablation, evaluated one substitution at a time.

        Masking variant-variant attention is *not* by itself enough to make
        predictions additive, and the reason is subtle.  When B is mutated,
        position B's reference key is withheld from the cross-attention (§1
        withholds the reference at every position in the shared ``positions``
        vector) *and* B's variant key is blocked by the mask — so A's token
        loses position B entirely, and A's contribution still depends on
        whether B was co-mutated.  The absence carries information even though
        the identity does not.

        Expanding each variant into one row per substitution fixes it exactly.
        With ``var_mask``, every slot the row does not mutate is clamped to the
        reference residual and is visible as a reference key, so each row's
        context is the complete reference plus its own single variant token —
        identical no matter what else the parent variant mutated, and
        identical to what that substitution would see on its own.  Summing the
        per-substitution scalars is then additive by construction, which is
        what the ablation is supposed to mean.
        """
        n, kk = batch.variant_ids.shape
        dev = batch.variant_ids.device
        nz = batch.var_mask.nonzero(as_tuple=False)          # (M, 2) -> (row, slot)
        m = int(nz.shape[0])
        h = torch.zeros(n, kk, self.cfg.d_model, device=dev,
                        dtype=self.additive_proj.weight.dtype)
        token_score = torch.zeros(n, kk, device=dev, dtype=h.dtype)
        if m == 0:                                           # a wild-type-only batch
            return h, token_score

        rows, slots = nz[:, 0], nz[:, 1]
        ar = torch.arange(m, device=dev)
        ref_at = batch.ref_ids.to(dev)[batch.positions]
        exp_ids = ref_at[None].expand(m, kk).clone()
        exp_ids[ar, slots] = var_ids[rows, slots]
        exp_mask = torch.zeros(m, kk, dtype=torch.bool, device=dev)
        exp_mask[ar, slots] = True
        eye = torch.eye(kk, dtype=torch.bool, device=dev)[None].expand(m, -1, -1)

        vout = self._backbone_forward(
            ref, batch.positions, exp_ids, exp_mask, eye,
            self._extra_embed(batch, variant_ids=exp_ids,
                              assay_ids=batch.assay_ids[rows]), backend)
        hm = vout.hidden[ar, slots]                          # (M, d)
        s = self.additive_proj(hm).squeeze(-1)               # (M,)
        h = h.index_put((rows, slots), hm.to(h.dtype))
        token_score = token_score.index_put((rows, slots), s.to(h.dtype))
        return h, token_score

    def forward(self, batch: VariantBatch, *, backend: str | None = None,
                use_ref_cache: bool = True) -> VariantPrediction:
        cfg = self.cfg
        backend = backend or cfg.backend
        ref = self.reference_states(batch, use_cache=use_ref_cache)
        if cfg.random_reference:
            ref = self._randomize(ref, batch.target_id)

        var_ids = (self._position_only_ids(batch) if cfg.position_only
                   else batch.variant_ids)

        if self._decompose(cfg):
            h, token_score = self._forward_decomposed(batch, ref, var_ids, backend)
        else:
            vout = self._backbone_forward(
                ref, batch.positions, var_ids, batch.var_mask,
                self._attn_mask(batch), self._extra_embed(batch), backend)
            h = vout.hidden                               # (N, K, d)
            token_score = self.additive_proj(h).squeeze(-1) * batch.var_mask.to(h.dtype)

        mask = batch.var_mask.to(h.dtype)
        additive = token_score.sum(-1)
        if cfg.use_marginals and batch.marginals is not None:
            additive = additive + self.marginal_weight * (
                batch.marginals.to(h.dtype) * mask).sum(-1)

        if cfg.interaction:
            rs = (self._ref_summary(ref).to(h.dtype).expand(batch.n, -1)
                  if cfg.use_ref_summary
                  else h.new_zeros(batch.n, cfg.d_model))
            interaction = self.inter_mlp(torch.cat([
                self._pool(h, mask), rs,
                self.assay_embed(batch.assay_ids).to(h.dtype)], -1)).squeeze(-1)
        else:
            interaction = torch.zeros_like(additive)

        a = batch.assay_ids
        pred = (self.assay_bias(a).squeeze(-1).to(h.dtype)
                + self.assay_scale(a).squeeze(-1).to(h.dtype) * (additive + interaction))
        return VariantPrediction(pred=pred, additive=additive,
                                 interaction=interaction, token_score=token_score,
                                 hidden=h)

    def _pool(self, h, mask):
        denom = mask.sum(-1, keepdim=True).clamp_min(1.0)
        if self.cfg.pool == "sum":
            return (h * mask[..., None]).sum(1)
        if self.cfg.pool == "max":
            return (h.masked_fill(~mask.bool()[..., None],
                                  torch.finfo(h.dtype).min)).max(1).values
        return (h * mask[..., None]).sum(1) / denom

    @torch.no_grad()
    def predict(self, batches, backend: str | None = None):
        """(row_index, prediction) over a list of batches, as 1-D tensors."""
        self.eval()
        idx, out = [], []
        for b in batches:
            p = self.forward(b, backend=backend)
            out.append(p.pred.detach().float().cpu())
            idx.append(b.row_index if b.row_index is not None
                       else torch.arange(b.n))
        if not out:
            return torch.empty(0, dtype=torch.long), torch.empty(0)
        return torch.cat(idx), torch.cat(out)
