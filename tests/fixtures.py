"""
tests/fixtures.py
-----------------
Stand-ins for the two layers that do not exist yet, built strictly to the
contracts in INTERFACES.md so that the model / loss / eval / baseline code
written against them keeps working when the real modules land.

Two things live here:

1. ``StubESMC`` — a ~30k-parameter transformer that honours INTERFACES.md §1:
   ``load_esmc``-shaped attributes, ``forward``, ``forward_reference``,
   ``forward_variants``, ``forward_variants_bruteforce``, and the three hooks
   the model layer needs (``variant_extra_embed``, ``variant_attn_mask``,
   ``disable_cross_attention``).  Its variant path implements the real cache
   semantics — variant queries attend to {reference keys at non-variant
   positions} ∪ {variant keys}, one softmax over the concatenation, which is
   the log-sum-exp merge of the two attentions — so a bruteforce equivalence
   test is meaningful here and not a tautology.

2. ``make_synthetic_dataset`` — parquet tables matching INTERFACES.md §2
   (``targets`` / ``variants`` / ``score_sets`` / ``splits/``), generated from a
   known additive + pairwise-coupling ground truth, so "the loss drops on a toy
   task with *real* epistasis" is a statement with teeth: the additive ablation
   provably cannot reach the ceiling.

Nothing here is imported by the library; it is test scaffolding only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AA = "ACDEFGHIKLMNPQRSTVWY"

# id layout: 0 bos, 1 eos, 2 pad, 3 mask, 4 X (unknown), 5..24 the 20 residues
BOS_ID, EOS_ID, PAD_ID, MASK_ID, X_ID = 0, 1, 2, 3, 4
AA_TO_ID = {a: 5 + i for i, a in enumerate(AA)}
ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}
VOCAB_SIZE = 5 + len(AA)


class StubTokenizer:
    """``tok.encode(seq) -> (L+2,) LongTensor`` including BOS/EOS, per §1."""

    bos_id = BOS_ID
    eos_id = EOS_ID
    pad_id = PAD_ID
    mask_id = MASK_ID
    aa_to_id = AA_TO_ID

    def encode(self, seq: str) -> torch.Tensor:
        ids = [BOS_ID] + [AA_TO_ID.get(c, X_ID) for c in seq.upper()] + [EOS_ID]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> str:
        return "".join(ID_TO_AA.get(int(i), "X") for i in ids
                       if int(i) not in (BOS_ID, EOS_ID, PAD_ID))


@dataclass
class ReferenceStates:
    """§1: per-layer residual-stream *inputs* for one reference sequence."""
    layer_inputs: list[torch.Tensor]        # n_layers x (1, L+2, d)
    final: torch.Tensor                     # (1, L+2, d)
    final_prenorm: torch.Tensor | None = None   # (1, L+2, d), for var_mask clamping
    ids: torch.Tensor | None = None         # (L+2,) convenience, not in §1
    key: str | None = None                  # target_id, for caching


@dataclass
class VariantOutput:
    hidden: torch.Tensor                    # (N, K, d)
    logits: torch.Tensor                    # (N, K, V)
    layer_hidden: list[torch.Tensor] | None = None


@dataclass
class BackboneOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    layer_inputs: list[torch.Tensor] = field(default_factory=list)


def _mha(q, k, v, n_heads, mask=None):
    """q (N,Q,d), k/v (N,S,d), mask (N,Q,S) bool — True = may attend."""
    n, qlen, d = q.shape
    slen = k.shape[1]
    hd = d // n_heads
    q = q.view(n, qlen, n_heads, hd).transpose(1, 2)
    k = k.view(n, slen, n_heads, hd).transpose(1, 2)
    v = v.view(n, slen, n_heads, hd).transpose(1, 2)
    att = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
    if mask is not None:
        att = att.masked_fill(~mask[:, None], torch.finfo(att.dtype).min)
    att = att.softmax(-1)
    out = att @ v
    return out.transpose(1, 2).reshape(n, qlen, d)


class StubLayer(nn.Module):
    """Pre-norm transformer block whose projections are shared between the
    reference stream and the variant stream — as they are in the real cache."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d) for _ in range(4))
        self.ln2 = nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, 2 * d), nn.Linear(2 * d, d)

    def _ffn(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))

    def kv(self, x):
        h = self.ln1(x)
        return self.wk(h), self.wv(h)

    def forward(self, x, mask=None):
        h = self.ln1(x)
        x = x + self.wo(_mha(self.wq(h), self.wk(h), self.wv(h), self.n_heads, mask))
        return self._ffn(x)

    def forward_variant(self, xv, ref_k, ref_v, var_mask, use_cross):
        """xv (N,K,d); ref_k/ref_v (1,S,d) already restricted to non-variant
        positions; var_mask (N,K,K) bool over the variant block."""
        n, kk, _ = xv.shape
        h = self.ln1(xv)
        q, k, v = self.wq(h), self.wk(h), self.wv(h)
        if use_cross and ref_k.shape[1] > 0:
            rk = ref_k.expand(n, -1, -1)
            rv = ref_v.expand(n, -1, -1)
            keys = torch.cat([rk, k], 1)
            vals = torch.cat([rv, v], 1)
            cross = torch.ones(n, kk, rk.shape[1], dtype=torch.bool, device=xv.device)
            mask = torch.cat([cross, var_mask], -1)
        else:
            keys, vals, mask = k, v, var_mask
        xv = xv + self.wo(_mha(q, keys, vals, self.n_heads, mask))
        return self._ffn(xv)


class StubESMC(nn.Module):
    """A tiny stand-in for ``mutant_cache.backbone.load_esmc(...)``."""

    def __init__(self, d_model=32, n_layers=2, n_heads=4, max_len=512, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.d_model, self.n_layers, self.n_heads = d_model, n_layers, n_heads
        self.vocab_size, self.max_len = VOCAB_SIZE, max_len
        self.aa_to_id = dict(AA_TO_ID)
        self.tokenizer = StubTokenizer()
        self.mask_token_id = MASK_ID
        self.tok_embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList(StubLayer(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, VOCAB_SIZE)
        # §1 hooks, set by the model layer
        self.variant_extra_embed = None
        self.variant_attn_mask = None
        self.disable_cross_attention = False

    # -- embeddings -----------------------------------------------------------
    def _embed(self, ids, positions=None):
        if positions is None:
            positions = torch.arange(ids.shape[-1], device=ids.device)
        return self.tok_embed(ids) + self.pos_embed(positions.clamp(max=self.max_len - 1))

    def _apply_extra(self, emb):
        """§1: the hook's output is *added to* the variant input embeddings."""
        if self.variant_extra_embed is None:
            return emb
        return emb + self.variant_extra_embed(emb)

    # -- §1 forward -----------------------------------------------------------
    def forward(self, ids):
        if ids.dim() == 1:
            ids = ids[None]
        x = self._embed(ids)
        layer_inputs = []
        for layer in self.layers:
            layer_inputs.append(x)
            x = layer(x)
        hidden = self.ln_f(x)
        return BackboneOutput(logits=self.lm_head(hidden), hidden=hidden,
                              layer_inputs=layer_inputs + [x])

    def forward_reference(self, ids, key: str | None = None,
                          grad: bool = False) -> ReferenceStates:
        # ``grad`` as in the real backbone: keep the graph when the base (or a
        # shared adapter) is trainable, so the reference carries gradient
        with (torch.enable_grad() if grad else torch.no_grad()):
            out = self.forward(ids)
        # layer_inputs[-1] from forward() is the post-last-layer residual; §1's
        # list is the per-layer *inputs*, so keep n_layers of them and park the
        # trailing one as final_prenorm (what var_mask clamping needs).
        det = (lambda t: t) if grad else (lambda t: t.detach())
        li = [det(t) for t in out.layer_inputs]
        return ReferenceStates(layer_inputs=li[:-1], final=det(out.hidden),
                               final_prenorm=li[-1],
                               ids=ids if ids.dim() == 1 else ids[0], key=key)

    def _slot_key_mask(self, var_mask, n, kk, device):
        """(N, K, K) bool. Combines ``variant_attn_mask`` with the padded-slot
        rule: a padded slot holds the reference state, so it is visible exactly
        when cross-attention is."""
        sm = self.variant_attn_mask
        if sm is not None:
            sm = torch.as_tensor(sm, device=device).bool()
            if sm.dim() == 2:
                sm = sm[None]
        if var_mask is not None:
            base = sm if sm is not None else torch.ones(
                1, kk, kk, dtype=torch.bool, device=device)
            ref_ok = torch.tensor(not self.disable_cross_attention, device=device)
            sm = torch.where(var_mask[:, None, :], base, ref_ok)
        if sm is None:
            sm = torch.ones(1, kk, kk, dtype=torch.bool, device=device)
        return sm.expand(n, -1, -1) if sm.shape[0] == 1 else sm

    @staticmethod
    def _clamp(x, ref_layer, positions, var_mask):
        """Padded slots take the reference residual at their position."""
        if var_mask is None:
            return x
        return torch.where(var_mask[..., None], x,
                           ref_layer[0, positions].to(x.dtype))

    def forward_variants(self, ref: ReferenceStates, positions, variant_ids,
                         var_mask=None):
        """N variants sharing one reference and one set of K token positions.

        With ``var_mask``, padded slots are clamped to the reference residual
        at every layer, so a variant's output does not depend on which other
        variants shared the call.
        """
        positions = positions.to(variant_ids.device).long()
        n, kk = variant_ids.shape
        seq_len = ref.final.shape[1]
        keep = torch.ones(seq_len, dtype=torch.bool, device=variant_ids.device)
        keep[positions] = False                      # reference keys at NON-variant positions
        slot = self._slot_key_mask(var_mask, n, kk, variant_ids.device)
        use_cross = not self.disable_cross_attention

        x = self._apply_extra(self._embed(variant_ids, positions[None]))
        x = self._clamp(x, ref.layer_inputs[0], positions, var_mask)
        layer_hidden = []
        for li, layer in enumerate(self.layers):
            layer_hidden.append(x)
            rk, rv = layer.kv(ref.layer_inputs[li])
            x = layer.forward_variant(x, rk[:, keep], rv[:, keep], slot, use_cross)
            nxt = (ref.layer_inputs[li + 1] if li + 1 < self.n_layers
                   else ref.final_prenorm)
            if nxt is not None:
                x = self._clamp(x, nxt, positions, var_mask)
        hidden = self._clamp(self.ln_f(x), ref.final, positions, var_mask)
        return VariantOutput(hidden=hidden, logits=self.lm_head(hidden),
                             layer_hidden=layer_hidden)

    def forward_variants_bruteforce(self, ref: ReferenceStates, positions,
                                    variant_ids, var_mask=None):
        """The oracle: full-sequence layers with non-variant positions clamped to
        the reference residual stream at every layer input."""
        positions = positions.to(variant_ids.device).long()
        n, kk = variant_ids.shape
        seq_len = ref.final.shape[1]
        assert ref.ids is not None, "bruteforce needs the reference token ids"
        full = ref.ids.to(variant_ids.device)[None].repeat(n, 1)
        full[:, positions] = variant_ids
        x = self._embed(full)
        extra = self._apply_extra(self._embed(variant_ids, positions[None]))
        x[:, positions] = self._clamp(extra, ref.layer_inputs[0], positions, var_mask)
        slot = self._slot_key_mask(var_mask, n, kk, variant_ids.device)
        use_cross = not self.disable_cross_attention

        # (S,S) bool: who may attend to whom, in full-sequence coordinates
        pidx = positions
        other = torch.ones(seq_len, dtype=torch.bool, device=x.device)
        other[pidx] = False
        base = torch.ones(n, seq_len, seq_len, dtype=torch.bool, device=x.device)
        if not use_cross:
            base[:, pidx, :] = False
        base[:, pidx[:, None], pidx[None, :]] = slot
        for li, layer in enumerate(self.layers):
            x = x.clone()
            x[:, other] = ref.layer_inputs[li][0][other]
            x[:, pidx] = self._clamp(x[:, pidx], ref.layer_inputs[li],
                                     positions, var_mask)
            x = layer(x, base)
        hidden = self.ln_f(x)[:, positions]
        hidden = self._clamp(hidden, ref.final, positions, var_mask)
        return VariantOutput(hidden=hidden, logits=self.lm_head(hidden))


def make_stub_backbone(d_model=32, n_layers=2, n_heads=4, seed=0) -> StubESMC:
    m = StubESMC(d_model=d_model, n_layers=n_layers, n_heads=n_heads, seed=seed)
    m.eval()
    return m


# ── synthetic data (INTERFACES.md §2) ────────────────────────────────────────

@dataclass
class SyntheticTruth:
    """The generative model behind the synthetic tables, kept so a test can ask
    how much of the achievable signal a model recovered."""
    sequences: dict[str, str]
    additive: dict[str, np.ndarray]         # (L, 20)
    coupling: dict[str, np.ndarray]         # (L, L) symmetric, zero diagonal
    residue_effect: np.ndarray              # (20,)
    epistasis_scale: float

    def score(self, target_id: str, muts: list[tuple[int, str, str]]) -> float:
        a, c, r = (self.additive[target_id], self.coupling[target_id],
                   self.residue_effect)
        idx = [(p - 1, AA.index(alt)) for p, _, alt in muts]
        tot = sum(a[p, j] for p, j in idx)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                pi, ai = idx[i]
                pj, aj = idx[j]
                tot += self.epistasis_scale * c[pi, pj] * r[ai] * r[aj]
        return float(tot)


def make_synthetic_dataset(
    out_dir: str | Path,
    *,
    n_targets: int = 3,
    seq_len: int = 24,
    n_score_sets_per_target: int = 1,
    n_singles: int = 120,
    n_doubles: int = 160,
    n_triples: int = 60,
    epistasis_scale: float = 1.0,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[Path, SyntheticTruth]:
    """Write §2 parquet tables under ``out_dir`` and return (dir, ground truth).

    Score sets alternate assay types: even-indexed sets report a dG-like score
    (identity link, so `score` is directly the latent trait and the Megascale
    ε construction is exact), odd-indexed sets a fitness-like score squashed
    through tanh (so rank-normalization is the only honest target).
    """
    import pandas as pd

    out_dir = Path(out_dir)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    residue_effect = rng.normal(0, 1, len(AA))
    seqs, additive, coupling = {}, {}, {}
    target_rows, ss_rows, var_rows = [], [], []

    for t in range(n_targets):
        tid = f"TGT{t:02d}"
        seqs[tid] = "".join(rng.choice(list(AA), seq_len))
        additive[tid] = rng.normal(0, 1, (seq_len, len(AA)))
        c = rng.normal(0, 1, (seq_len, seq_len))
        c = np.triu(c, 1)
        c = c + c.T
        # sparse couplings: only ~25% of site pairs interact at all
        c *= (rng.random((seq_len, seq_len)) < 0.25)
        c = np.triu(c, 1)
        coupling[tid] = c + c.T
        target_rows.append({"target_id": tid, "sequence": seqs[tid],
                            "uniprot": None, "source": "synthetic"})

    truth = SyntheticTruth(seqs, additive, coupling, residue_effect, epistasis_scale)

    def sample_mut(tid, n):
        pos = rng.choice(seq_len, size=n, replace=False) + 1
        out = []
        for p in sorted(pos):
            ref = seqs[tid][p - 1]
            alt = rng.choice([a for a in AA if a != ref])
            out.append((int(p), ref, str(alt)))
        return out

    ss_i = 0
    for t in range(n_targets):
        tid = f"TGT{t:02d}"
        for s in range(n_score_sets_per_target):
            ssid = f"SS{ss_i:02d}"
            is_dg = (ss_i % 2 == 0)
            gain = float(rng.uniform(0.7, 1.4)) * (1.0 if is_dg else -1.0)
            reliability = float(rng.uniform(0.75, 0.97))
            rows, seen = [], set()
            for depth, count in ((1, n_singles), (2, n_doubles), (3, n_triples)):
                tries = 0
                made = 0
                while made < count and tries < 20 * count:
                    tries += 1
                    muts = sample_mut(tid, depth)
                    key = tuple(muts)
                    if key in seen:
                        continue
                    seen.add(key)
                    made += 1
                    latent = truth.score(tid, muts)
                    val = gain * latent + rng.normal(0, noise)
                    score = val if is_dg else math.tanh(val / 3.0)
                    rows.append({
                        "score_set_id": ssid, "target_id": tid,
                        "mutant": ":".join(f"{r}{p}{a}" for p, r, a in muts),
                        "n_mut": depth, "score": float(score),
                        "score_se": float(abs(rng.normal(0, 0.05)) + 0.01),
                        "assay_type": "dG" if is_dg else "fitness",
                    })
            r = pd.DataFrame(rows)
            r["score_rank"] = (r["score"].rank(method="average") - 0.5) / len(r)
            var_rows.append(r)
            ss_rows.append({"score_set_id": ssid, "target_id": tid,
                            "n_variants": len(r), "max_n_mut": int(r["n_mut"].max()),
                            "assay_type": "dG" if is_dg else "fitness",
                            "reliability": reliability, "license": "CC0"})
            ss_i += 1

    variants = pd.concat(var_rows, ignore_index=True)
    targets = pd.DataFrame(target_rows)
    score_sets = pd.DataFrame(ss_rows)

    # protein-level split: everything trains, the last two targets become
    # val/test when there are enough targets to spare them.
    labels = ["train"] * n_targets
    if n_targets >= 3:
        labels[-2], labels[-1] = "val", "test"
    elif n_targets == 2:
        labels[-1] = "test"
    split = pd.DataFrame({"target_id": targets["target_id"], "split": labels})
    # depth split: train on singles, test on 2+ — the Phase 6 gate, in miniature
    depth_split = variants[["score_set_id", "mutant", "n_mut"]].copy()
    depth_split["split"] = np.where(depth_split["n_mut"] == 1, "train", "test")
    val_mask = (depth_split["split"] == "train") & (rng.random(len(depth_split)) < 0.15)
    depth_split.loc[val_mask, "split"] = "val"
    depth_split = depth_split.drop(columns=["n_mut"])

    targets.to_parquet(out_dir / "targets.parquet", index=False)
    variants.to_parquet(out_dir / "variants.parquet", index=False)
    score_sets.to_parquet(out_dir / "score_sets.parquet", index=False)
    split.to_parquet(out_dir / "splits" / "protein.parquet", index=False)
    depth_split.to_parquet(out_dir / "splits" / "depth.parquet", index=False)
    return out_dir, truth
