# Interfaces between modules

These contracts let the backbone, data, and model/eval layers be built in
parallel. Change one only by updating this file and every caller.

Conventions: Python 3, torch, env `mutant_cache` (`~/anaconda3/envs/mutant_cache`).
Raw data and caches live under `$MUTANT_CACHE_SCRATCH` (default `~/mutant_cache_scratch`),
never in the repo.

## 1. Backbone + variant cache (`mutant_cache/backbone`, `mutant_cache/cache`)

```python
from mutant_cache.backbone import load_esmc            # local, hackable ESM-C
model = load_esmc("esmc_300m", device="cuda", dtype=torch.bfloat16)
tok = model.tokenizer                                   # or mutant_cache.backbone.tokenize
ids = tok.encode("MKTAYIAK...")                         # (L+2,) incl. BOS/EOS, LongTensor
model.d_model, model.n_layers, model.n_heads            # ints
model.aa_to_id: dict[str, int]                          # 20 canonical residues -> token id

# Full forward (the exact oracle)
out = model(ids[None])                  # out.logits (B, L+2, V), out.hidden (B, L+2, d), out.layer_inputs: list[(B, L+2, d)]

# Reference cache: per-layer residual-stream inputs for one reference sequence
ref = model.forward_reference(ids)      # ReferenceStates: .layer_inputs list[n_layers] of (1, L+2, d), .final (1, L+2, d)

# Variant forward: N variants sharing one reference, each substituting the
# same set of K token positions (pad with reference residues where a variant
# has fewer mutations; pass var_mask to mark real substitutions if needed).
vout = model.forward_variants(
    ref,                                 # ReferenceStates (batch 1, frozen)
    positions,                           # (K,) LongTensor, token indices into ids: residue p (1-based) is token p, since BOS is token 0
    variant_ids,                         # (N, K) LongTensor token ids at those positions
)
vout.hidden      # (N, K, d) final-layer hidden at variant positions
vout.logits      # (N, K, V)
vout.layer_hidden  # optional list per layer

model.forward_variants_bruteforce(...)  # same signature; full-seq layers with non-variant
                                        # positions clamped to the reference — correctness oracle
```

`forward_variants` must support gradients flowing into the variant path and
hooks for variant-only LoRA (see `CARBON_modules/variant_lora_layers.py` in
~/svar for the pattern). Reference positions never attend to variants: that is
the approximation.

Hooks the model layer needs (`mutant_cache/models`):
- `model.variant_extra_embed`: optional callable `(N, K, d) -> (N, K, d)` added
  to the variant input embeddings before layer 0 (for assay/position embeddings).
- `model.variant_attn_mask`: optional bool `(K, K)` or `(N, K, K)`; False blocks
  variant->variant attention (the "additive" ablation sets this to identity).
- `model.disable_cross_attention`: bool; variant queries see only variant keys
  (the "no cross-attention" ablation).

## 2. Canonical data tables (`mutant_cache/data`)

Every dataset loader writes parquet under `$MUTANT_CACHE_SCRATCH/processed/<dataset>/`:

`targets.parquet` — one row per reference sequence
| column | type | note |
|---|---|---|
| target_id | str | stable id, e.g. MaveDB target urn or Tsuboyama domain name |
| sequence | str | amino-acid reference, uppercase, 20 canonical letters (+X allowed) |
| uniprot | str or null | |
| source | str | mavedb / tsuboyama / flip_gb1 / proteingym / combingym |

`variants.parquet` — one row per measured variant
| column | type | note |
|---|---|---|
| score_set_id | str | assay identity (one per MaveDB score set / Tsuboyama dataset) |
| target_id | str | FK to targets |
| mutant | str | colon-joined, 1-based on target sequence: `A23G:K45R`; `""` for WT |
| n_mut | int | number of substitutions |
| score | float | raw reported score |
| score_rank | float | rank-normalized within score_set_id to (0,1) |
| score_se | float or null | reported uncertainty if any |
| assay_type | str or null | abundance / binding / growth / dG / fitness ... |

`score_sets.parquet` — per score set: score_set_id, target_id, n_variants,
max_n_mut, assay_type, reliability (float or null), license.

`splits/<split_name>.parquet` — columns `target_id, split` in {train,val,test}
(protein-level split) or `score_set_id, mutant, split` for within-assay splits
like FLIP `2-vs-rest`.

`exclusion/proteingym_overlap.parquet` — target_ids to exclude from training.

Helper used by the model layer:
```python
from mutant_cache.data import load_variants, parse_mutant
parse_mutant("A23G:K45R") -> [(23, "A", "G"), (45, "K", "R")]   # 1-based
```

## 3. Eval (`mutant_cache/eval`)
`depth_spearman(df, pred_col, score_col) -> DataFrame` with per score set
Spearman at depth buckets 1/2/3/4/5+ and overall; `reliability` column carried
through where known.

## Deviations / clarifications (section 1, as implemented)

Nothing in section 1 was impossible for ESM-C; these are additions and
pinned-down semantics (`mutant_cache/backbone/esmc.py`, `mutant_cache/cache/`):

- `forward_variants(ref, positions, variant_ids, var_mask=None, output_logits=True, output_layers=False)`;
  `vout.layer_hidden` is filled only with `output_layers=True` (per-block outputs, pre final norm).
  `vout.hidden` is post final norm (same space as `out.hidden`).
- `ReferenceStates` also carries `ids`, `final_prenorm`, and lazily memoized per-layer
  reference `kv` (normed + rotated K, V). `ref.final` is post final norm.
  `forward_reference(ids, grad=False)` runs under `no_grad` by default.
- `var_mask[n, j] = False` makes slot j *exactly* a non-variant position for variant n
  (clamped to the reference every layer, no adapters, visible as a reference key, outputs = reference).
- `variant_extra_embed` is called with the variant token embeddings `(N, K, d)`; its output is added.
- With `var_mask`, padded slots are reference keys: hidden by `disable_cross_attention`,
  not affected by `variant_attn_mask`.
- Variant-only LoRA (`mutant_cache.cache.lora.attach_variant_lora`): ESM-C fuses q/k/v and gate/up,
  so targets are logical names `q,k,v,o,gate,up,down` (default `q,o,gate,up,down`); adapters live in
  `model.variant_adapters`. `forward_variants_bruteforce` honours adapters (applied on variant rows only).
- Shared LoRA (`attach_shared_lora`): `model.shared_adapters`, a low-rank change to the base applied
  everywhere — `ESMC.forward` (so `forward_reference`), the reference K/V, and all variant rows. Without
  it `ESMC.forward` takes the stock path (official-parity). The reference then depends on trainable
  weights: `VariantEffectModel` recomputes it with grad in train mode and drops its cache on every
  train/eval switch; `base_fingerprint` covers shared adapters.
- The variant path computes attention scores/softmax in fp32 regardless of model dtype.
