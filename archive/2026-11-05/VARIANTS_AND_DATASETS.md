# Model variants and datasets under consideration

Reference for what can be configured and what can be trained on. Companion to
PLAN.md (work list), INTERFACES.md (module contracts), DATASETS.md (sourcing and
licensing notes), HANDOFF.md (project state) and
`experiments/THERMOMPNN_SPLIT_RESULTS.md` (measured results).

Dataset counts below are **measured from
`$MUTANT_CACHE_SCRATCH/processed/`**, not quoted from papers.

---

# Part 1 — Model variants

## The base architecture

One reference sequence's per-layer residual-stream states are cached. A variant
is represented by K *variant tokens* (one per substituted position) carrying
position, alt-residue and ref-residue embeddings. At every layer a variant query
attends to {reference keys at non-variant positions} ∪ {variant keys}, the two
attentions merged by log-sum-exp so it is one softmax over the union. The
reference stays at batch 1 and is never recomputed per variant.

**The single approximation:** reference positions never attend to the variants,
so every non-variant token keeps its wild-type state. Everything else is exact.

The head has two branches:

- **additive** — a per-token scalar (`additive_proj`) summed over real
  substitutions, plus `marginal_weight ×` Σ ESM-C masked-marginal LLRs.
  With `zero_init_additive` (default) the learned part starts at exactly zero,
  so an untrained model *is* the masked-marginal zero-shot baseline.
- **interaction** — an MLP over (pooled variant tokens ⊕ reference summary ⊕
  assay embedding).

Then `pred = assay_bias(a) + assay_scale(a) · (additive + interaction)`.

Exact additivity is a property of the *head*, not just the attention mask:
masking variant-variant attention alone does not make the prediction additive,
because a pooled-then-MLP head remixes independently computed tokens. The
`additive` ablation therefore does both (see below).

## Axis 1 — backend (`--backend`, `--control-backend`)

| value | meaning |
|---|---|
| `cache` | the method: recompute only mutated positions against cached reference states |
| `recompute` | `forward_variants_bruteforce` — full-sequence layers with non-variant positions clamped to the reference. The correctness oracle |

`--control-backend` scores the *same weights* through the other backend, giving
both the accuracy delta and the throughput ratio (PLAN.md 22).
`--throughput-variants N` times N variants through each.

**Measured** (Tsuboyama doubles, trained head-only): accuracy delta
cache − recompute = **+0.0001 / +0.0004 / +0.0000** at depth 1 / 2 / all —
an order of magnitude below the ±0.005 initialisation-seed noise. The
approximation is accuracy-neutral in this regime.

Throughput is strongly configuration-dependent and non-monotonic in batch size,
because mean K (the *union* of mutated positions across a batch) grows with it:
0.69× at batch 32 (K=2.5), **2.26× at batch 128 (K=5.9)**, 1.06× at batch 512
(K=17.5). Phase 0's 12–32× figures are L=256–1024; at Tsuboyama's median 56 aa
its own grid predicts ~3.8×. `--max-union-positions` caps K independently of
batch size and is the untested lever here.

## Axis 2 — trainable surface

| variant | flag | trainable | reference cache | status |
|---|---|---|---|---|
| head only | (default) | 3.64M | valid, reused | **best measured** |
| + variant-only LoRA | `--lora-r R` | +3.46M at R=8 | valid — reference provably untouched | measured, does not help |
| + shared (dual) LoRA | `--shared-lora-r R` | +params on both streams | **invalidated every step**; recomputed with grad in train mode | one-seed pilot only |
| full fine-tuning | `--train-backbone` | +~300M | **BROKEN — see Known issues** | never run |

Variant LoRA (`attach_variant_lora`) applies to variant rows only, so the
reference stream is unchanged and the cache stays valid. `lora_B` is zero-init,
so a *fresh or frozen* adapter is exactly the base model — frozen-LoRA is
mathematically identical to head-only, and running it measures only
initialisation-seed variance. Targets are logical names `q,k,v,o,gate,up,down`
(default `q,o,gate,up,down`) because ESM-C fuses q/k/v and gate/up.
`--lora-lr` sets a separate adapter LR (unset ⇒ inherits `--lr`); `--lora-lr 0`
freezes them cleanly, since AdamW's decoupled weight decay also scales with lr.

Shared LoRA moves the wild-type context too, which costs the reference every
step (~1.46× in the pilot) but is the only cache-compatible analogue of full
fine-tuning, and its staleness handling is correct.

**Measured** (`thermompnn_mega_strict`, val per-assay ρ):

| | singles 1ep | singles 2ep | doubles 1ep ρ@1 | doubles 1ep ρ@2 |
|---|---|---|---|---|
| head-only | 0.7039 | **0.7280** | **0.7129** | **0.6440** |
| + variant LoRA r=8 | 0.7097 | 0.7197 | 0.6950 | 0.6304 |

Variant LoRA is **negative in three of four matched comparisons**. The +0.006
at singles-1ep is a convergence-rate artifact: at identical step 6894 it is
+0.006 ahead fully annealed and −0.049 behind half-annealed. At 2 epochs it
reaches *lower train loss* (0.3858 vs 0.3979) with *worse* val ρ — overfitting.
On pooled doubles it collapses to 0.4488 vs 0.5768, with a pooled-vs-per-assay
gap of −0.182 vs head-only's −0.067, i.e. it ranks well within domains but puts
domains on inconsistent ΔΔG scales.

Capacity is not the binding constraint — the reliability ceiling is ~1.0 with
~0.28 of headroom that 2× the parameters cannot touch. The likely limit is the
**239 training domains**: variants within a domain share one reference, so the
effective sample size for anything domain-general is nearer 239 than 215,731.
This is the main argument against `--train-backbone`.

## Axis 3 — ablations (`--ablation`)

Presets over individual switches; the individual switches can be set directly
to mix and match.

| value | what it changes | the question it answers |
|---|---|---|
| `none` | — | the full model |
| `additive` | `variant_attn_mask` → identity **and** `interaction=False`; with `additive_decompose` also one forward per substitution | does the interaction module beat exact additivity? **The gate (PLAN.md 27)** |
| `no_cross_attn` | variant queries see only variant keys, **and the head's reference summary is cut off** | does ESM-C context contribute anything? |
| `random_ref` | reference activations replaced by random vectors of matching shape, drawn once per target and cached | is the reference carrying information, or just shape? |
| `position_only` | drops alt-residue identity from what the backbone sees; also `use_alt_residue=False`, `use_marginals=False` | is this a per-position tolerance predictor? (the degenerate solution DATASETS.md names) |

`no_cross_attn` and `position_only` deliberately cut more than their names
suggest, so that a *tie* is interpretable: leaving the ref summary in would feed
ESM-C activations through a side door, and leaving the marginal LLRs in would
re-supply alt-residue identity.

`additive` is asserted numerically by `tests/test_models_ablations.py`:
`f(A:B) − f(wt) == (f(A) − f(wt)) + (f(B) − f(wt))`.

**Status:** `additive` on doubles is running now; the other three have not been
run on any real split. Parameter count is identical across ablations
(`inter_mlp` is constructed either way, just not called), so these compare
*function class*, not capacity.

## Axis 4 — head composition

| flag | options | default | note |
|---|---|---|---|
| `--ref-summary` | `mean` \| `attn` | `mean` | PLAN.md open question. Phase 0 says **do not mean-pool** — untested here |
| `--pool` | `mean` \| `sum` \| `max` | `mean` | pooling over variant tokens |
| `--head-hidden`, `--head-layers` | int | 128, 2 | interaction MLP size |
| `--dropout` | float | 0.0 | |
| `--no-ref-residue`, `--no-position-embedding` | flag | on | token composition |
| `--assay-in-token` | flag | off | adds the assay embedding to variant tokens *before layer 0*, so it passes through all 30 blocks and can reach attention |
| `--no-marginals` | flag | marginals on | removes the ESM-C zero-shot prior from the additive branch |
| `--marginal-weight`, `--freeze-marginal-weight` | float / flag | 1.0, learned | |
| `--marginal-mode` | `masked` \| `wt` | `masked` | |
| `--no-zero-init-additive` | flag | zero-init | with zero-init, an untrained model = the zero-shot baseline |
| `--position-only-token` | `mask` \| `ref` | `mask` | only for the `position_only` ablation |

## Axis 5 — loss (`--loss`, `+`-joined mixes allowed)

`rank`, `mse_rank`, `mse_dg`, `listnet`, `listmle`, `soft_spearman`.

`rank` (`pairwise_rank_loss`) is RankNet logistic:
`softplus(−σ·(ŷᵢ − ŷⱼ − margin))`, restricted to within-group pairs. **At the
defaults `σ=1, margin=0` this is exactly the Bradley-Terry loss** that holds the
published GB1 `3-vs-rest` record (0.880 ± 0.003 vs 0.814 ± 0.049 for MSE,
Brookes et al. NeurIPS 2024). Its argued advantage is invariance to monotone
transformations of measured fitness, which is directly relevant to the censoring
problem below — so PLAN.md 14's monotone link may be less load-bearing than
assumed.

`mse_dg` is ΔG MSE with optional `--inverse-variance` weighting and a `--se-col`
noise floor; used for all Tsuboyama runs so far, and it matches ThermoMPNN's
objective. `--target-col` picks `score_rank` (cross-assay; absorbs the link) or
`score` (a single ΔG assay).

`ScoreSetBatchSampler` guarantees batches never cross a score set — not a
nicety, since a mixed batch makes most pairs noise and lets the model lower the
loss by learning per-assay offsets instead of within-assay order.

## Axis 6 — per-assay parameters

Three per-score-set quantities, all load-bearing and all problematic on
across-protein splits:

- **`assay_embed`** (PLAN.md 18) — one learned d=960 vector per score set.
  Enters the `inter_mlp` concat always, and the variant tokens only under
  `--assay-in-token`. **In the default config it touches only the interaction
  branch**, so under `--ablation additive` it is inert apart from bias/scale.
- **`assay_bias` / `assay_scale`** — scalar affine per score set. Not in
  PLAN.md. Inert for within-assay Spearman (a positive affine cannot change a
  ranking) but matters for ΔG MSE.
- **`per_assay_affine`** — declared in `VariantModelConfig`, **read nowhere**;
  the affine is applied unconditionally. HANDOFF 6.5's "remove, or keep behind a
  flag defaulting off" is a two-line wiring job.

On a protein-level split every eval assay's embedding is an **untrained random
vector**, which is why the LoRA pilot's eval numbers were not what they looked
like. That regime needs a story — predict the embedding from `assay_type`
(already in the schema), or drop it — before its numbers mean anything. An
embedding degrades gracefully (zero / mean / predict from metadata); a per-assay
LoRA would not, which is the main argument for keeping per-assay capacity as an
embedding rather than adapters.

## Axis 7 — precision (`--precision`)

`bf16` (default) for experiments: frozen bf16 base, fp32 heads and adapters.
`fp32` for equivalence tests, oracles and sanity notebooks, where bf16 noise
would hide bugs. The variant path computes attention scores and softmax in fp32
regardless of model dtype.

---

# Part 2 — Datasets

## Processed and available now

### Tsuboyama Megascale — the primary training set

**550,141 variants / 412 score sets / 479 targets.** Lengths **32–74 aa**
(median 56). Depths: 2,022 WT, 389,068 singles, 159,051 doubles. Assay: `dG`.

Small single domains: 318 natural (PDB id), 90 Rocklin-style de novo designs
(`EEHEE_rd3_1498`, `HHH_rd1_0142` — names encode topology and design round),
40 pipe-prefixed, 20 trRosetta hallucinations, 10 `v2_` natural.

**What is measured is protease resistance, not stability.** K50 is the protease
concentration cleaving half the molecules; comparing it to a *predicted
unfolded-state* K50 gives ΔG. Done independently with trypsin (`_t`) and
chymotrypsin (`_c`), then combined, with ML global fits (`dG_ML`, `ddG_ML`).
`aa_seq_full` is `SAGG` + domain + `SAGG` (cDNA-display linkers); `aa_seq` — the
scored one — still carries an `NLYFQG` TEV remnant, and **mutation numbering is
1-based on `aa_seq` including that remnant** (for `3DKM.pdb`, `N1Q` is the N of
`NLYFQG`). Our ingest is verified byte-identical to `aa_seq`, with matching
mutant strings and `parse_mutant` positions.

`score` is ΔΔG kcal/mol, **destabilizing negative** (WT rows sit at 0.0045 ±
0.111). Note ThermoMPNN uses the opposite sign — `datasets.py` negates raw
`ddG_ML` — which is harmless for Spearman and silently wrong for RMSE.

Singles ΔΔG: mean −0.758, median −0.495, range −5.72 to +4.02, 77.8%
destabilizing. Assay dynamic range ~5 kcal/mol.

Supporting tables: `epsilon.parquet` (131,143 doubles with both singles
resolved, plus uncertainties and `pos_a`/`pos_b`/`seq_sep` across 559 site
pairs), `wt_report.parquet`.

**The measurement floor is per-domain, and it is the central data hazard.**
Absolute ΔG cannot be measured below ≈ **−1.0 kcal/mol** — a hard clamp,
identical across domains. So the measurable range is set by each domain's own
stability: ΔΔG_observed ≳ −(ΔG_wt + 1.0), and `corr(dG_wt, min_ddG) = −0.873`
over the 28 ThermoMPNN test domains. This is **left-censoring with a per-domain
threshold**, and it manufactures the "diminishing costs" epistasis the field
reports: any double whose additive prediction falls below its domain's bound is
clamped, producing positive ε in proportion. That is HANDOFF 6.2's
ε-vs-additive r = −0.725 and the published −0.9 kcal/mol mean coupling, same
cause. Consequence: **any epistasis claim on this dataset needs a saturation
control**, and the sharp version requires a `dG_ML` column that
`variants.parquet` does not yet carry.

### MaveDB — the breadth pool

**6,112,263 variants / 2,465 score sets / 2,465 targets.** Lengths **1–1,863 aa**
(median 56). Depths: 70,506 WT, 3,356,833 singles, 2,328,588 doubles, 190,391
triples, then a tail to depth 15 (88,640 / 43,059 / 19,593 / 8,354 / 3,796 /
1,469 / 648 / 240 / 86 / 46 / 10 / 4).

Assay types over score sets: binding 1,183, abundance 565, **null 535**,
growth 90, expression 74, fluorescence 18. Heterogeneous by design —
rank-normalize within score set and condition the head on assay identity.
Score-set sizes are healthy: median 1,122 variants, only 20 sets below 100,
1,521 at or above 1,000. All CC0.

**Use it for training breadth, not for evaluation.** Five reasons, all measured:

1. **No comparators.** MaveDB is a deposit repository, not a benchmark — no
   leaderboard, no standard splits, no published numbers on any partition.
   ProteinGym and CombinGym are the benchmark layer over this kind of data. It
   is also *upstream* of ProteinGym, so evaluating here leaks against the one
   benchmark you might want.
2. **The depth signal is concentrated, not distributed.** By score set:
   max_n_mut 1 → 1,781, 2 → 653, **≥3 → 21** (of 2,465). All 356,336 depth-≥3
   variants live in **21 score sets / 21 targets**, dominated by two
   fluorescence sets (`urn:mavedb:00000080-a-1/a-2`, depth 15) and one
   expression set (`00000044-b-1`, 156k variants). So "depth extrapolation on
   MaveDB" means ~21 landscapes with all the attendant per-landscape variance —
   and CombinGym curates exactly those properly, with 0/1/2/3-vs-rest splits and
   a leaderboard.
3. **Reliability is mostly uncomputable: 84% of variants have no `score_se`.**
   That is why `reliability` is null for all 2,465 sets and it is not fixable
   from this table. DATASETS.md adds that 30–52% of human deposits with a
   computable ceiling fall *below* 0.90, so you would be reporting correlations
   against unknown and often low ceilings — which DATASETS.md's own rule
   forbids.
4. **Unknown per-assay monotone links.** These are fitness assays, each with its
   own squashing between a latent additive trait and the reported score. Compute
   ε without fitting a link and a sigmoid shows up as learned epistasis. Worse
   than Tsuboyama's censoring, where the mechanism is known and the bound
   derivable.
5. **2,465 assay embeddings**, each untrained for held-out assays on an
   across-protein split (six times Tsuboyama's 412) — and 22% of score sets have
   null `assay_type`, so the predict-from-metadata fallback does not cover them.

**It is also not the long-protein corpus it looks like.** Median length is 56 —
identical to Tsuboyama — with p75 68 and p95 308. Only **175 targets exceed
200 aa and 56 exceed 500**. It remains the only processed source of long
proteins, which matters for any throughput claim (Phase 0's 12–32× lives at
L=256–1024), but as a 7% tail rather than the bulk.

### FLIP GB1 — the literature-comparable benchmark

**158,094 variants / 2 score sets / 1 target** (8,733 in FLIP's downsampled
subset via their `keep` column). Depths: 2 WT, 103 singles, 2,487 doubles,
28,585 triples, 126,917 quadruples — dense over 4 sites (V39/D40/G41/V54).

The only benchmark runnable against published numbers today. Published SCC:
ridge-on-one-hot 0.59 (`two_vs_rest`) / 0.76 (`three_vs_rest`); best pLM
ESM-2 48L 0.635 / 0.850; **SOTA is a from-scratch CNN with a Bradley-Terry loss
at 0.880 ± 0.003** on `three_vs_rest`. Note the SOTA is a ~2.2M-param one-hot
CNN with no pretraining, and on `three_vs_rest` all singles and doubles are
observed in training, so the pLM prior is largely redundant there; the regimes
where pretraining demonstrably helps are `one_vs_rest` and `two_vs_rest`.

**Two data issues, both unfixed — see Known issues.**

## Reference-only

### ProteinGym

`DMS_substitutions.csv` only: 217 assays, 2,465,767 variants, 696k singles +
1.77M multiples, multiples confined to 69 assays — **50 of which are
Tsuboyama-derived**. The 5.9 GB score files are not downloaded, and the
supervised CV fold assignments live in those files.

`exclusion/proteingym_overlap.parquet` carries the overlap. Tsuboyama score sets
with both singles and doubles, by exclusion strictness: none 153 sets
(149k singles / 159k doubles); exact matches only 103 (98k / 98k); exact + 30%
identity clusters 42 (43k / 39k). **Strict costs 70% of usable domains** but is
not optional if a ProteinGym number is ever wanted.

Current supervised leaderboard sits ~0.66–0.69 average Spearman (Kermut,
PoET-2 GP), but that is averaged over 217 mostly-single assays and is not a
Tsuboyama comparator. Avoid the *zero-shot* leaderboard entirely: placing a
supervised number beside VenusREM (0.523) is a category error.

## Not yet pulled

| dataset | what it adds | status |
|---|---|---|
| **CombinGym** | 14 combinatorial sets, 9 proteins, >400k points, 4–16 mutations, 0/1/2/3-vs-rest splits with a published leaderboard | `datasets/Makefile` has the target; **not cloned**. Fills the one real gap for depth extrapolation |
| ProteinGym score files | supervised CV folds, full assays | 5.9 GB, not downloaded |
| MGnify Stability | >1.8M sequences, >200k clusters, measured ΔG, domains <80 aa | optional. Diverse natural domains, *not* mutational scans — no coupling signal, but a large labelled corpus for pretraining the reference encoder if the variant branch underfits |
| Domainome 1 | — | PLAN.md 15, later |

## Splits available

| name | kind | contents |
|---|---|---|
| `protein_level` | across-protein | 1317/165/165 targets |
| `protein_level_clusters` | across-protein, mmseqs2 30% | — |
| `flip_gb1_two_vs_rest` | within-assay | 424 train (depths 0–2) / 8,309 test (3–4) |
| `thermompnn_mega` | across-protein | 249/31/30 targets; 225,549 train singles |
| `thermompnn_mega_strict` | across-protein | **239/31/28 targets** — matches ThermoMPNN's published counts; test is 28,172 singles, exactly what their loader scores |

`apply_split` handles both protein-level (`target_id, split`) and within-assay
(`score_set_id, mutant, split`) tables.

**Not built yet, and wanted:**

- **site-pair-disjoint** (HANDOFF 6.1) — train on doubles at some site pairs of
  a domain, test on unseen site pairs of the *same* domain. Keeps the assay
  embedding trained while removing the per-position-tolerance shortcut. The
  sharpest available epistasis test; `epsilon.parquet` already has
  `pos_a`/`pos_b`.
- FLIP GB1 `one_vs_rest`, `three_vs_rest`, `low_vs_high` — loader supports them,
  only `two_vs_rest` is exported. `three_vs_rest` (2,990 train, pure depth-4
  test) is the cleaner, less seed-sensitive depth test.
- ThermoMPNN's 5 CV folds — present in `mega_splits.pkl` (`cv_train_{0..4}` etc.,
  e.g. 188/59/51), unused.

## Evaluation regimes

Four setups, not interchangeable (HANDOFF 6.1):

| regime | split | comparators |
|---|---|---|
| zero-shot | none | ProteinGym main leaderboard — context row only |
| within-assay | variants of one protein | FLIP, ProteinGym supervised, CombinGym |
| across-protein | held-out proteins | ThermoMPNN, ESMRank. Breaks the assay embedding |
| pooled, variant-level | variants pooled over proteins | standard in ddG prediction |

**Metric convention is not a detail.** ThermoMPNN's headline 0.73 is *pooled*
variant-level (one `SpearmanCorrCoef`, `.update()` per mutation across all test
PDBs); our `eval/depth.py` `ALL` row is the *mean of per-score-set* Spearmans.
Measured gap on singles: −0.022 to −0.045 across six runs. On doubles it is
−0.067 (head-only) to −0.182 (LoRA), and pooling singles+doubles together
*flips* it positive (+0.022) because the model profits from easy between-depth
separation. **`ρ@all` pooled is a flattered number — quote depth-stratified
rows.** `experiments/pooled_vs_perassay_spearman.py` prints both.

## Reliability ceilings

`reliability` is **null for all 412 Tsuboyama and all 2,465 MaveDB score sets**,
and DATASETS.md requires every correlation reported against its ceiling.

Computed for the ThermoMPNN split via `attenuation_ceiling` =
`sqrt(1 − E[se²]/Var)`: **val singles 0.9979, test singles 0.9984** (mean se
0.056 against Var 0.86). The ceiling is ~1.0, so saturation explains nothing —
we are at ~0.70 with ~0.28 of headroom, and ThermoMPNN's 0.725 is nowhere near
a noise limit either. ε has a computed ceiling of **0.978**.

Caveat: `score_se` is per-variant ΔG *fitting* precision. The 2,022 WT replicate
rows have std **0.111**, a better reproducibility estimate than 0.056. The
honest ceiling is nearer 0.11.

---

# Known issues

1. **`--train-backbone` is silently wrong.** Reference-cache staleness is gated
   on shared LoRA only (`variant_model.py` `reference_states`). With a trainable
   backbone and no shared adapters the reference is computed once per domain and
   cached forever while the weights move, `train()` never invalidates it, and
   the cached path calls `forward_reference(ids)` without `grad=` (defaults to
   `no_grad`), so it is detached too — while the optimizer dutifully adds ~300M
   params. Fix: gate on `_shared_lora() or any(p.requires_grad for p in
   backbone.parameters())`. No test exercises this flag end to end.
2. **`onehot_ridge` silently vanishes on across-protein splits.**
   `OneHotRidgePerAssay` fits one ridge per `score_set_id`; with no training
   rows for eval score sets every prediction is NaN and the harness skips the
   block without a word. It should warn. (So "reproduce FLIP's ridge first" is
   right for within-assay GB1 and impossible on protein-level splits.)
3. **`per_assay_affine` is dead config** — declared, never read.
4. **FLIP GB1's reference is a 265-aa fusion construct**, not GB1:
   `FLIP_GB1_5LDE_1` is GB1's 56 residues, then `LEVLFQGP` (PreScission site),
   a fusion partner, and a C-terminal His6. FLIP's own splits carry the 56-aa
   sequence. Mutated sites still land correctly (residues 39/40/41/54 are
   V/D/G/V, since GB1 is the N-terminal 56) so there is no off-by-one, but the
   masked marginals, the additive-branch init and the mean-pooled `ref_summary`
   all see ~200 residues of irrelevant linker — and 4-of-265 is far sparser than
   4-of-56, which **flatters both the speedup and the depth-4 fidelity**.
5. **Our FLIP GB1 depth splits do not match FLIP's.** `low_vs_high` matches
   exactly (5,089/3,644); the depth splits do not — `one_vs_rest` 28 vs 29,
   `two_vs_rest` 424 vs 427, `three_vs_rest` 2,990 vs **2,968**. Fault is in
   depth-boundary assignment, not downsampling.
6. **`variants.parquet` carries no ΔG_wt / `dG_ML`**, which blocks the
   per-domain censoring control and a proper Tobit-style censored likelihood.
