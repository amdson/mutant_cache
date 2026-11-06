# Handoff: mutant_cache, as of 2026-10-21

State of the project before the first major training run. Companion to PLAN.md
(the work list), INTERFACES.md (module contracts), DATASETS.md (data notes) and
experiments/PHASE0_RESULTS.md (the go/no-go numbers). Read CLAUDE.md first for
conventions.

HEAD is `4037d0d`, working tree clean, 177 tests passing.

---

## 1. What the project is

Cache one reference protein's per-layer ESM-C states, then recompute **only the
mutated positions** for thousands of mutants. A variant query at position p
attends to {reference keys at non-variant positions} ∪ {variant keys}, merged by
log-sum-exp so it is one softmax over the union. The reference stays at batch 1.

The single approximation: reference positions never attend to the variants, so
every non-variant token keeps its wild-type state. Everything else is exact.

Phase 0 verdict was **GO**: 12–32× faster than full re-forwarding at realistic
mutation depths, reproducing exact masked-marginal rankings at Spearman
0.92–0.998. It degrades with mutation *density*, not mutation count — short
proteins are the binding constraint, not deep mutants.

---

## 2. What is built and verified

| component | state | evidence |
|---|---|---|
| ESM-C 300M port (`backbone/`) | done | bit-identical to official `esm` in fp32 (`tests/test_backbone_equivalence.py`, `notebooks/01`) |
| Variant cache (`cache/variant_cache.py`) | done | matches bruteforce oracle ~1e-6 fp32; exact when mutations = wild type (`notebooks/02`) |
| Variant-only LoRA | done + wired | `--lora-r`; reference provably untouched |
| Shared (dual) LoRA | done + wired | `--shared-lora-r`; applies to both streams |
| Reference cache | done | disk-backed, fingerprinted (shared adapters included) |
| Data layer (`data/`) | done for 3 datasets | see §4 |
| Harness (`harness.py`) | done | one shared loop, ~60 flags, end-to-end smoke test |
| Model + losses + eval + baselines | done | ablations, depth-stratified Spearman, epsilon |

**Precision**: frozen bf16 base, fp32 heads and adapters. `harness --precision`
defaults to bf16; tests/oracles/notebooks stay fp32 so bf16 noise cannot hide
bugs.

---

## 3. Results so far

All on Tsuboyama unless noted. None of these are publishable numbers; they are
plumbing checks and a single-seed pilot.

**Zero-shot (ESM-C masked marginals)**, protein-level val, 17 domains, 25,675
variants: singles 0.343, doubles 0.293, all 0.368.

**LoRA pilot** (`experiments/lora_pilot.sh`), 20k train variants, 1 epoch,
protein-level split, one seed:

| config | singles | doubles | all | interaction | wall |
|---|---|---|---|---|---|
| head only | 0.625 | 0.434 | 0.631 | 0.01 | 4.6 min |
| + variant LoRA r=8 | 0.668 | 0.501 | 0.671 | 0.09 | 6.7 min |
| + variant + shared r=8 | 0.681 | 0.497 | 0.688 | 0.02 | 9.7 min |

Reading: training works and roughly doubles zero-shot. LoRA adds ~0.04 overall.
Shared adds ~0.017 over variant-only, which is inside single-seed noise. **No
config learns epistasis** (interaction correlation 0.01–0.09) — on doubles these
models are essentially summing the two singles. All three were still improving
at their final step.

**Two caveats on the pilot.** (a) It used a protein-level split, so every eval
assay's `assay_embed` was an untrained random vector at evaluation time — see
§6.5. (b) It mixed singles and doubles in training and tested on held-out
*domains*, so it measured cross-protein transfer, not depth extrapolation.

**Overfit-10 sanity check** (`experiments/overfit10_*.py`): the model memorizes
10 variants of one domain to MSE ~1e-5, Spearman 1.0, with and without the
zero-shot prior, both through the harness and through a raw loop. Confirms
gradients reach everything trainable.

---

## 4. Data inventory (measured, not quoted)

Everything under `$MUTANT_CACHE_SCRATCH/processed/`.

| dataset | variants | score sets / targets | length | depths | assay |
|---|---|---|---|---|---|
| Tsuboyama | 550,141 | 412 / 479 | 32–74 aa | 389k singles, 159k doubles | ΔG |
| MaveDB | 6,112,263 | 2,465 / 2,465 | 1–1,863 aa | 3.36M singles, 2.33M doubles, 350k depth 3+, max 15 | binding, abundance, growth, expression, fluorescence |
| FLIP GB1 | 158,094 (8,733 in FLIP's downsampled subset) | 2 / 1 | 265 aa | 1–4, dense over 4 sites | binding |

Supporting tables: `tsuboyama/epsilon.parquet` (131,143 doubles with both singles
resolved, plus uncertainties and `pos_a`/`pos_b`/`seq_sep` across 559 site
pairs), `exclusion/proteingym_overlap.parquet`, and splits
`protein_level` (1317/165/165 targets), `protein_level_clusters`,
`flip_gb1_two_vs_rest`.

**ProteinGym** is reference-CSV only (217 assays, 2,465,767 variants, 696k
singles + 1.77M multiples, but multiples confined to 69 assays — **50 of which
are Tsuboyama-derived**). The 5.9 GB score files are not downloaded, and the
supervised CV fold assignments live in those files.

**CombinGym** is not cloned; `datasets/Makefile` has the target.

### FLIP GB1 published splits

Only `two_vs_rest` is exported. The other three green splits are in the raw zip
and the loader already supports them.

| split | train (depths) | test (depths) |
|---|---|---|
| `one_vs_rest` | 28 (0–1) | 8,705 (2–4) |
| `two_vs_rest` | 424 (0–2) | 8,309 (3–4) |
| `three_vs_rest` | 2,990 (0–3) | 5,743 (4 only) |
| `low_vs_high` | 5,089 (low fitness) | 3,644 (high fitness) |

`sampled` is orange under FLIP's semaphore and deliberately not exported.

---

## 5. Bugs found and fixed (what to watch for)

Recorded because each went undetected for a while and the same classes will
recur.

1. **Residue→token off-by-one.** Residue *p* (1-based) is token *p*, because BOS
   is token 0. The batch builder and the zero-shot scorer both used *p+1*, so
   the model mutated the neighbouring residue. Tsuboyama zero-shot went
   0.009 → 0.343 on the fix. It survived because tests only checked the code
   against itself and the sanity check validated the sequence string rather than
   the substituted token.
2. **Wrong end-of-sequence token** — reference ids ended in `<pad>` because the
   builder used the stub's vocabulary. Now uses the backbone's tokenizer.
3. **bf16 states into the fp32 head** — crashed every bf16 run.
4. **GPU row indices handed to pandas** — crashed every GPU evaluation, fp32 too.
5. **Splits looked up in the wrong directory.**

All five now have regression tests, each verified by reverting the fix and
confirming the test fails.

---

## 6. Open questions before a major run

### 6.1 Which evaluation regime? (the biggest one)

Four distinct setups, not interchangeable:

| regime | split | comparators | notes |
|---|---|---|---|
| zero-shot | none | ProteinGym main leaderboard | context row only, never beside a supervised number |
| within-assay | variants of one protein | FLIP, ProteinGym supervised, CombinGym | the cache's natural case: thousands of mutants, one reference |
| across-protein | held-out proteins | ESMRank only | hardest claim; breaks the assay embedding (§6.5) |
| pooled, variant-level | variants, pooled over proteins | standard in ddG prediction | the regime the architecture is actually built for |

`apply_split` already handles protein-level and within-assay split tables.

The sharpest variant for the epistasis question is a **site-pair-disjoint**
split: train on doubles at some site pairs of a domain, test on unseen site
pairs of the same domain. It keeps the assay embedding trained while removing
the per-position-tolerance shortcut that DATASETS.md names as the degenerate
solution. The epsilon table supports it directly via `pos_a`/`pos_b`.

### 6.2 Tsuboyama's epistasis is mostly assay saturation

The epsilon table looks rich — 80% of 131k doubles at |z|>3, epsilon variance
73% of total ΔΔG variance — but epsilon correlates with the predicted additive
effect at **r = −0.725**:

| predicted additive ΔΔG | observed | mean epsilon |
|---|---|---|
| −5.25 | −3.17 | +2.08 |
| −2.87 | −2.00 | +0.87 |
| −0.70 | −0.55 | +0.15 |

That is a measurement floor (observed ΔG bottoms out at −1.1), not biology. A
model would collect most available "epistasis" credit by learning the
compression curve. Restricting to ΔΔG_additive > −3 leaves 81k doubles with mean
|epsilon| 0.60 and r = −0.41, so real epistasis exists underneath but is smaller
and still entangled.

**Consequence**: PLAN.md item 14 (per-assay monotone link) is a prerequisite for
the gate experiment, not a later nicety. At minimum, evaluate the
ΔΔG_additive > −3 subset separately as a saturation control — if the model beats
the additive ablation only on the saturated subset, it learned the assay floor.

### 6.3 Reliability ceilings are not computed

`reliability` is null for all 412 Tsuboyama and all 2,465 MaveDB score sets.
DATASETS.md requires every correlation to be reported against its ceiling.
Tsuboyama carries per-variant standard errors, so this is computable.

### 6.4 How strict should the ProteinGym exclusion be?

Tsuboyama score sets with both singles and doubles:

| exclusion | sets | singles | doubles |
|---|---|---|---|
| none | 153 | 149k | 159k |
| exact matches only | 103 | 98k | 98k |
| exact + 30% identity clusters | 42 | 43k | 39k |

Strict costs 70% of usable domains. Since 50 of ProteinGym's 69 multi-mutant
assays are Tsuboyama-derived, strict exclusion is not optional if a ProteinGym
number is ever wanted.

### 6.5 Per-assay parameters: `assay_bias` / `assay_scale`

`assay_embed` is PLAN.md item 18 and is load-bearing. But `assay_bias` and
`assay_scale` (scalar affine calibration per score set) were added when the
model was built, are **not in PLAN.md**, and have no config switch. The nearest
planned thing is item 14's monotone link, which these pre-empt crudely.

They are inert for within-assay Spearman (affine with positive scale cannot
change a ranking) but matter for ΔG MSE, and they are two more per-assay
quantities that a protein-level split leaves untrained. **Decide: remove, or
keep behind a flag defaulting off.**

Related: across-protein splits leave `assay_embed` at random initialization for
every eval assay. That regime needs a story — predict the embedding from assay
metadata, or drop it — before its numbers mean anything.

### 6.6 Comparability

The only benchmark runnable today against published numbers is **FLIP GB1**, and
the data layer preserves FLIP's exact downsampling (8,733 rows via their `keep`
column) specifically so the comparison is valid. The validating move is to run
our `onehot_ridge` baseline on the split first: ridge on one-hot is *FLIP's own
baseline*, so reproducing their number validates split, metric and downsampling
before any model number is claimed.

Caveat: GB1 depth 4 is where the cache is weakest (Phase 0: 0.845 vs ≥0.92
typical), so `--control-backend recompute` is mandatory there, and with only 424
training variants `two_vs_rest` needs several seeds. `three_vs_rest` (2,990
train, pure depth-4 test) is a cleaner and less seed-sensitive epistasis test.

### 6.7 LoRA: variant vs shared not yet separated

The 0.017 gap is one seed. Needs 3+ seeds before shared LoRA's 1.46× training
cost (the reference is recomputed every step) is justified.

---

## 7. Suggested order

1. Decide §6.1 (regime) and §6.5 (per-assay scalars) — both change the run.
2. Compute reliability ceilings (§6.3); fit or defer the monotone link (§6.2).
3. FLIP GB1 `two_vs_rest` + `three_vs_rest`, with `onehot_ridge` reproduction
   and `--control-backend recompute`, 3–5 seeds. Cheap, and the only
   literature-comparable result available today.
4. `make combingym` — fills the one real gap for depth extrapolation.
5. The gate (PLAN.md 27): Tsuboyama, site-pair-disjoint, against the additive
   ablation, with the saturation control.
