# Results: Tsuboyama on ThermoMPNN's released split

2026-10-21. All runs frozen bf16 `esmc_300m`, fp32 head, `--loss mse_dg`,
`--target-col score` (ΔΔG kcal/mol), seed 42, `--eval-split val`, batch 32,
cosine schedule. Companion to HANDOFF.md. Scripts: `build_thermompnn_split.py`,
`pooled_vs_perassay_spearman.py`, `thermompnn_split_ceilings.py`.

**Every number below is val, single seed.** Test was deliberately not touched.

## The split

ThermoMPNN's `dataset_splits/mega_splits.pkl` (Kuhlman-Lab/ThermoMPNN) is
domain-level 239/31/28, MMseqs2 easy-cluster at 25% identity, random 80/10/10
over 163 clusters. Their domain names are *exactly* our `target_id`s — 298/298
match, no fuzzy mapping.

Two splits are written:

| | train | val | test |
|---|---|---|---|
| `thermompnn_mega` | 249 targets / 225,549 singles | 31 / 27,328 | 30 / 30,474 |
| `thermompnn_mega_strict` | 239 / 215,731 | 31 / 27,328 | **28 / 28,172** |

`_strict` drops modified-wild-type backgrounds (`1GB4.pdb_F53D`), matching their
Dataset Curation funnel (607,839 reliable → 391,090 singles → 272,712 after
dropping modified-WT). Its test set is 28 domains / 28,172 singles, which is
exactly the count their own loader scores (`mega_test.csv` is 28,312 rows minus
140 wild-type rows that `datasets.py` filters with `mut_type != "wt"`). Derived
two independent ways; they agree.

Non-modified-WT targets inherit their base domain's split, or the same domain
lands in both train and test. `build_thermompnn_split.py` asserts this.

With `--max-n-mut 2`, `_strict` gives 342,430 train / 38,940 val / 50,393 test.

**Matching their curation costs ~0.015 pooled**: identical config scored 0.7017
on `thermompnn_mega` vs 0.6869 on `_strict`. The 10 extra domains and 8,604
extra singles are useful training data. `_strict` is the honest split for a
published comparison; non-strict is the better model.

## Metric convention — this matters

ThermoMPNN's headline 0.73 is **pooled, variant-level**: one torchmetrics
`SpearmanCorrCoef`, `.update()` per mutation across all test PDBs, one
`.compute()` (`analysis/thermompnn_benchmarking.py`; `analysis/bootstrap.py`
resamples the flat dataframe with no protein grouping). The paper contains **no
methods sentence** stating this — cite Fig. 2B's single N = 28,312 scatter and
the repo code, not a quote.

Our `eval/depth.py` `ALL` row is the *mean of per-score-set* Spearmans. The two
are not the same number. `pooled_vs_perassay_spearman.py` prints both.

Measured gap (pooled − per-assay), singles: **−0.022 to −0.045** across six
runs. Subtract ~0.025 from any internal per-assay figure before setting it
beside a ThermoMPNN-style number.

On doubles the gap is much larger (−0.067 head-only, **−0.182** LoRA), and
pooling singles+doubles together *flips* it positive (+0.022) because the model
profits from easy between-depth separation. **`ρ@all` pooled is a flattered
number; quote the depth-stratified rows.**

## Published reference points (verified from primary sources)

PNAS full text via Europe PMC (PMC10861915) + SI Appendix.

ThermoMPNN, Megascale test, pooled: **SCC 0.73 / PCC 0.75 / RMSE 0.71**
(Table 2); triplicate **0.725 ± 0.003 / 0.754 ± 0.004 / 0.708 ± 0.007**
(Table 1). Their rejected "Added fine-tuning" variant reaches **SCC 0.747** —
better on Megascale test, dropped for poor Fireprot transfer. So the honest
"beat published" bar is 0.747.

Same test set, Table 2 (SCC): RaSP 0.67†, PROSTATA 0.59†, FoldX 0.57,
Rosetta 0.56, ProteinMPNN 0.49, MAESTRO 0.47, ACDC-NN 0.45, ACDC-NN-Seq 0.44,
mCSM 0.41, MUPRO 0.29, ThermoNet 0.27. († retrained on Megascale.)

**ThermoMPNN requires structure** (frozen ProteinMPNN `v_48_020.pt`, wild-type
backbone only), and all Megascale structures are *predicted*: 181 AlphaFold
natural domains (pLDDT > 0.75), 90 trRosetta hallucinations, 19 RosettaDesign.
Our model is sequence-only, so 0.725 is not a like-for-like target.

Their ddG sign is **destabilizing positive** — `datasets.py` negates raw
`ddG_ML`. Ours keeps Tsuboyama's raw convention (destabilizing negative).
Harmless for Spearman magnitude, silently wrong for RMSE/MSE comparisons.

Doubles: ["Protein stability models fail to capture epistatic interactions of
double point mutations"](https://pmc.ncbi.nlm.nih.gov/articles/PMC11659742/),
Megascale-D (127,476 doubles / 153 proteins): **additive ThermoMPNN 0.59 beats
epistasis-aware ThermoMPNN-D 0.54**; Mutate Everything 0.55, FoldX 0.47,
Rosetta 0.42. Best ΔΔΔG-only correlation anyone reports: ~0.24–0.25.

## Reliability ceiling (HANDOFF 6.3, now answered for this split)

`attenuation_ceiling` = sqrt(1 − E[se²]/Var):

| | n | pooled ceiling | per-assay mean |
|---|---|---|---|
| val singles | 27,328 | 0.9979 | 0.9967 |
| test singles | 28,172 | 0.9984 | 0.9974 |

Mean se 0.056 against Var(score) 0.86. **The ceiling is ~1.0, so saturation
explains nothing** — we are at ~0.70 with ~0.28 of headroom, and ThermoMPNN's
0.725 is nowhere near a noise limit either.

Caveat: `score_se` is per-variant ΔG *fitting* precision. The 2,022 WT replicate
rows have std **0.111**, which is a better reproducibility estimate than the
0.056 SE column. The honest ceiling is nearer the 0.11 figure.

## Singles, `_strict` — the 2×2

Per-assay ρ@1 (pooled in parentheses):

| | 1 epoch (6,894 steps) | 2 epochs (13,788 steps) |
|---|---|---|
| head-only (3.64M trainable) | 0.7039 (0.6787) | **0.7280 (0.7005)** |
| + variant LoRA r=8 (7.10M) | 0.7097 (0.6869) | 0.7197 (0.6947) |
| **Δ (LoRA − head)** | **+0.0058** | **−0.0083** |

**Variant-only LoRA does not help.** The +0.006 at 1 epoch is a
convergence-rate artifact — the adapters converge slower, so they look better
when both arms are under-trained and lose once they aren't. Demonstrated
directly: at identical step 6894, fully annealed, LoRA is +0.006 ahead; at the
same step half-annealed, it is **−0.049 behind**.

Capacity is not the binding constraint. With a ceiling of ~1.0 and 0.28 of
headroom that 2× the parameters cannot touch, the limit is most plausibly the
**239 training domains** — variants within a domain share one reference, so the
effective sample size for anything domain-general is nearer 239 than 215,731.
This is the main argument against `--train-backbone`.

### The "2 epochs beats 1 epoch" result is mostly a schedule artifact

`lr_scale` computes `prog = (step - warmup) / (total - warmup)` with `total`
across *all* epochs, so a 1-epoch run anneals to zero by step 6894 while a
2-epoch run is at ~half peak there. Decomposing head-only's +0.024:

- at step 6894: 2ep run 0.7214 vs 1ep run 0.7039 → **+0.0175 from the gentler anneal**
- 0.7214 → 0.7280 over the whole second epoch → **+0.0066 from the extra epoch**

So ~73% is the 1-epoch run over-annealing. Epoch 2 drops train loss 52%
(0.8906 → 0.4114) for +0.002 val ρ — memorization. **A 1-epoch run with a
cosine floor or `--schedule constant` should recover most of it at half cost.**

Intermediate points of a cosine run say little about convergence: the curve
always flattens near its end because the LR is going to zero.

## Multiple mutations, `_strict`, `--max-n-mut 2`, 1 epoch (10,817 steps)

Per-assay (pooled in parentheses):

| | ρ@1 | **ρ@2** | ρ@all |
|---|---|---|---|
| head-only | **0.7129** (0.6914) | **0.6440 (0.5768)** | 0.7212 (0.7434) |
| + variant LoRA r=8 | 0.6950 (0.6676) | 0.6304 **(0.4488)** | 0.7028 (0.6930) |
| **Δ** | −0.0179 | −0.0136 **(−0.128)** | −0.0184 |

**LoRA hurts on doubles too, including at depth 2** — the depth where
variant-variant attention and the interaction module are the only thing it
could be helping. Worse than the singles story: on singles LoRA was ahead at
1 epoch; here it is already −0.018 behind at the budget that previously
flattered it.

The pooled doubles collapse (0.4488 vs 0.5768, against a per-assay gap of only
0.014) is diagnostic. LoRA's doubles convention gap is −0.182 vs −0.067 for
head-only: it ranks well *within* domains but puts domains on inconsistent
ΔΔG scales. That is domain-specific overfitting, and it is **invisible in
per-assay numbers**.

**Training on doubles improves singles**: ρ@1 0.7129 at 1 epoch vs 0.7039 for
the singles-only 1-epoch run, +0.009. Argues for always using `--max-n-mut 2`.

Against the Megascale-D references (pooled): head-only **0.5768** sits just
under the additive ThermoMPNN baseline (0.59) and above ThermoMPNN-D (0.54);
LoRA's 0.4488 is below everything but ThermoNet.

## Known bugs and gaps found

1. **`--train-backbone` is silently wrong.** Reference-cache staleness is gated
   on shared LoRA only (`variant_model.py` `reference_states`:
   `if self._shared_lora() and self.training`). With a trainable backbone and no
   shared adapters, `_shared_lora()` is False, so the reference is computed once
   per domain and **cached forever** while the weights move — and `train()` only
   clears the cache under `_shared_lora()`, so the train/eval switch does not
   invalidate it either. The cached path also calls `forward_reference(ids)`
   without `grad=`, which defaults to `no_grad` (INTERFACES.md:109), so the
   reference is detached too. Meanwhile the optimizer dutifully adds all ~300M
   backbone params. Fix: gate on
   `self._shared_lora() or any(p.requires_grad for p in backbone.parameters())`.
   No test exercises `--train-backbone` end to end.

2. **`onehot_ridge` silently vanishes on across-protein splits.**
   `OneHotRidgePerAssay` fits one ridge *per `score_set_id`*; on a protein-level
   split the eval score sets have no training rows, every prediction is NaN,
   `ev2` empties and the harness skips the block without a word. A requested
   baseline missing from the results table is how a run gets misread later —
   this should warn. (It also means "reproduce FLIP's ridge first" is right for
   within-assay GB1 and impossible here.)

3. **`per_assay_affine` is dead config.** Declared in `VariantModelConfig`, read
   nowhere; the `assay_bias`/`assay_scale` affine is applied unconditionally and
   `build_config` never passes the flag. HANDOFF 6.5's "decide: remove, or keep
   behind a flag defaulting off" is a two-line wiring job.

4. **FLIP GB1's reference is a 265-aa fusion construct**, not GB1.
   `target_id` is `FLIP_GB1_5LDE_1`: GB1's 56 residues, then `LEVLFQGP`
   (PreScission site), a fusion partner, and a C-terminal His6. FLIP states its
   splits carry the 56-aa sequence. Mutated sites still land correctly (residues
   39/40/41/54 are V/D/G/V, since GB1 is the N-terminal 56), so there is no
   off-by-one — but the ESM-C masked marginals, the additive-branch
   initialisation and the mean-pooled `ref_summary` all see ~200 residues of
   irrelevant linker, and 4-of-265 is far sparser than 4-of-56, which
   **flatters the cache's speedup and its depth-4 fidelity**. Trim before
   quoting either.
   (Tsuboyama's ingest is clean: our stored sequence is byte-identical to their
   `aa_seq`, and mutant strings and `parse_mutant` positions match exactly.)

5. **Our FLIP GB1 depth splits do not match FLIP's.** `low_vs_high` matches
   exactly (5,089/3,644) but all three depth splits differ: one_vs_rest 28 vs
   29, two_vs_rest 424 vs 427, three_vs_rest 2,990 vs **2,968**. The fault is in
   depth-boundary assignment, not downsampling.

## The ΔΔG measurement floor is per-domain (explains HANDOFF 6.2)

Absolute ΔG cannot be measured below **≈ −1.0 kcal/mol** — a hard clamp,
identical across all 28 test domains (min `dG_ML` per domain: median −0.96,
range −1.00 to +0.53). Past that the domain is fully unfolded and there is no
protease signal left. So the measurable ΔΔG range is set by each domain's own
stability:

> ΔΔG_observed ≳ −(ΔG_wt + 1.0)

`corr(dG_wt, min_ddG) = −0.873` over the 28 test domains. A domain with
ΔG_wt = 1.33 can never show worse than about −2.3; one at 4.71 reaches −5.7.

This is **left-censoring with a per-domain threshold**, and it manufactures the
"diminishing costs" epistasis the field reports: any double whose additive
prediction falls below its domain's bound is clamped, producing positive ε in
proportion. That is HANDOFF's ε-vs-additive r = −0.725 and the published −0.9
kcal/mol mean coupling, same cause.

Two consequences:

- HANDOFF 6.2's saturation control can be **much sharper**. Instead of a global
  ΔΔG_additive > −3 cutoff (81k doubles), keep every double whose additive
  prediction lies inside its own domain's measurable window — removes the
  artifact more precisely, discards less data.
- It needs a column we do not have. `variants.parquet` carries ΔΔG only; this
  requires ΔG_wt (`dG_ML`) per target. That addition would also enable a proper
  **censored (Tobit-style) likelihood** with a per-domain left-censoring
  threshold, which is the principled version of what PLAN.md 14's monotone link
  approximates crudely.

## Context: this split is easier than `protein_level`

Zero-shot ESM-C masked marginals score **0.4725** here vs 0.343 on
`protein_level` val. ThermoMPNN's domain selection is substantially easier for
ESM-C. The splits are not interchangeable, and it cuts both ways — our numbers
look better here, but so does everyone's.

## Dataset shape, measured

479 targets, 32–74 aa (median 56): 318 natural (PDB id), 90 Rocklin-style de
novo, 40 pipe-prefixed, 20 trRosetta hallucinations, 10 `v2_` natural.

Measurement is **protease resistance**, not stability directly. K50 is the
protease concentration cleaving half the molecules; comparing it to a predicted
*unfolded-state* K50 gives ΔG. Done independently with **trypsin** (`_t`) and
**chymotrypsin** (`_c`), then combined (`deltaG`), with ML global fits
(`dG_ML`, `ddG_ML`). `aa_seq_full` is `SAGG` + domain + `SAGG` (display
linkers); `aa_seq` — the scored one — still carries an `NLYFQG` TEV remnant, and
**mutation numbering is 1-based on `aa_seq` including that remnant** (for
`3DKM.pdb`, `N1Q` is the N of `NLYFQG`).

Singles ΔΔG: mean −0.758, median −0.495, range −5.72 to +4.02; 77.8%
destabilizing. Assay dynamic range ~5 kcal/mol (the paper names this as a
limitation for training).

## Open, in priority order

1. **`--ablation additive`** on `_strict` `--max-n-mut 2` at the same budget.
   Decides whether head-only's ρ@2 = 0.6440 (0.5768 pooled) is real epistasis or
   just summing singles. If it matches the additive arm, the interaction module
   is dead weight — the exact result the published paper reports for
   ThermoMPNN-D. **Nothing else here means much until this runs.**
2. 2-epoch doubles pair, to confirm the LoRA verdict at a budget where it is not
   convergence-limited.
3. `--schedule constant` or a cosine floor at 1 epoch, to test the schedule
   decomposition above.
4. Fix bug 1, warn on bug 2, wire bug 3.
5. Per-domain difficulty analysis — per-assay mean 0.7280 vs median 0.7645 is
   left-skewed, so a few domains drag the mean. If they are the de novo designs
   or the shortest domains that is structural; otherwise noise. (A script for
   this was blocked twice by a PreToolUse hook timeout and never ran.)
