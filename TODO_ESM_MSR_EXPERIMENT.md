# Initial Megascale experiment using the ESM-MSR benchmark

Created 2026-11-02. Status: plan only; data have not been downloaded or audited
for this experiment, and no runs have been launched by this task.

## Question and scope

Can the current sparse mutant-attention model learn transferable stability
effects on the ESM-MSR training partition, improve on an additive predictor
on unseen proteins, and do so at a useful computational cost?

Reuse ESM-MSR's released data preparation and protein partitions. Start from
the original pretrained ESM-C 300M weights, not an avGFP-finetuned checkpoint.
This is supervised Megascale training and held-out-protein evaluation, not yet
a demonstration of a generally useful mutation-pretrained initialization.
Sequence pretraining, few-shot adaptation, and a heterogeneous mutation corpus
are follow-ups, not prerequisites.

Sources to pin before implementation:

- [ESM-MSR repository and reproduction instructions](https://github.com/SKTeamLab/esm-msr#reproducing-benchmarks)
- [Released data archive](https://doi.org/10.5281/zenodo.21539277)
- [ESM-MSR preprint](https://www.biorxiv.org/content/10.64898/2026.06.04.730231v1.full)
- [Original Megascale study](https://www.nature.com/articles/s41586-023-06328-6)
- Local implementation evidence: [pilot results](experiments/IMPLEMENTATION_PILOT_RESULTS.md)

The repository README names `preprocessed_data.tar.gz` and the benchmark split
`hyperopt_splits`. Verify their contents and their relationship to the paper's
training run; a benchmarking archive is not automatically a complete training
release. The README also distinguishes its small bundled checkpoint from the
exact paper checkpoints on Hugging Face. Do not silently substitute one.

## 1. Import and freeze the data protocol

- [ ] Pin the ESM-MSR code revision, archive version/checksum, split-file hash,
  paper version, and upstream data provenance/terms. Keep downloads and caches
  under `$MUTANT_CACHE_SCRATCH`, following `CLAUDE.md`; keep scripts and small
  manifests under `experiments/` exactly as run.
- [ ] Inspect the training loader and preprocessing scripts as well as the
  README. Identify exact source tables, target column, sign convention, filters,
  exclusions, duplicate handling, reference definitions, and any augmentation.
  Distinguish measured examples from synthetic/reversion examples. Document
  deviations instead of claiming an exact training reproduction.
- [ ] Preserve released train/validation/test protein assignments. Do not make
  a random variant split or substitute the existing ThermoMPNN manifests.
  Confirm that `hyperopt_splits` really supplies the intended final test set
  and determine which partitions were used for model selection.
- [ ] Produce a split audit: proteins, unique reference sequences, unique
  variants, singles, doubles, WT rows, exclusions, label distributions, and
  available uncertainty/censoring fields. The paper describes 43,700 test
  variants (32,968 singles and 10,732 doubles); treat these as reconciliation
  checks, not instructions to discard arbitrary rows until counts match.
- [ ] Reconstruct every mutant from its reference and substitutions. Assert
  correct numbering, reference residue identities, canonical mutation ordering,
  and depth. Normalize equivalent variant strings before checking duplicates.
- [ ] Audit exact sequence/genotype overlap and homologous references across
  partitions, including alternate names and engineered reference backgrounds.
  Reproduce the published homology filtering where possible. Flag discrepancies;
  keep a separately named stricter split if needed rather than silently changing
  the published benchmark. Do not use test outcomes to choose exclusions.
- [ ] Build complete A/B/AB measurement groups within each held-out partition
  and the same reference background. Record how many doubles have both singles.
  Measured held-out singles may define evaluation residuals, but cannot become
  training labels or prediction inputs in this zero-shot-across-proteins task.
- [ ] Preserve raw energy labels. Any scaling or calibration is fitted on
  training data only and inverted before physical-unit metrics. Never compute
  epistasis from rank-transformed labels or independently standardized proteins.
- [ ] Save a machine-readable exclusion log and immutable split manifests.
  Loader checks should fail on missing validation, target ambiguity, unexpected
  overlap, and mutation reconstruction errors.

**Gate:** a documented mapping from upstream records to local manifests, with
unresolved differences explicitly listed before full training.

## 2. Minimum experiment arms

| Arm | Definition | Purpose |
| --- | --- | --- |
| S: sparse joint model | Current ragged mutant attention, variant-only LoRA, `sum_mean_count` head; train on singles and doubles | Main candidate |
| A: learned additive | Same pretrained backbone and variant LoRA scope; score each substitution independently and sum the single effects; train on the same singles and doubles | Tests whether joint processing helps under matched label access |
| S-add: additive readout of S | At evaluation, sum S's predictions for the constituent singles instead of scoring the double jointly | Paired diagnostic with identical learned parameters; no new training |

- [ ] Give A gradients from double labels through the sum of its predicted
  singles, so A and S have access to the same measured training rows. Cache and
  reuse single predictions where valid. A separate singles-only-trained baseline
  is optional and must be labelled as having different supervision.
- [ ] Define a zero-WT effect consistently. If the head outputs an unconstrained
  scalar `f`, use `d(x) = f(x) - f(WT)` for both training and evaluation, or
  implement an explicitly zero-anchored head. Then A predicts
  `d(A) + d(B)` and S predicts `d(AB)`. Verify WT centering for every protein.
- [ ] Do not call disabling variant-to-variant attention an additive baseline:
  nonlinear pooling and changing reference masks can still produce interactions.
- [ ] Use no learned assay/protein-ID lookup that requires parameters for unseen
  proteins. Reference sequence is available; held-out experimental labels are not.

The detailed E3 section of the pilot report selects `sum_mean_count`; its old
header and the old TODO status still say E3 is unrun. Use the detailed completed
results as evidence. E3 supports this starting head on avGFP, not its superiority
on Megascale. Do not repeat the entire head screen here.

## 3. Training recipe and bounded tuning

- [ ] Start with ESM-C 300M, frozen reference backbone, variant LoRA rank 8,
  alpha 16, targets q/o/gate/up/down, and the minimal `sum_mean_count` head.
  Use AdamW, weight decay 0.01, gradient clipping 1. These are starting choices,
  not established Megascale optima.
- [ ] Use MSE on the signed experimental delta-delta-G target for the first
  comparison, with identical loss and target convention in S and A. A rank-only
  objective does not identify the additive energy scale needed for residuals.
  Rank loss or a regression/ranking mixture is a later controlled change.
- [ ] Sample proteins uniformly, then variants uniformly within a selected
  protein, using the same sampling policy in both arms. Log effective singles/
  doubles exposure. Group microbatches by reference for cache reuse. If choosing
  upstream sampling instead, document and freeze that choice before screening.
- [ ] Run numerical checks in fp32. For training, use the repository convention
  of a frozen bf16 base with fp32 stored heads/adapters, provided a small
  train/validation-only smoke run is stable. Record adapter arithmetic dtype
  separately from storage dtype. Keep precision identical across arms.
- [ ] Smoke-test one seed of S and A on a fixed training subset and validation
  subset: finite predictions/losses, intended parameters update, frozen reference
  stays fixed, checkpoints reload correctly, and memory fits. Never smoke-test
  on final test outcomes. Record actual batch size and accumulation.
- [ ] Set an initial budget of ten training-set-equivalent exposures, with one
  exposure defined as N_train sampled rows under the frozen sampler. Evaluate
  full validation once per exposure; use cosine decay with 5% warmup and record
  actual optimizer steps. The avGFP 400-step budget must not be copied blindly
  to a much larger dataset. If both arms are still improving at the boundary,
  extend both to twenty exposures under a documented schedule before testing.
- [ ] Small screen, seed 0: adapter LR `{3e-4, 1e-3}` crossed with head LR
  `{1e-3, 3e-3}` for each arm (eight configurations total). No rank, head, target,
  and schedule Cartesian sweep. Allow each arm its own validation-selected LR.
- [ ] Primary checkpoint/configuration criterion: lowest macro-per-protein
  validation double-mutant RMSE. Report singles validation metrics alongside it;
  tie-break with singles macro RMSE, then earliest checkpoint. Keep this rule
  fixed rather than switching to whichever metric favours a model.
- [ ] Repeat each selected configuration with seeds 1 and 2, retaining seed 0
  as part of the three-seed report. Seeds change optimization and sampling, not
  protein split membership. Report all three, not only the best seed.
- [ ] Record code/diff hash, manifests, seed, effective sampling, optimizer
  groups, precision, schedule, selected checkpoint, active parameters, training
  time, validation curves, and peak allocated/reserved memory for every run.

**Gate:** S and A train reliably, have comparable tuning access, and have frozen
selected configurations. If training fails, diagnose on train/validation before
adding more architectures or accessing final test outcomes.

## 4. Evaluation protocol

- [ ] Implement and validate the evaluator using validation data first. Once
  configurations are frozen, evaluate all selected seeds on the test partition
  once as a batch of final analyses. Any later test-informed changes are
  exploratory and need new independent confirmation.
- [ ] Report singles and doubles separately: per-protein Spearman, MAE, and
  RMSE; macro means as primary summaries; median and pooled scores as clearly
  labelled secondary summaries. State minimum sample counts and excluded
  constant/undefined correlations. Compare methods on identical eligible rows.
- [ ] On complete measurement groups compute:

  ```text
  epsilon_observed = ddG(AB) - ddG(A) - ddG(B)
  epsilon_predicted = d(AB) - d(A) - d(B)
  ```

  For absolute free-energy source labels, first derive consistent WT-relative
  effects; the equivalent cycle includes the WT energy term. Use physical units.
  Report epistasis RMSE/MAE and per-protein correlation, with matched-group counts.
  The additive model has epsilon = 0: its residual error is meaningful but its
  residual correlation is undefined, not zero.
- [ ] Report S versus both A and S-add on total double effects. Do not infer
  learned interactions solely from improved total-double Spearman.
- [ ] Include the sum of *measured* singles as a privileged diagnostic. It uses
  held-out labels, is not a deployable prediction baseline, and is not a ceiling.
- [ ] Evaluate stabilizing-variant retrieval using the upstream threshold/sign
  and NDCG@96 definition after verifying their implementation, including gain
  transformation and small-protein handling. Report precision/recall at the
  fixed budget as an interpretable companion. Do not redefine metrics to match
  a favourable published number.
- [ ] Show full published-filter results first. Add a prespecified high-confidence,
  uncensored sensitivity analysis if metadata support it. Explain floor effects
  and selection bias; do not select extreme observed residuals after seeing
  results. Fit uncertainty is not a complete experimental noise ceiling.
- [ ] Report paired per-protein differences and protein-bootstrap intervals,
  retaining seed-level results. Do not treat thousands of variants from one
  protein as independent replication. Site-pair breakdowns are useful diagnostics.
- [ ] Benchmark on the same hardware and variant lists: cold time including
  reference preparation, warm-cache throughput, peak memory, and cache storage.
  Use matched batches and record variant count per protein. Include single and
  double inference separately. Avoid extrapolating short-domain speedups to
  long proteins without measurement.

## 5. External ESM-MSR comparison

- [ ] Obtain the exact paper checkpoint/configuration and reproduce its scores
  on the frozen test examples. Audit checkpoint training overlap. Record missing
  predictions and evaluate the intersection as a secondary comparison if needed.
- [ ] Use the full mutant-context inference setting for double/epistasis claims;
  the README's `--skip_reverse` shortcut is an additive approximation. Optionally
  report that shortcut separately as an accuracy/cost baseline.
- [ ] Match evaluation definitions, aggregation, sign, and units. Do not compare
  our macro correlation with another paper's pooled or median correlation.
- [ ] ESM-MSR uses ESM3 and structural inputs; ours uses ESM-C sequence inputs.
  Describe this as a comparison of systems, not an isolated architecture ablation.
  If local reproduction is unavailable, clearly label published results and
  avoid a direct superiority claim based on unmatched evaluations.

## 6. Interpretation and next decision

There is no justified universal target such as 'beat 0.59' across incompatible
splits. Define progress through paired comparisons on this exact protocol.

| Outcome | Interpretation / next action |
| --- | --- |
| S consistently improves double error and ranking over A and S-add, with useful residual prediction | Joint mutation processing merits further study; add the reference-only nonlinear decoder control |
| S improves total-double metrics but not residual prediction | Better stability prediction, not yet evidence of improved epistasis; inspect calibration and single-effect gains |
| S ties A | Retain additive efficiency; diagnose objective, head, and representation on validation before scaling |
| S approaches ESM-MSR accuracy with lower measured cost | Promising system tradeoff; establish a same-backbone full-mutant control and test on an external dataset |
| S trails ESM-MSR | Diagnose fairly: structure, backbone, supervision, and optimization differ; this alone does not reject sparse computation |

A repeated +0.02 macro double Spearman together with lower double and epistasis
error would be an encouraging pilot signal, not a significance threshold or
publication requirement. Report uncertainty and effect size even if smaller.
Likewise, report the measured accuracy/cost frontier rather than declaring an
arbitrary acceptable accuracy loss after seeing results.

If promising, the next controlled arms are (1) an ESM-C reference-feature nonlinear
mutation decoder with equal supervision/tuning and (2) full-mutant ESM-C LoRA.
Verify adapter application in the latter: the existing `exact` backend with
variant-only LoRA is not automatically conventional all-position LoRA, and
`recompute` is the clamped correctness oracle, not full-mutant processing.
Then test external transfer or held-out-protein few-shot adaptation with explicit
support/query splits. Do not conclude general mutation pretraining from this
single stability benchmark.

## Deliverables

- [ ] Frozen upstream provenance and local split/exclusion audit.
- [ ] Raw import, training, and evaluation scripts under `experiments/`.
- [ ] Three selected seeds each for S and A, plus S-add diagnostic predictions.
- [ ] Per-variant predictions, per-protein metrics, residual-group table,
  learning curves, and compute measurements.
- [ ] ESM-MSR reproduction or an explicit explanation of remaining mismatch.
- [ ] A concise results document with keep/reject/investigate decisions and
  limitations. Preserve failed/negative runs and separate validation from test.
