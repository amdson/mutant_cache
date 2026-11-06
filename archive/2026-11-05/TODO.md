# Implementation and prediction-head improvements

Progress (2026-11-02): §1, §2, §4, E0-E4 run (E3 head H2 sum_mean_count chosen; H3 rejected
by its capacity control; E4 neutral on GB1). bf16 deployment check and E5 not run. Pilot scale
only — revisit along a label-budget scaling curve before treating any choice as settled. E3 with V_1e-3 is recommended below. Results and the latent defect:
`experiments/IMPLEMENTATION_PILOT_RESULTS.md`.

## Immediate experiment recommendation (updated 2026-11-02)

The requested first experiment—avGFP ds1280, original minimal head, comparing
full-backbone / variant-LoRA / shared-LoRA adaptation—is **already completed
as E2 below**. Do not rerun the screen by default. Its detailed protocol and
run matrix remain in E2; outcomes are in
[IMPLEMENTATION_PILOT_RESULTS.md](experiments/IMPLEMENTATION_PILOT_RESULTS.md).

Selected-setting mean development Spearman over replicates 0/1/2:

| Configuration | Mean development Spearman | Mean cold wall time |
| --- | --- | --- |
| Full backbone, LR 1e-5 | 0.568 | 211 s |
| Shared LoRA r8, LR 1e-3 | 0.605 | 166 s |
| Variant-only LoRA r8, LR 1e-3 | 0.688 | 136 s |

Variant LoRA won on every replicate, but much of the aggregate improvement
was between mutation depths. These are pilot results, not a broad superiority
claim. All arms selected early checkpoints (steps 40–60) and later overfit;
longer training is not the next default experiment. The audit found no
missing-LoRA-path defect.

**Next recommended run: E3, isolate the prediction head using V_1e-3.**
This recommendation resolves which configuration to use for the head comparison;
it does not authorize or record a launched run. Keep F_1e-5 as historical context.

- [x] Compare `mean_count`, `sum_mean_count`, and `wt_relative` against the
  existing `mean` baseline, using avGFP nominal budget 1,280, replicates 0/1/2.
  Nine new runs; reuse the three compatible V_1e-3 baseline runs from E2.
- [x] Freeze training settings: ESM-C 300M, sparse ragged execution, fp32,
  variant LoRA r8/alpha16, targets q/o/gate/up/down, adapter LR 1e-3,
  head LR 3e-3, uniform RankNet on raw score, AdamW wd 0.01, clip 1,
  batch 256, 400 optimizer steps, cosine schedule and the same resolved
  40-step warmup as E2. The script requests warmup 50; the harness caps it
  at 10% of the step budget. Verify that resolved value in each run record.
- [x] Use the existing development manifests: 1,024 train / 256 validation /
  5,000 development genotypes. Select by lowest validation rank loss at the
  same 20-step evaluation cadence. Do not select using the METL supertest or
  change the criterion after examining development results.
- [x] From the repository root, the existing script's command is:

  ```bash
  REF=V_1e-3 bash experiments/e3_avgfp_heads.sh
  ```

  It uses GPU 0, writes into scratch `runs/pilot_e3`, and skips completed runs.
  Check GPU availability and that any reused results match the current code
  and protocol before execution. Preserve its logs and raw script.
- [x] Report each paired head-minus-mean difference, overall AND within depth,
  validation curves, selected step, active parameter count, cold time, and peak
  memory. Repeated +0.02–0.03 development Spearman would be encouraging; report
  count-driven gains honestly and do not treat three replicates as a definitive
  significance test.
- [x] If WT-relative wins, run the parameter-matched mutation-only projection
  control specified in E3. Then confirm the chosen head on E4's avGFP-320 and
  GB1 tasks. If no head helps consistently, keep the mean head and investigate
  warmup/peak-LR sensitivity as a separate experiment with equal tuning access.
  Do not alter the schedule during the head comparison.

Pretraining, larger rank sweeps, and new datasets remain deferred until this
small controlled comparison is reviewed. No experiments were launched when
this recommendation was added.

Requested 2026-11-01. Prioritize implementation correctness, LoRA behavior,
and a small readout improvement. This is an implementation plan, not evidence
that LoRA is broken. Preserve existing configurations for controlled comparisons.
Do not start a broad architecture redesign or new benchmark sweep as part of
these tasks.

Start with the ordered experiment queue in section 5. Sections 1–4 specify
the implementation work and safeguards needed by that queue.

## 1. Audit LoRA before tuning it

Relevant code: `mutant_cache/cache/lora.py`, `backbone/esmc.py`,
`cache/variant_cache.py`, `models/variant_model.py`, and `harness.py`.

- [x] Run existing LoRA, ragged-cache, minimal-model, and harness tests first.
  Existing tests already cover zero-initialized identity, reference invariance,
  adapter updates, and shared-reference gradients; extend gaps rather than
  duplicating them.
- [x] Audit every logical projection (`q,k,v,o,gate,up,down`) across ragged,
  padded, and clamped-recompute execution. Check normalization, RoPE placement,
  fused-projection slices, adapter scaling, and masking. Compare forward values
  AND gradients with nonzero adapters in fp32. Full `exact` execution is a
  different architecture and is not the numerical equivalence oracle.
- [x] Verify optimizer groups contain each active parameter exactly once, use
  the requested head/adapter/backbone learning rates, and exclude frozen base
  parameters. Check shared-only, variant-only, and combined configurations.
- [x] Log adapter parameter dtype, gradient norms, actual parameter update
  norms, and adapter-delta/base-output ratios on a small diagnostic batch.
  With zero-initialized B, zero A gradients on the first step are expected;
  verify A receives gradients after B changes.
- [x] Check mixed precision explicitly: fp32 adapter storage does not imply
  fp32 adapter arithmetic. `LoRADelta.forward` currently casts weights to the
  activation dtype. Compare existing behavior with an fp32 adapter-compute
  diagnostic, including where the delta is added/cast back. Measure prediction,
  gradient, update, runtime, and memory differences before changing defaults.
- [x] Verify reference caches stay reusable with frozen base + variant LoRA,
  but rebuild with gradients when shared/base weights move. Verify evaluation
  after an optimizer step and checkpoint restoration cannot reuse stale states.
  The old train-backbone cache bug appears addressed by `_reference_moves()`;
  validate the current behavior rather than assuming the old bug persists.
- [x] Add a small multi-step optimization regression for any confirmed defect;
  record the failing configuration and before/after behavior.

Acceptance: numerical agreement with the clamped oracle within justified
precision-specific tolerances, correct stream-specific gradients and cache
lifetime, and observable updates to the intended parameters.

## 2. Implement an opt-in mutation-aware prediction head

Start from the successful minimal head, not the older full head. Retain the
current mean-pool -> MLP path unchanged as the baseline. Add explicit head
choices to configuration/CLI and include them in run records/checkpoints.

- [x] Add a pooling-only variant: concatenate sum-pooled mutant states,
  mean-pooled mutant states, and `log1p(k)`, then apply a small MLP. Also allow
  mean + count alone to isolate the value of count from sum pooling.
- [x] Add a WT-relative variant. For each real substitution at position i,
  gather the final normalized reference state at the same sequence position
  and form:

  `z_i = phi(concat(h_mut_i, h_wt_i, h_mut_i - h_wt_i, e_ref_i, e_alt_i))`

  Use a shared small projection/MLP `phi` (initial output width 128), with
  small learned canonical-residue embeddings. Predict with:

  `MLP(concat(sum_i(z_i), mean_i(z_i), log1p(k)))`.

- [x] Obtain reference states through the existing reference-state lifecycle.
  Match normalization and token indexing, including boundary-token offsets.
  Preserve reference gradients for shared-LoRA/backbone training; do not
  accidentally detach them or force gradients for a frozen reference.
- [x] Use only real substitutions for all pooling/counts. Define WT/k=0 as
  zero pooled vectors and zero count feature with a finite head prediction.
  Padding and neighboring variants in a batch must not affect predictions.
- [x] Support ragged, padded, and exact execution with the same head semantics.
  WT-relative features use WT states even for the exact mutant execution path.
- [x] Keep marginal priors, assay IDs, extra backbone-input embeddings, and
  reference-summary branches out of these minimal-head variants. Construct
  and optimize only their active head modules where practical, while retaining
  compatibility with existing configurations/checkpoints.
- [x] Test mutation-order invariance, padding/batch-composition invariance,
  correct WT-position gathering, k=0 handling, and gradients through both
  trainable streams. Check ragged/padded prediction and gradient agreement.

Acceptance: the old default remains reproducible, all new head choices are
recorded, and the new readout adds no full mutant-sequence backbone pass.

## 3. Small controlled optimization comparisons

- [ ] Use avGFP development data as the main diagnostic and GB1 as a regression
  check. Existing avGFP runs copied GB1 hyperparameters without avGFP retuning;
  do not attribute that gap to architecture alone.
- [ ] At nominal budgets 320 and 1,280, compare original minimal, mean+count,
  sum+mean+count, and WT-relative heads. Use three matched data replicates,
  the same initialization seeds where meaningful, and equal validation access.
  Report actual training/validation counts separately.
- [ ] Establish the head comparison with a fixed training configuration first.
  After correctness checks, investigate LoRA scope separately: variant-only,
  shared-only, and sparse full-backbone training. Do not change head, rank,
  targets, precision, and schedule simultaneously.
- [ ] For LoRA, start at rank 8; tune adapter LR independently of head LR
  (e.g. 1e-4, 3e-4, 1e-3) before increasing rank. Compare rank 8 versus 16 at
  fixed alpha/r, then default projection targets versus all projections if
  justified. Variant k/v adapters are a hypothesis to test, not a known fix.
- [ ] Compare identical precision first when diagnosing LoRA versus full
  training, then benchmark the intended mixed-precision deployment settings.
  Do not conclude that LoRA loses from unequal schedules or training budgets.
- [ ] Record train/validation curves, best step, depth-stratified and overall
  Spearman, runtime, and peak memory. Count-aware heads must improve within-depth
  discrimination or have their gains explicitly described as between-depth.
- [ ] Select using validation only. Reuse existing disjoint development splits
  where available; establish an avGFP development split excluding the supertest
  if needed. Avoid repeatedly selecting changes using the already inspected
  supertests. Reserve new tasks for later confirmation.

Decision: keep a change only when its development benefit repeats across
replicates at an acceptable cost. A +0.03 Spearman gain would be interesting,
but is not a required or promised outcome. A verified correctness fix is
valuable even without a positive accuracy change.

## 4. Experiment hygiene needed for these comparisons

- [x] Refresh GB1/avGFP summary CSVs from completed raw results before comparing
  new runs. The checked-in avGFP summary is stale relative to scratch results.
- [x] Remove silent validation-to-test fallback in `prepare_frames`; missing
  validation must fail clearly for model-selection runs.
- [x] Fit rank/magnitude transformations on training labels for objectives
  that use target magnitudes. Plain pairwise ordering is invariant to a
  monotone rank transform; do not describe every old rank-loss run as leaked.
- [x] Preserve raw experiment scripts under `experiments/`, per `CLAUDE.md`.
  Include code revision, dataset/split hashes, head choice, adapter targets,
  precision, and optimizer groups in results.

Deferred: expanding the active residue set, learned context tokens, new
pretraining objectives, and a wholesale model redesign. Reconsider only after
the implementation, LoRA, and readout diagnostics above.

## 5. First experiments to run, in order

This queue is the recommended next work, not a request to rerun all historical
learning curves. Complete each gate before launching the next expensive stage.
Do not run a Cartesian product of head, precision, rank, targets, and schedules.

### E0 — Reconcile results and freeze the development protocol

**Question:** What has already run, and what comparisons remain missing?

- [x] Refresh summaries from completed `results.json` files in
  `/home/andrew/mutant_cache_scratch/runs/`, retaining replicate counts and
  distinguishing medians from means. Preserve old snapshots/provenance.
- [x] Record the existing minimal full-backbone test medians as historical
  context: GB1 0.8294/0.9176 and avGFP 0.4418/0.5606 at nominal budgets
  320/1,280, respectively. These are NOT new development-set targets.
- [x] Inventory completed tuning configurations and reuse results only when
  code, split, precision, loss, and selection protocol match. In particular,
  GB1 minimal LoRA/full-backbone development comparisons already exist.
- [x] Primary diagnostic task: `metl_avgfp_dms`, budget 1,280. Secondary budget:
  320. Regression task: `metl_gb1_dms`, the same two budgets. Start with
  replicate 0 for smoke/calibration, then replicates 0, 1, 2 for comparisons.
  These replicate IDs change both sampled data and historical training seeds;
  label them as data/training replicates, not purely optimizer seeds.
- [x] Preserve released METL train/validation memberships. Expected counts
  under the existing 80/20 convention are 256/64 and 1,024/256; assert actual
  unique-genotype counts from the manifests rather than assuming them.
- [x] Reuse GB1's existing `*_tune_ds{n}_r{r}` manifests. Construct an analogous
  fixed avGFP development holdout from genotypes outside the supertest AND
  the union of train/validation rows of all replicates/budgets in this queue.
  Target 5,000 genotypes, or use the available pool if smaller and report its
  size. Pin seed and hashes; do not filter by outcome or model performance.
- [x] Assert no genotype/replicate overlap within each split, validate reference
  sequences and mutation reconstruction, and record depth/label distributions.
  If no adequate avGFP holdout remains, use validation for development and
  explicitly report that there is no additional development holdout.
- [x] Choose checkpoints by the released validation set only. Score the common
  development holdout after checkpoint selection, not every optimization step.
  Historical harness files label this holdout `test`; reports must label it
  DEVELOPMENT. Architecture choices made using it make it development data.
  Do not use METL supertests to choose changes in this queue.

**Deliverable:** one run inventory, four task/budget manifests with three
replicates each, and a protocol record. Stop for missing/overlapping partitions;
never silently fall back to the real test set.

### E1 — LoRA correctness and numerical diagnostics

**Question:** Is there an implementation defect or precision issue worth fixing
before interpreting LoRA accuracy?

- [x] Complete the relevant section-1 checks using existing tests first.
- [x] On a tiny deterministic batch containing singles and multi-mutants, run
  20 optimizer steps with the original minimal head, rank loss, and rank-8
  adapters. Compare variant-only, shared-only, and combined adapters. Use fp32
  first, then the supported bf16-base/fp32-adapter configuration.
- [x] Compare ragged execution against the clamped oracle at initialization
  and after nonzero updates; log projection-level delta norms, parameter
  updates, loss, and reference-cache behavior. A tiny-batch loss decrease is
  an optimization sanity check, not generalization evidence.
- [x] If precision diagnostics show a problem, compare fp32 adapter arithmetic
  as an isolated implementation experiment. Do not promote it merely because
  its gradients are numerically different; measure downstream effect and cost.
- [x] Fix only reproduced issues, add focused regression tests, and repeat the
  failing diagnostic. Maintain a distinction between correctness fixes and
  optional numerical/optimization variants.

**Gate:** proceed when intended parameters update, frozen parameters do not,
and reference/gradient semantics are correct. Existing nonzero-adapter oracle
tests passing is evidence against a basic missing-LoRA-path explanation.

### E2 — Establish avGFP's optimization baseline and test LoRA scope

**Question:** Can straightforward training/adapter choices improve the current
avGFP result without changing the head?

Hold the original minimal mean-pool head fixed. Start on avGFP, budget 1,280.
Use ESM-C 300M, sparse ragged execution, fp32 for the initial comparison,
uniform pairwise rank loss (sigma 1, margin/min-gap 0), head LR 3e-3,
AdamW weight decay 0.01, gradient clip 1, batch size 256, accumulation 1,
and the existing cosine/warmup semantics. Record resolved optimizer settings.
Use raw `score` as the ranking target (ordering is unchanged), avoiding a
dependency on globally normalized target magnitudes.

| Arm | Trainable backbone components | LR to compare |
| --- | --- | --- |
| F | Full backbone | 3e-6, 1e-5, 3e-5 |
| V | Variant-only LoRA, r=8, alpha=16 | 1e-4, 3e-4, 1e-3, 3e-3 |
| S | Shared-only LoRA, r=8, alpha=16 | 1e-4, 3e-4, 1e-3, 3e-3 |

The 3e-3 adapter rate retains the historical minimal-LoRA setting as an anchor.
Use the default `q,o,gate,up,down` targets for both adapter arms. Shared-only
LoRA changes the reference and must pay its recomputation/backpropagation cost.
It still uses the sparse architecture, not full mutant-sequence execution.

- [x] Reproduce F at backbone LR 1e-5 on replicate 0 and calibrate runtime.
  Use 400 optimizer steps for this diagnostic queue. Derive epoch count from
  actual batch counts or add a logged max-step stop; do not silently inherit
  different total step counts from the old epoch clamps. Record any difference
  from historical schedules rather than calling it an exact reproduction.
- [x] Screen the 11 arm/LR combinations on replicate 0. Reuse the anchor run.
  Evaluate validation at a fixed 20-step cadence and the final step; restore
  the checkpoint with lowest validation rank loss, matching the earlier
  selection criterion. Log validation Spearman without changing the criterion
  after viewing results.
- [x] Select one LR per arm by validation loss on replicate 0, then run those
  three settings on replicates 1 and 2 (six additional runs). Compare paired
  development Spearman, train/validation curves, best steps, and cost.
  This is a development screen, not a statistically independent final claim.
- [ ] If all arms are still improving near the endpoint, extend all three
  selected arms to a separately specified 800-step schedule with equal access.
  If training fit is good but validation falls, do not solve that by blindly
  increasing training duration or adapter rank.
- [ ] If adapter runs cannot fit even the training set while F can, examine
  update/gradient diagnostics first. Only then test rank 16 at fixed alpha/r=2
  or all-projection targets, changing one variable at a time. Defer combined
  shared+variant adapters unless the individual-stream comparison motivates it.

**Gate:** pick an accuracy/cost reference configuration for the head experiment.
Retain F as the historical anchor even if V or S becomes preferable. A shared
LoRA gain would suggest adapting reference representations is useful; it would
not establish that frozen-reference LoRA is buggy.

### E3 — Isolate the prediction-head change

**Question:** Does a small readout change help after basic optimization issues
are controlled?

On avGFP budget 1,280, use the selected E2 training configuration and exactly
the same split replicates, schedule, loss, and checkpoint criterion. Do not
retune backbone/adapter settings separately for each head in this first pass.

| Head | Input to prediction network | Purpose |
| --- | --- | --- |
| H0 | Mean of mutant states | Existing baseline |
| H1 | Mean + log1p(k) | Isolate explicit mutation count |
| H2 | Sum + mean + log1p(k) | Test accumulation-aware readout |
| H3 | Section-2 WT-relative token projection, sum + mean + count | Test matched WT/mutant information |

- [x] Implement and validate opt-in choices from section 2. Use 128 hidden
  units for the final prediction MLP; use a 128-wide token projection and
  16-dimensional ref/alt residue embeddings for H3. Record active parameter
  counts, since equal hidden width does not mean equal parameter count.
- [x] Run H1/H2/H3 on replicates 0, 1, 2 (nine new runs); reuse compatible H0
  results from E2. Initialize the pretrained backbone identically across heads.
- [x] If H3 wins, add a mutation-only token-projection control of comparable
  parameter count to check whether extra head capacity, rather than WT
  information, explains the gain. Preserve the same pooling and training setup.
- [ ] Recognize that sum = k * mean for k>0: H2 supplies a useful computation
  explicitly, not new information beyond mean plus count. H3's difference
  feature is likewise redundant given both states; interpret it as an
  inductive-bias choice, not an independent biological signal.

**Gate:** retain the simplest head with a repeatable development benefit.
An aggregate gain driven only by mutation count must be described that way.
If all changes are within replicate variation, keep H0 rather than adding
complexity by default.

### E4 — Verify the chosen change across data regimes

- [x] Freeze the selected E2 settings and E3 head. Compare H0 against the chosen
  head on avGFP-320, GB1-320, and GB1-1,280, using replicates 0, 1, 2: at most
  18 new runs, less when a fully compatible baseline already exists.
- [x] For this transfer check, copy the selected settings without retuning;
  describe it as robustness to dataset/label-budget changes. If later tuning
  is needed, give H0 and the new head equal validation search access.
- [x] Require paired results on the same genotype splits. Show every replicate
  and depth bucket, not only the best seed or a pooled mean. Three replicates
  are a pilot; do not present them as a strong cross-protein significance test.
- [ ] After a LoRA configuration is competitive in fp32, compare its intended
  bf16-base deployment configuration on the same cases. Confirm the accuracy
  and memory benefit together; do not assume precision is cost-free.

**Gate:** a useful improvement should repeat on avGFP and avoid a material GB1
regression. As provisional development criteria, investigate a paired mean
gain >=0.02 (>=0.03 especially encouraging), require gains in at least two of
three replicates, and flag GB1 losses >0.01 for review. These are triage rules,
not confidence intervals or publication thresholds. Report the entire result.

### E5 — Conditional controls, only after E2–E4

- [ ] If avGFP remains weak, independently train the original minimal head
  through full `exact` execution with full-backbone training, compared against
  sparse full-backbone training. Start at budget 1,280, three replicates, with
  identical examples, precision, loss, and effective batch/pair construction.
  Gradient accumulation alone does not reproduce cross-microbatch RankNet
  pairs; if memory requires smaller batches, match them in both arms.
- [ ] Give both backends the same validation LR-search opportunity before
  concluding there is an architectural accuracy gap. Same-weight backend
  switching is not a substitute. Checkpoint selection uses validation only.
- [ ] If exact substantially improves but sparse does not, investigate sparse
  execution restrictions. If both remain weak, prioritize optimization/data/
  readout diagnosis. Defer extra context tokens and larger architecture changes.
- [ ] Before a publication-scale sweep, add a cheap WT-feature mutation-set
  predictor and one-hot ridge with training-only preprocessing. This establishes
  whether adaptive sparse transformer computation is useful beyond cached
  features. It need not delay correctness fixes or the first head comparison.

### Common measurements, cost limits, and deliverables

- [ ] For every run save selected-checkpoint predictions keyed by genotype,
  raw score, mutation depth, split, and replicate. Save enough configuration
  and timing metadata to reproduce the result.
- [ ] Report development Spearman overall and at depths 1/2/3/4/5+, sample
  counts, train/validation rank loss, selected step, active parameter count,
  and head/adapter/base update diagnostics. Mark constant-label or undersized
  depth bins undefined rather than manufacturing a correlation.
- [ ] Report cold total wall time (loading/preparation/compilation/training/
  validation), steady training throughput, peak allocated and reserved GPU
  memory, and reference-cache size. Keep hardware and competing GPU load fixed.
  Record development scoring cost separately and report total tuning cost.
- [ ] Calibrate with E2's first run before committing GPU-hours. E2 is 17
  training runs including its anchor, E3 adds nine, and E4 adds at most 18:
  up to 44 primary runs before conditional controls, extensions, or H3's
  capacity control. Scale the queue from measured cost; stop between stages
  when evidence does not justify expansion. Do not invent a calendar estimate.
- [ ] Produce `experiments/IMPLEMENTATION_PILOT_RESULTS.md` with the run matrix,
  paired accuracy/cost tables, unresolved defects, and a clear keep/reject/
  investigate decision for each change. Refresh existing summaries separately;
  never mix historical supertest metrics with new development metrics.

The immediate launch order is E0 -> E1 -> E2's avGFP anchor and adapter screen
-> E3's isolated head comparison. Complete these before new datasets or a
large rank/target sweep. Implementing this TODO and launching experiments are
separate tasks; this document records the plan only.
