# Paper intent and experiment plan

**Scope clarification:** the cache is a fine-tuning architecture. Its primary
evaluation compares separately fine-tuned models, each trained and evaluated
through its own architecture. Out-of-the-box quality and same-weight
cache/exact agreement are not success requirements. Fixed-weight backend
switches below are optional representation-shift diagnostics. See
[PAPER_FINDINGS.md](PAPER_FINDINGS.md) for the updated concise framing.

Literature review: 2026-11-01. Three gpt-6-luna reviewers covered stability,
fitness benchmarks, and efficiency/implementation; parent synthesis checked
repository notes, current accuracy CSVs, and primary-source web results.
This is a research proposal, not a report of new experiments or a guarantee
of publication. Numerical gates below are proposed decision thresholds.

## Recommended paper question

**Can reference-cached sparse mutant updates retain the predictive accuracy
of full mutant-context PLM adaptation while substantially reducing the cost
of training and screening mutation libraries?**

Working title: *Sparse mutant-context adaptation of protein language models*.

The contribution should be the accuracy–compute frontier, with an explicit
domain of validity in mutation density and sequence length. Better epistasis
prediction is a possible second contribution, conditional on experiments.
Do not make a generic claim that fast multi-mutant prediction is new.

The implemented model caches reference activations and updates variant
positions through the backbone, then uses additive token scores and an
optional interaction head. This is an approximate mutant-context model.
Variant-only adapters preserve reusable reference states; shared adapters or
base-weight training require reference recomputation during training.

## What the literature changes

| Primary source | Relevant evidence | Consequence |
| --- | --- | --- |
| [Mutate Everything, NeurIPS 2023](https://arxiv.org/abs/2310.12979) | Shared-reference parallel decoding already supports singles, doubles, and higher orders. PTMul higher-order evaluation uses a homology constraint. | Compare against WT-only decoding as well as full mutant passes. Distinguish enumerating all doubles from scoring a supplied sparse library. |
| [Dieckhaus & Kuhlman, Protein Science 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11659742/) | Additive predictors are highly competitive; improvement in total double-mutant stability does not establish accurate epistasis. Matched measured singles are available only for a subset; double-mutant pairs are spatially biased. | Evaluate interaction residuals and coverage, and bootstrap by protein. Report stabilizer selection separately. |
| [ESM-MSR, 2026 preprint](https://www.biorxiv.org/content/10.64898/2026.06.04.730231v1.full) | Distinct WT/mutant LoRAs explicitly encode mutant context and improve multi-mutant prediction. Uses ESM3 with structure inputs and a full mutant path. | Closest conceptual competitor for contextual epistasis. Its expensive mutant path motivates our compute question; cross-model speed ratios alone cannot isolate caching. |
| [FLIP2, ICML 2026](https://flip.protein.properties/) | Seven additional datasets and challenging distribution shifts; simple models often match or outperform fine-tuned PLMs. | Frozen-feature and one-hot baselines are mandatory. Random-split gains alone are weak evidence. |
| [CombinGym repository](https://github.com/sitonglab/CombinGym) | Higher-order prediction from lower-order mutants; Spearman and NDCG benchmarks. Repository and website counts may differ. | Pin a commit and use released splits, not reconstructed approximations. |
| [ProteinGym, NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/file/cac723e5ff29f65e3fcbb0739ae91bee-Paper-Datasets_and_Benchmarks.pdf) | Supervised random, contiguous-position, and modulo-position CV answer different questions. | Label adaptation and generalization settings explicitly; avoid mixing with zero-shot leaderboards. |
| [METL, Nature Methods 2025](https://www.nature.com/articles/s41592-025-02776-2) | Biophysics pretraining followed by DMS adaptation; released label-budget splits. | Useful optional transfer benchmark, but match objective, prior data, and split before claiming parity. |
| [ESMRank, Molecular Systems Biology 2026](https://link.springer.com/article/10.1038/s44320-026-00252-3) | Cross-protein MAVE-supervised ranking, with strongest transfer on stability-related readouts. | Broad MAVE-wide generalization is a crowded, separate project. Do not start with the entire MaveDB dump. |

ESM-MSR is a preprint; its structures and ESM3 backbone differ from sequence-only
ESM-C. It is a task comparator, not a controlled cache baseline. Its exact
epistasis numbers and supplementary split artifacts need verification before
setting a numerical state-of-the-art target. Similarly, confirm the current
ProteinGym/CombinGym versions before freezing manifests.

## What the repository currently establishes

- `HANDOFF_2026-10-24.md` reports large A6000 throughput and memory benefits.
  These were not independently rerun for this review. They justify measuring
  an end-to-end frontier, not claiming a final benchmark win.
- `experiments/cache_vs_exact_accuracy.csv` has only three seeds and noisy
  train-through differences. Tsub4 doubles are 0.564 cache-trained/cache
  versus 0.611 exact-trained/exact; the same cache weights through exact give
  0.469. Sparse approximation is not yet established as negligible.
- `experiments/THERMOMPNN_SPLIT_RESULTS.md` reports validation-only,
  single-seed results. Its strict split and pooled/macro distinction are
  valuable infrastructure; published test numbers cannot be compared to our
  validation scores as a head-to-head.
- The newer untracked avGFP minimal CSV has one replicate at 10/20/40 labels:
  approximately -0.019/0.192/0.306. Its comparison columns put METL-Local much
  higher, but their numeric provenance still needs checking. This is a warning
  against a few-shot superiority thesis, not a definitive model ranking.
- METL Rosetta total-score results use a different objective from METL's
  multi-task pretraining. Do not use the reported 0.955 as a METL win.

The handoff's statements about a plateau proving a capacity limit or a
parameter increase ruling out capacity are hypotheses, not identified causes.
Noise, optimization, regularization, model bias, and sample diversity remain
possible explanations. Absence of a significant gap is not equivalence.

## Three experiments, in priority order

### A. Primary claim: accuracy at a fraction of the compute

Use frozen-reference ESM-C 300M first. Compare cache and exact backends with
the same scorer, loss, trainable parameter configuration, data and validation
search budget. Include head-only and variant-LoRA configurations separately.
Use a conventional full-sequence LoRA baseline to test practical usefulness;
identify its different reference-update semantics explicitly.

Initial panel: strict ThermoMPNN Megascale protein split; BRCA1 as a longer
sparse within-assay case; FLIP GB1 2-vs-rest as a dense stress test; avGFP
as a longer multi-mutant case. Pin the precise BRCA1 assay and report its
measurement floor. BRCA1 need not become a headline accuracy benchmark if
all trained methods remain weak. Expand to all eligible CombinGym landscapes
only after the pilot works, reporting every selected task and any exclusions.

Use two different comparisons:

1. **Retrained quality frontier:** cache-trained/cache versus exact-trained/
   exact, with independently validation-tuned configurations. Compare both
   equal training examples/updates and equal GPU-hour budgets.
2. **Fixed-weight fidelity:** score each trained checkpoint through both
   backends. This diagnoses representation shift; it does not substitute for
   retraining. The `recompute` backend implements clamped cache mathematics
   and is not an exact full-mutant control; use `exact`.

Proposed primary gate: within 0.02 macro Spearman of exact-trained/exact on
the preregistered sparse subset, while using at most 10% of its end-to-end
training time and at most 25% of peak GPU memory. Demonstrate noninferiority
with a paired interval: the lower confidence bound for cache-minus-exact
must exceed -0.02, rather than accepting a nonsignificant difference.
Define sparse before test access (initially mutation density <=1%, with a
separate 1–5% stratum). Also report all densities; do not select the supported
range after inspecting test results. Revise gates using pilot validation only.

Report cold-start and amortized costs, reference encoding, compilation,
cache storage/transfers, training+validation, and scoring at library sizes
100/1k/10k/100k. Optimize batch size per method under the same memory limit;
also show equal-batch diagnostics. Include frozen-feature ridge/MLP feature
extraction in total cost. Sweep L and k separately, not only k/L. A model
that is fast only after scoring millions of variants needs that qualification.

### B. Secondary claim: useful higher-order extrapolation

Use released CombinGym 2-vs-rest and 3-vs-rest splits, with FLIP GB1 as a
sanity check. Keep the benchmark's validation convention; no test labels in
hyperparameter selection. Report depth-stratified Spearman and NDCG using
the released metric implementation. Macro-average over proteins first so
multiple landscapes/antigens do not inflate the independent sample count.

Baselines: one-hot ridge; additive predicted-single model; additive model
with fitted monotone assay link (global epistasis); pairwise regularized
model where affordable; WT-feature interaction MLP; frozen ESM-C feature
ridge/MLP; exact contextual scorer. Add matched released benchmark methods
where reproducible. Mutate Everything is most appropriate on stability tasks.

Proposed stronger gate: >=0.03 macro Spearman over the strongest cheap
validation-selected baseline on higher-order variants, paired protein-level
95% interval above zero, while retaining a substantial compute advantage
over contextual recomputation. This is an ambition, not a literature-derived
publication cutoff. If quality ties strong cheap models at higher cost,
speedup over an unnecessarily expensive baseline is insufficient.

### C. Optional biological claim: interaction prediction on unseen proteins

Use released Megascale-D folds/splits from
[Dieckhaus & Kuhlman data](https://doi.org/10.5281/zenodo.13345274), or the
ESM-MSR split if its artifacts and exact exclusion rules are obtainable.
Never compare numbers from different partitions as direct competitors.
Train on singles AND doubles of training proteins; hold out protein clusters.
Keep all modified-WT backgrounds/duplicates with their source protein.
Audit local alignment and domain homology across folds, beyond assigning
cluster labels alone. Reserve PTMUL-D as an external confirmation after
excluding homologs of its proteins from training and model selection.

For raw folding free energy G define epsilon = G_AB - G_A - G_B + G_WT.
For consistently WT-relative delta-delta-G labels it becomes y_AB-y_A-y_B.
Use matched experimental singles to construct the evaluation target, but
do not feed held-out labels to the deployable model. Compare predicted
epsilon formed from model double/single predictions, and residual error
against a predicted-single additive baseline. Measured-single addition is
an informative privileged baseline, not a noise-free oracle or ceiling.

Report epsilon Pearson/Spearman, error, total double-mutant Spearman, and
stabilizing-mutant retrieval separately. Stratify contact/distance where
structures are available and report complete-case coverage. A constant-zero
epsilon predictor has zero variance: its correlation is undefined, so compare
its error rather than assigning it a correlation. Propagate uncertainty only
with justified assumptions about shared WT/single errors and covariance.

Proposed gate: a positive protein-bootstrap interval for improvement over
learned additive/global-epistasis and WT-interaction baselines on both epsilon
error and double-mutant ranking. An epsilon correlation around 0.3 would be
interesting, but cannot be treated as beating published 0.24–0.25 estimates
without matching the data, filters and aggregation exactly.

## Splits and inference discipline

- Assign repeated measurements by canonical genotype; keep duplicate score
  sets, WT backgrounds and same-protein assay duplicates together where needed.
- Separate within-assay adaptation, unseen mutation positions, deeper mutants,
  and unseen proteins. They support different claims.
- Fit normalization/link/calibration only on training labels. For unseen assays,
  use a declared unknown-assay embedding or assay metadata; an untrained new
  assay-ID embedding cannot support a cross-protein transfer claim.
- Report macro per-protein and pooled correlations with their exact definitions.
  Bootstrap proteins/families, nesting assays where necessary; random seeds
  quantify training variation, not independent biological sample size.
- Reserve test access for the frozen plan. Existing validation results are
  development data. Label any already-used tests as exploratory, and use
  new held-out tasks/folds for confirmation where needed.
- Rank normalization preserves ordering but does not remove nonlinear global
  epistasis. Singles-only training does not identify genuine interaction terms
  without transferred interaction supervision or a justified prior.

## Execution and decision sequence

1. **Protocol freeze:** dataset hashes/manifests, genotype deduplication,
   reference/background grouping, train/val monitoring, unknown-assay behavior,
   exact backend audit, and external comparator provenance. No new model work.
2. **Pilot A:** three seeds on the four-case panel, modest identical validation
   search, end-to-end timing. Use validation to choose schedule/LoRA scope.
   Stop few-shot METL expansion until its weak pilot and comparison setup are
   understood. Do not start MaveDB-wide training yet.
3. **Confirm A + B:** five seeds for selected settings, fixed test manifests,
   all eligible chosen benchmark tasks, protein-bootstrap intervals, accuracy
   versus GPU-hour plots and density failure boundary.
4. **Attempt C only if justified:** interaction-supervised training and
   matched stability comparators. Treat external PTMUL as confirmation, not
   a small dataset on which to tune the model repeatedly.

Budget the pilot in measured GPU-hours after a short calibration run; the
available notes do not justify a reliable calendar/compute estimate for the
entire matrix. Cap the pilot before scaling. Required paper figures: method
diagram; end-to-end accuracy–compute frontier; density/length failure map;
higher-order results against cheap baselines; optional interaction residuals.

Decision: pursue a methods paper if A succeeds across diverse proteins and
the method remains competitive with cheap alternatives. Add an extrapolation
claim if B succeeds. Add an epistasis claim only if C succeeds. If gains occur
only against full recomputation while a cheap additive/frozen-feature model
matches quality and cost, narrow to an implementation contribution or revisit
the intended application before committing to a paper.
