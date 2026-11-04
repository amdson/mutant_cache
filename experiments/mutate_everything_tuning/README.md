This is the entry point for the Mutate Everything tuning program.

**[Live status and results](STATUS.md)** · **[Exact experiment matrix](plan.json)**

The queue runs on physical GPUs 0 and 2. GPU 1 is left to its existing job.
There are 18 seed-0 screening fits and up to four confirmation fits. Jobs run
sequentially within each GPU; later stages wait for their prerequisite study.
This is a bounded first pass, with three sampled-row exposures per ordinary
fit and six only in the explicit longer-budget arms. Runtime will depend on
the selected parent recipe; shared-reference adaptation is more expensive.

Every trial has an ID, one stated question, a resolved parent, exact arguments,
source/data hashes, live progress, a training log and separate selected
checkpoints. Artifacts live outside the repository at
`/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1/`.

| Stage | IDs | Question and controls |
| --- | --- | --- |
| A | A00–A03 | 2x2 head/adapter learning-rate screen: head {1e-3,3e-4}, adapter {3e-4,1e-4}. A00 is the fixed-budget control. |
| B | B01–B05 | Isolate Huber, positive-double weighting, double subsampling and task weighting; then test their combination. |
| C | C01–C03 | Singles-first curriculum at the original budget; longer-budget mixed and curriculum arms form a second pair. |
| D | D01–D02 | WT-relative readout and a mutation-projection control, compared with the inherited pooled head. Parameter counts are recorded, not assumed equal. |
| E | E01–E02 | Rank-16 variant adapters versus rank-8 variant plus rank-8 shared adapters. Both double adapter count; only the latter trains reference representations. |
| F | F01–F02 | Measured reversals versus measured-background augmentation. Both use a bounded training-only reference bank. |
| G | G_control/selected_s1/s2 | Repeat A00 and the selected recipe with seeds 1 and 2, retaining seed 0. Identical selected/control recipes share confirmation jobs. |

Stages B, C and D inherit the best validation recipe from their prerequisite,
including the unchanged parent as a candidate. A treatment does not displace a
better control. E and F branch from D independently; this avoids automatically
combining shared-reference gradients with the extra augmented references.
The best recipe across these studies is frozen before G starts. All fits
initialize from pretrained ESM-2; parent inheritance transfers hyperparameters,
not trained weights. Curriculum is performed within one continuous training
run, with one optimizer and schedule.

Selection and budget:

- Primary checkpoint selection: **macro all-depth validation Spearman**, then
  double Spearman, then the earliest step. Stage promotion compares primary
  scores, then lower macro double RMSE, then ID for a deterministic tie.
- Every trajectory separately saves the best checkpoints for double RMSE,
  all-depth Spearman, double Spearman and double nDCG@30. nDCG uses the
  reproduced comparator definition; undefined retrieval criteria are skipped.
- Screening uses a fixed training budget with early stopping disabled. That
  lets checkpoint-selection rules be compared on the same trajectory.
- Validation is the existing 12-protein holdout. Sampling remains one protein
  per microbatch, four microbatches of up to 64 rows per optimizer step.
- Three exposures mean up to 2,250 updates here, not three passes over every
  protein's full label set as in the upstream epoch definition.
- The older three-seed sweep remains a historical untuned control. A00 is
  repeated because this program changes the budget/selection protocol.

Balancing and augmentation definitions:

Canonical scores are positive stabilizing. B02 applies 4x weight to doubles
with score >= 0. B03 samples at most 8*(number of positive doubles+1) negative
doubles per protein before drawing a microbatch. B04 uses separately normalized
single and double task losses with coefficients 0.1 and 1. B05 combines these
with Huber delta=1. These adopt upstream objective ingredients in our sampled
training regime; they do not reproduce its measured-single residual decoder
or its requirement for complete constituent-single measurements.

F01 replaces 25% of microbatch rows with inverses of measured training singles.
F02 re-references a microbatch to a measured single-mutant background with
probability 0.5. Targets are **measured score(variant) - measured
score(background)**, with an explicit reverse-to-WT example. This can produce
three-substitution training examples from observed doubles. No additive-label
assumption is made. The bank has at most four backgrounds per training protein;
its exact members and transformed-row counts are saved. Augmentation is off
during a singles-only warmup. Validation sequences and labels are unchanged.

The tuning queue cannot run cDNA2, S669, ProTherm or other external tests.
Their already-observed results remain exploratory context, not inputs to
promotion. `frozen_recipe.json` records the selected configuration before the
confirmation seeds. The final test comparison is a separate explicit step.

Operational details:

- `registry.json` is the machine-readable record; `summary.csv` is a flat export.
- Each `runs/<ID>/attempt_N/` contains `resolved_experiment.json`, `train.log`,
  `progress.json`, live `history.csv`, `training_protocol.json`,
  `selection_checkpoints.json`, `checkpoints/*.pt`, and final `results.json`.
- Reference-bank trials also save `training_reference_bank.csv`.
- A failed run retries once in a new attempt directory. Errors remain visible;
  failed treatments never become a winning recipe.
- The source is snapshotted before launch. Running jobs do not change when the
  working tree is edited. Raw data and split SHA-256 hashes are also recorded.

The launcher is [start_mutate_everything_tuning.py](../start_mutate_everything_tuning.py).
It uses `python -m mutant_cache.tuning init`, followed by two detached workers.
Use `python -m mutant_cache.tuning report --root <run-root>` to refresh the
status manually. Workers update it automatically while jobs run.

Future sweep initializations support [W&B tracking](../WANDB_TRACKING.md),
including a [preset for the project](../wandb_tracking.json).

Launch validation (2026-11-04): numerical loss/augmentation checks, synthetic
training checks and queue/promotion tests passed. Three additional real 650M
bf16 preflights passed: balanced Huber with a singles curriculum, shared
adapters with the WT-relative head, and measured-background augmentation.
The exact preflight script is
[mutate_everything_tuning_preflight.py](../mutate_everything_tuning_preflight.py).
The full queue launched at 22:43 UTC; both initial runs reached 50 optimizer
updates with finite training losses.
