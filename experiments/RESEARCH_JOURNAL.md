# Research journal

Concise reminders of experiments, results and decisions. Append dated entries as work progresses; record whether scores are internal validation or external test results. Exact configurations, histories and provenance live in the linked experiment programs and scratch run directories.

**Goal:** a sparse variant-cache stability model directly evaluated against Mutate Everything on its released benchmarks, with matched training data, test rows and metrics. Differences in architecture and training strategy remain explicit.

## Retrospective entries: 2026-10-16 through early November

Added on 2026-11-05 from the earlier work with Claude: git history, dated handoffs, raw scripts and result tables. Dates below are source-document dates or explicit ranges, not newly invented run timestamps. Some work was committed days after the handoff describing it. These are historical findings and dataset inventories, not claims about today's code or current benchmark performance.

### 2026-10-16–17 — Bootstrap, downloads and cache go/no-go

- Established the module layout, scratch-data convention and implementation/oracle plan; ported ESM-C and the bidirectional sparse variant cache. Built the first model, losses, baseline methods and shared training/evaluation harness.
- Downloaded Tsuboyama stability data and the MaveDB dump from Zenodo; downloaded FLIP GB1 and ProteinGym reference metadata. The old FLIP hosting URLs were dead, so the raw script switched to the FLIP repository. Checksums were recorded. The initial download did **not** include the full ProteinGym score archive or CombinGym.
- Phase-0 experiments compared sparse and full-mutant ESM-C 300M forwards on GB1, ubiquitin, avGFP and PTEN, across mutation count, length and precision. The clamped recompute oracle verifies implementation; it is **not** the full-mutant model and cannot measure approximation cost.
- Runtime example on one A6000, bf16, 256 mutants/reference, including reference amortization: length 256, one substitution **326 → 5,167 mutants/s (15.9×)**; length 512 **159 → 5,074 (31.9×)**. Speedups shrink at higher depth and with few variants per reference. These are specific early ESM-C workloads, not a speed comparison with Mutate Everything.
- Initial verdict: proceed. Low-depth masked-marginal rankings largely agree with full forwards, but approximation errors increase as more reference context becomes incorrect.

Sources: commits `280752c`, `be93e86`, `52552cc`; [raw downloads](../archive/2026-11-05/experiments/phase2_download.sh); [phase-0 results](PHASE0_RESULTS.md), including throughput grids and scripts.

### By 2026-10-21 — Data layer, numerical fixes and clustered-mutation warning

- The dated handoff records processed Tsuboyama **550,141 variants**, MaveDB **6,112,263**, and FLIP GB1 **158,094** (8,733 in its downsampled subset). Built depth/assay metadata, splits, ProteinGym overlap exclusions and **131,143** Tsuboyama double-mutant groups with both constituent singles. These are the historical general-purpose datasets, not the later 116-protein Mutate Everything training release.
- Fixed a residue-to-token off-by-one: mutations had hit the neighbouring residue. Tsuboyama zero-shot Spearman changed **0.009 → 0.343**. Also corrected the terminal token, bf16-to-fp32 head boundary, GPU-to-pandas indexing and split lookup. Added regression checks against the actual substituted tokens, not just agreement between two implementations sharing the same mistake.
- Wired variant-only and shared LoRA. Established frozen bf16 base / fp32 adapters and heads, with fp32 correctness oracles.
- A clustered-mutation follow-up qualifies the original density story: **local layout matters**, not just global mutation fraction. PTEN at ten substitutions falls from cache/full LLR Spearman **0.993 scattered → 0.879 contiguous**; the real four-site FLIP GB1 layout gives **0.845** at depth four. Keep full-mutant controls for dense/clustered workloads.

Sources: [October 21 handoff](../archive/2026-11-05/HANDOFF.md); fixes/adapters in commits `eeb670f`, `4037d0d` (October 20); clustered follow-up in [phase-0 results](PHASE0_RESULTS.md). The follow-up's exact execution date is not separately recorded here.

### 2026-10-21 — First stability benchmark on ThermoMPNN's split

- Imported ThermoMPNN's released Tsuboyama split and a separately named stricter exclusion variant. Used frozen ESM-C 300M, seed 42; **all reported outcomes were validation**, with test untouched.
- Strict split, singles: two-epoch head-only macro Spearman **0.7280**, versus variant LoRA **0.7197**. Singles+doubles, one epoch: doubles macro Spearman **0.6440** head-only versus **0.6304** with LoRA; pooled doubles results differ substantially. No clean adapter win.
- Recorded important interpretation problems: pooled versus macro metrics, label sign/reference conventions, per-assay parameters on unseen proteins, and changing cosine schedules when changing epoch budgets. A trainable-base reference-cache defect was identified; its later fix belongs to the October 29 work below.
- Decision: first establish fair splits, additive controls and correct training/reference behavior; do not interpret the initial head-only results as demonstrated interaction learning or a paper-level win.

Source: [ThermoMPNN split results](../archive/2026-11-05/experiments/THERMOMPNN_SPLIT_RESULTS.md), explicitly dated October 21.

### 2026-10-24 work, committed October 27 — Ragged execution, fused attention and METL

- Added ragged variant execution, fused flex cross-attention and a genuine full-mutant `exact` backend. This removed work on padded mutation slots and enabled meaningful cache-versus-full comparisons.
- October 24 handoff: BRCA1 workload **278 → 20,183 variants/s evaluation (~73×)** and **77 → 8,110 training (~105×)**, with training peak memory roughly **23 → 2 GiB**, against a full forward of every mutant. GB1 showed ~29× evaluation / ~42× training. These depend on workload, batch, precision and amortization; they are not universal speedups.
- Added cache-versus-exact supervised controls. Training each backend separately showed no clear gap in the small three-seed pilots; swapping a cache-trained model to exact execution could reduce accuracy. Exact execution is a different learned-feature distribution, not automatically an improvement.
- Added METL Rosetta/DMS loaders and split conversion. Explicitly checked Rosetta's 1-based versus DMS's 0-based mutation numbering, distinct METL/FLIP GB1 references, unique-genotype split assignment and compiler fallback behavior.
- GB1 single-task Rosetta total-score pilot reached median test Spearman **~0.955** near 57,600 training examples, then plateaued. **Not a matched METL win:** METL pretraining uses many Rosetta attributes and much more data. This task mismatch motivated scalar DMS comparisons instead.

Sources: [October 24 handoff](../archive/2026-11-05/HANDOFF_2026-10-24.md); commit `7461239` (October 27); [`ragged_vs_padded_runtime.csv`](../archive/2026-11-05/experiments/ragged_vs_padded_runtime.csv), [`naive_vs_cache_runtime.csv`](../archive/2026-11-05/experiments/naive_vs_cache_runtime.csv), [`cache_vs_exact_accuracy.csv`](../archive/2026-11-05/experiments/cache_vs_exact_accuracy.csv), [`metl_gb1_scaling.csv`](../archive/2026-11-05/experiments/metl_gb1_scaling.csv).

### 2026-10-29 — Minimal model, cache lifecycle fixes and DMS learning curves

- Added the minimal prediction-head preset and fixed reference-state handling when training the base. Earlier shared-only invalidation did not cover a trainable backbone; moved toward caching references only when their generating parameters cannot change.
- Ran METL-style GB1 and avGFP DMS train-size grids and diagnostics of full finetuning versus sparse execution. Saved replicate-level records and summaries rather than selecting one attractive run.
- Minimal-model GB1 median Spearman scales from **0.288 at nominal 10 labels** to **0.918 at 1,280** and **0.943 at 2,560**. avGFP remains harder: **0.704 at 2,560**, **0.836 at 20,480**. Published-target columns in those CSVs are contextual references; training/pretraining and source-number provenance must be checked before treating them as matched wins.
- Lesson: label budget, adaptation scope and readout strongly affect results. Strong performance on simulated Rosetta energies does not establish few-shot experimental fitness performance.

Sources: commits `1bc3ab7`, `2afb59d`; [GB1 table](../archive/2026-11-05/experiments/metl_gb1_dms_h2h_minimal.csv), [avGFP table](../archive/2026-11-05/experiments/metl_avgfp_dms_h2h_minimal.csv); raw DMS launch/diagnostic scripts in this directory.

### 2026-11-01–02 — Implementation audit and controlled head/adapter pilots

- Audited every LoRA projection and both streams against the clamped oracle, including gradients and optimizer groups. Real ESM-C updates confirmed intended parameters move and frozen weights stay fixed. No missing-adapter-path defect was found. Fixed stale reference reuse after checkpoint restoration for callers remaining in eval mode.
- Built development manifests separate from METL supertests. Checkpoint choice used released-validation rank loss; the additional development holdout assessed alternatives. Replicates vary both sampled data and training seed.
- avGFP nominal 1,280-label pilot: mean development Spearman **0.568 full backbone**, **0.605 shared LoRA**, **0.688 variant LoRA**, with the original mean head. All arms selected early checkpoints and subsequently overfit.
- Head pilot: mean **0.688**, mean+count **0.728**, sum+mean+count **0.773**, WT-relative **0.776**. A parameter-matched mutation-only head reached **0.779**, so WT-relative features were not uniquely supported in this task. Chose the simpler sum/mean/count head for subsequent work.
- Transfer check: sum/mean/count improved avGFP-320 by **+0.211** overall, mostly between mutation depths; GB1-320/1280 effects were near zero. Small pilot results did not justify broad conclusions. The later ESM-2 stability success of WT-relative/shared adaptation is a different task and does not retroactively change this finding.

Source: [implementation pilot results](../archive/2026-11-05/experiments/IMPLEMENTATION_PILOT_RESULTS.md), E0–E4 and linked scripts. Its dated status is November 2; the batch was committed in `4271e9a` on November 4.

### 2026-11-02–04 — Stability-transfer groundwork before Mutate Everything

- Created the ESM-MSR stability-transfer and pretrained/sparse-architecture mismatch plans. Ran layer/state mismatch diagnostics (`am0_*`) and ESM-MSR joint/additive learning-rate screens; the preserved driver logs and selection CSV show completed screen runs. Do not read the old plan-only header as proof no later work occurred.
- Established protein-uniform Megascale training, exposure budgets, validation-only selection, additive/epistasis evaluation and separate final-test scripts. Selected checkpoints were often very early; schedule and sampling exposure needed explicit reporting.
- The presence of a final-test script does **not** prove that final ESM-MSR analysis ran. This reconstruction found screen/tuning run directories, but is not recording a completed ESM-MSR final result or a successful distillation/pretraining experiment.
- The next work moved to the separately released Mutate Everything training/test split and an executable comparator reproduction, documented in the entries below.

Sources: [ESM-MSR plan](../archive/2026-11-05/TODO_ESM_MSR_EXPERIMENT.md), [architecture mismatch plan](../archive/2026-11-05/TODO_ARCHITECTURE_MISMATCH_PRETRAINING.md), `am0_*`, [ESM-MSR screen selections](../archive/2026-11-05/experiments/esm_msr_select.csv), screen/tuning driver logs; commit `4271e9a` includes this work by November 4.


## 2026-11-04 — Establish the comparison and initial baseline

- Built the released cDNA data pipeline and reproduced the Mutate Everything ESM-2 comparator. Upstream/local metrics agree; cDNA2 nDCG@30 reproduces the published 0.38. S669 requires the separately released single-only checkpoint to reproduce Spearman 0.47.
- Initial model: frozen bf16 ESM2-650M, fp32 variant LoRA rank 8, sum/mean/count head, WT-zero anchor, joint multi-mutant prediction. About 5.06M trainable parameters.
- Created an internal protein split within the paper's 116 training proteins: 104 train / 12 validation. This split is ours; the paper's cDNA2 test set remains separate (18 proteins). Tuning macro validation Spearman is not comparable numerically with pooled test Spearman.
- Historical three-seed joint fits: cDNA2 pooled test Spearman **0.5422**, versus comparator **0.4539**; cDNA2 nDCG **0.3834** versus **0.3792**. Other-panel Spearman trails: cDNA1 **0.7230 vs 0.7478**, S669 **0.4560 vs 0.4736**, ProTherm Multiple **0.3182 vs 0.3918**. These fits trained on only 104 proteins.
- Implemented joint versus summed-singles scoring and independently trained additive controls. The early cDNA2 gain did not imply broad superiority. Reverse-direction accuracy was especially weak.

References: [comparator reproduction](../archive/2026-11-05/experiments/MUTATE_EVERYTHING_BASELINE.md), [pilot](../archive/2026-11-05/experiments/MUTATE_EVERYTHING_PILOT_COMPARISON.md), [breadth experiments](../archive/2026-11-05/experiments/MUTATE_EVERYTHING_BREADTH_RESULTS.md).

## 2026-11-04–05 — Systematic tuning and regularization

- Completed **22 tuning fits** and **10 regularization fits**. Tested learning rates, Huber/MSE, positive-double weighting, negative-double subsampling, separate singles/doubles losses, singles-first schedules, training budgets, readout features, adapter capacity and shared adaptation, measured reversals/backgrounds, dropout and weight decay.
- Main finding: **WT-relative readout + shared adapters (E02)**. The head uses variant/reference states, their difference and residue identities. Shared LoRA adapts the reference stream as well as variant states. E02 has rank-8 variant + rank-8 shared adapters, about **9.99M trainable parameters**.
- Three-seed **internal validation**: overall macro Spearman **0.8092 → 0.8532**, doubles Spearman **0.6625 → 0.6858**. Doubles RMSE worsens **0.8596 → 0.8906**. Ranking gains survive training seeds on this one protein split; alternate protein splits were not yet tested.
- Most other changes give little benefit. Strong regularization tops out at overall validation Spearman **0.8133** on the original control. Augmentation was tested on the WT-relative variant-only model, not on E02.
- Interesting tradeoff: combined Huber/balancing raises internal doubles nDCG@30 **0.3909 → 0.6914**, while overall Spearman falls to **0.7729**. Only five proteins are eligible; two account for ~87% of the nDCG increase. Precision@30 is unchanged. Keep as a retrieval hypothesis, not a general improvement.
- Added W&B and sampled diagnostics: loss/metrics, fixed train probes, gradient/clipping/update norms, activation and adapter-output norms, calibration/epistasis, sampling exposure and performance. Preserved completed old fits and restarted unfinished ones with tracking.

References: [tuning](mutate_everything_tuning/STATUS.md), [regularization](mutate_everything_regularization/STATUS.md), [diagnostics](DIAGNOSTICS.md).

## 2026-11-05 — Full-training-set matched comparison completed

**Decision:** freeze the original control and E02; train three fresh seeds each on **all 116 training proteins**, using the original fixed budgets (control: 3 exposures; E02: 6). Save final weights, with no early stopping or test-dependent checkpoint selection. The former validation proteins are now training data; their `val/*` metrics are in-sample diagnostics.

**Result:** E02 improves mean pooled test Spearman on all four forward panels. Same canonical rows for both systems; complete coverage. Three-seed means below; comparator is a fixed released checkpoint.

| Panel / metric | Mutate Everything | Full-data control | Full-data E02 |
| --- | ---: | ---: | ---: |
| cDNA2 Spearman | 0.4539 | 0.5225 | **0.5649** |
| cDNA1 Spearman | 0.7478 | 0.7231 | **0.8060** |
| S669 Spearman | 0.4736 | 0.4525 | **0.5018** |
| ProTherm Multiple Spearman | 0.3918 | 0.3510 | **0.4005** |
| cDNA2 nDCG@30 | 0.3792 | 0.4220 | **0.5102** |
| S669 reverse Spearman, uncalibrated diagnostic | 0.1655 | 0.0692 | **0.3537** |

- The cDNA2 retrieval gain is supported by the paired protein bootstrap: mean nDCG difference **+0.1310**, 95% interval **[+0.0408, +0.2199]**, conditional on these fits and eligible proteins.
- ProTherm is uncertain: E02 seed SD **0.0389** exceeds its small mean advantage. Mean-score wins alone are not significance claims.
- Reverse results are uncalibrated diagnostics, not reproduction of the paper's calibrated reverse table. S669 forward uses the single-only comparator; other panels use its joint checkpoint. These are ESM-2 system comparisons, not the paper's AlphaFold results or a causal isolation of caching.
- This round establishes that E02's internal ranking improvement transfers beyond the tuning proteins. It does not remove the documented benchmark homology caveats or establish controlled inference-speed superiority.
- Added a configurable matched-suite runner: drop in checkpoints or canonical prediction CSVs, reject incomplete/mismatched rows, report per-seed and per-protein metrics, bootstrap intervals and hashes. Verified it against the complete historical three-seed suite.

References: [frozen final-fit protocol](mutate_everything_full_comparison/README.md), [comparison runner](mutate_everything_matched/README.md), [completed report](/home/andrew/mutant_cache_scratch/runs/mutate_everything_full_comparison_v1/matched_report/README.md). Run root: `scratch/runs/mutate_everything_full_comparison_v1`.

## 2026-11-05 — Current research direction: E02 accuracy round 2

**Hypothesis:** the remaining useful gains will come from better interaction structure, an objective aligned with ranking, or more reference/backbone adaptation. Further generic regularization looks lower priority.

Implemented and launched a separate screen, starting fresh from pretrained weights on the **104/12 internal split**:

- **Ranking:** MSE + within-protein/assay, within-mutation-depth RankNet; weights 0.1, 0.3 and 1.0.
- **Interaction structure:** sum predicted single effects + a separately parameterized joint residual. Residual starts at zero and is constrained to zero for singles/WT. Measured doubles supervise the total; no measured-single teacher forcing.
- **Shared adaptation:** shared ranks 16/32 with scaling held at 2; separately add shared K/V targets.
- **Partial backbone:** train last 2/4 blocks plus final norm at LR 1e-5, retaining E02 adapters. Use fp32 and an fp32 E02 control to distinguish precision effects.
- **Confirmation:** original-split seed repeats for the selected recipe, followed automatically by E02-versus-selected checks on two additional protein holdouts. No external panel evaluation is scheduled by this queue.

**Status when this entry was written:** E02 control and ranking weight 0.1 are running on GPUs 2/0; remaining treatments are queued. No accuracy results yet. Short real-weight training checks passed for residual, shared K/V and partial-backbone paths; synthetic checks cover residual initialization, singles invariance, gradients, optimizer freezing and training plumbing.

Primary queue: 11 screen fits plus up to four seed confirmations; then four extra protein-holdout fits. Select using internal validation, inspect doubles/RMSE/retrieval tradeoffs, and confirm across protein splits before declaring another external evaluation round. Keep the completed full-data comparison intact.

References: [round-2 plan and monitoring](mutate_everything_accuracy_v2/README.md), [live status](mutate_everything_accuracy_v2/STATUS.md). Run roots: `scratch/runs/mutate_everything_accuracy_v2`, then `_holdout1` / `_holdout2`.

## 2026-11-05 — Repository reorganization (no new measurements)

**Moved:** 144 superseded plans, handoffs, completed reports and their raw scripts/results into [archive/2026-11-05](../archive/2026-11-05/README.md). Each move is in `moves.json` with its pre-move SHA-256 hash, and all were checked against commit `4f74a31`. The only content edits were relative link targets in three archived markdown files. This includes the raw DMS scripts and the `am0_*`/screen driver logs that earlier entries describe as "in this directory". This journal, `docs/VARIANT_EFFECT_NOTES.md`, `INTERFACES.md`, code, tests and the live round-2 directory stayed in place.

**Added:**
- `python -m mutant_cache.train`, which runs one seed per launch. It wraps `megascale` and records the resolved config and provenance.
- Recipes in `experiments/recipes/`. The ESM-2 recipe is the E02 round-2 control base settings; the ESMC recipe has not been run.
- Protocols in `datasets/protocols/` and `make -C datasets mutate-everything | esm-msr`, which reproduce the existing processed paths. They were checked with `make -n` only and nothing was rebuilt.
- Experiment records in `experiments/stability_paper/`.

**Still planned, not implemented:**
- StabilityArc data, import and evaluation. The protocol is marked planned, and both the launcher and the make target reject it.
- A matched ESM-MSR external comparison (checkpoint provenance, row mapping, metrics).
- A general evaluator. `matched_suite` remains Mutate Everything–specific.

Raw make rules don't regenerate the holdout1/2 split files or two audit files.

**Checks:** dry-runs resolve for the ME and ESM-MSR experiment definitions, CLI unit tests pass (7), the ME comparison config validates on CPU, and there are 0 broken relative markdown links. No training was launched.

## Code map

- `mutant_cache/data/mutate_everything.py`: released-data normalization, signs, duplicates, splits and overlap provenance.
- `mutant_cache/models/variant_model.py`, `cache/`, `backbone/`: sparse model, readouts, residual branch and adapter/cache mechanics.
- `mutant_cache/models/loss.py`, `training_protocol.py`, `megascale.py`: objectives, sampling/augmentation, supervised fits, partial adaptation and full-training refits.
- `mutant_cache/tuning.py`: frozen experiment plans, worker queue, promotions, checkpoint summaries and continuation.
- `mutant_cache/tracking.py`, `diagnostics.py`: W&B and sampled local diagnostics.
- `mutant_cache/benchmark.py`, `comparison.py`, `matched_suite.py`, `eval/mutate_everything.py`: frozen-model scoring and exact matched benchmark reports.
- `experiments/`: raw launch scripts and readable plans; scratch run roots contain immutable source/configuration copies and detailed artifacts.

## Next journal update

Record round-2 results and failures, whether changes beat E02 on doubles as well as overall ranking, and whether gains survive the extra protein holdouts. Explicitly distinguish completed measurements from queued ideas and internal validation from external test results.
