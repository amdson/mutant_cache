# Mutate Everything comparison: model and benchmark state

Snapshot: 2026-11-05. This is a scientific status note; run directories contain the exact configurations and live progress.

## Update: full-training comparison completed

The six final fits and matched evaluation completed on 2026-11-05 at 21:22 UTC. These are three-seed means of pooled test Spearman on identical rows:

| Panel | Released comparator | Full-data control | Full-data E02 |
| --- | ---: | ---: | ---: |
| cDNA2 | 0.4539 | 0.5225 | **0.5649** |
| cDNA1 | 0.7478 | 0.7231 | **0.8060** |
| S669 | 0.4736 | 0.4525 | **0.5018** |
| ProTherm Multiple | 0.3918 | 0.3510 | **0.4005** |
| Uncalibrated S669 reverse diagnostic | 0.1655 | 0.0692 | **0.3537** |

E02 cDNA2 nDCG@30 is 0.5102 versus comparator 0.3792. ProTherm E02 Spearman seed SD is 0.0389, larger than its mean gain over the comparator, so that result is particularly uncertain. Mean-score wins alone do not establish statistical significance. Other comparator scope, homology and reverse-calibration caveats below still apply.

Full artifacts: `/home/andrew/mutant_cache_scratch/runs/mutate_everything_full_comparison_v1/matched_report/`. The historical sections below describe the preceding state; this update supersedes statements that E02 external performance is unknown or final fits are running.

## Objective and current conclusion

The project aims to train a sparse variant-cache model and compare its accuracy directly with Mutate Everything on the released benchmark panels. The final comparison uses all 116 released training proteins, identical evaluation variants and upstream metric definitions. It compares complete trained systems; it does not isolate the effect of caching from decoder, adaptation or training differences.

The sparse model already trains and scores the benchmark suite successfully. Earlier three-seed fits outperform the released ESM-2 comparator on cDNA2 pooled Spearman, but trail on several other panels. Internal validation tuning identifies a better recipe—WT-relative readout with both variant and shared adapters—with consistent rank-correlation gains across three seeds. Its external performance is not yet known. Six full-training-set fits are now running, followed automatically by matched evaluation.

## Model

Both the original control and selected model initialize from ESM-2 650M (`esm2_t33_650M_UR50D`, 33 layers, hidden size 1280). Base parameters are frozen in bf16; adapters and prediction heads are trainable fp32. Labels are measured WT-relative stability changes in kcal/mol, with positive values meaning stabilizing.

The cache method constructs per-layer states for a reference sequence, then recomputes variant states at mutated positions. Variant tokens interact with the cached reference context and other variant positions. This is a sparse model approximation: unchanged positions do not undergo a full mutation-conditioned recomputation. The model scores a supplied variant library; Mutate Everything's dense decoder can generate a landscape. Those are different inference workloads.

| Component | Original control A00 | Selected recipe E02 |
| --- | --- | --- |
| Variant adapters | LoRA rank 8, alpha 16 | LoRA rank 8, alpha 16 |
| Shared adapters | None | LoRA rank 8, alpha 16 |
| Adapter targets | q, o, fc1, fc2 | q, o, fc1, fc2 |
| Readout | Sum, mean and log mutation count of variant states | WT-relative features, residue embeddings, learned projection, then sum/mean/count |
| Trainable parameters | 5,058,945 | 9,990,993 |
| Composition | Joint multi-mutation prediction | Joint multi-mutation prediction |
| WT anchoring | Prediction constrained to zero for WT | Prediction constrained to zero for WT |

The WT-relative readout uses the variant state, reference state at the same position, their difference, and embeddings of the reference/alternate residues. Shared adapters modify the reference and variant streams; the reference representation therefore receives gradients and is recomputed during training. Variant-only adapters leave the reference stream fixed. Increasing adapter rank alone did not beat the shared-adapter recipe.

A joint checkpoint can also score each substitution independently and sum its predictions. That inference control is distinct from an independently trained additive model. Historical breadth runs include both separately trained joint and additive models.

## Data and what each score means

The released development data contain 116 proteins. After canonicalization and replicate-median aggregation, we use 210,842 measured variants: 116,688 singles and 94,154 doubles. Data provenance includes file hashes, exclusions, label-sign checks and exact-sequence overlap checks.

For tuning we created our own seed-0 protein holdout: 104 training proteins / 191,922 variants and 12 validation proteins / 18,920 variants. This is **not** an upstream validation split. All tuning and regularization scores below refer to this internal holdout.

For final fits, all 116 proteins are included in training. The old validation frame remains only as an in-sample diagnostic. In those runs, `val/*` and `predictions_val.csv` are **not held-out measurements**. Checkpoints are the final scheduled weights, selected without those diagnostics or test scores.

External panels retain their released variant coverage. The headline Spearman in external reports is **pooled over variants**; the tuning selection metric is **macro per-protein Spearman over singles and doubles**. They must not be compared numerically. nDCG@30 is macro over eligible proteins, using rectified positive stability as linear gain. Eligibility requires at least one measured variant with score >= 0.5 kcal/mol. Our internal doubles nDCG has only five eligible proteins; cDNA2 has 11.

## Completed external benchmark results: earlier recipe

These are the historical **104-protein fits**, not E02 or the ongoing full-training-set fits. Our joint results are means across three seeds; the comparator is a fixed released Mutate Everything ESM-2 checkpoint evaluated on identical rows.

| Panel | Canonical rows | Comparator pooled Spearman | Our joint mean ± seed SD |
| --- | ---: | ---: | ---: |
| cDNA2 doubles | 21,933 | 0.4539 | **0.5422 ± 0.0085** |
| Released cDNA1 singles | 19,645 | **0.7478** | 0.7230 ± 0.0053 |
| S669 forward | 669 | **0.4736** | 0.4560 ± 0.0162 |
| ProTherm Multiple | 846 | **0.3918** | 0.3182 ± 0.0132 |
| S669 reverse, uncalibrated diagnostic | 669 | **0.1655** | -0.0372 ± 0.0114 |

S669 uses the released **single-only** comparator checkpoint; the other panels use the joint comparator. The reverse diagnostic is not a reproduction of the paper's calibrated reverse table. ProTherm reports additionally separate doubles from 3–10 substitutions.

On cDNA2, the historical three-seed joint model's mean nDCG@30 is **0.3834**, versus **0.3792** for the comparator. The earlier single-seed pilot was 0.3600. This supports a rank-correlation advantage in this panel; it does not establish a convincing retrieval advantage or broad benchmark superiority.

The comparator implementation loads the upstream model/checkpoint and reproduces published cDNA2 nDCG 0.38 and S669 Spearman 0.47 within rounding. Original and local evaluators agree. Exact joins have zero missing/extra rows. Sequence-identity sensitivity checks exist, but their parameters are not asserted to reproduce every detail of the paper's filtering. These historical comparisons are useful system evaluations despite the disclosed 104-versus-116 training-data difference.

## Completed tuning: what helped

The main screen completed 22 fits, including three-seed control/selected confirmation. The regularization screen completed another 10 fits. Neither queue used external test results for automatic promotion.

| Internal validation metric | Control, three-seed mean ± SD | E02, three-seed mean ± SD |
| --- | ---: | ---: |
| Macro all-depth Spearman | 0.8092 ± 0.0031 | **0.8532 ± 0.0010** |
| Macro doubles Spearman | 0.6625 ± 0.0142 | **0.6858 ± 0.0081** |
| Macro doubles RMSE, lower is better | **0.8596 ± 0.0139** | 0.8906 ± 0.0161 |

The ranking improvement survives training seeds on the same protein split. These seed repeats do not establish robustness to different validation proteins. Regression calibration/error remains a weakness: E02 improves ranking while increasing doubles RMSE.

Other findings:

- WT-relative readout improves seed-0 overall Spearman from the longer-budget control's 0.8136 to 0.8392; adding shared rank-8 adapters reaches 0.8542.
- Rank-16 variant adapters with the WT-relative head score 0.8375, below E02.
- Huber alone gives a small doubles Spearman improvement, without an overall ranking gain. Lower learning rates and the singles-first schedules do not provide a clear win.
- The combined Huber/balancing recipe improves internal doubles nDCG@30 from 0.3909 to 0.6914 at the overall-Spearman-selected checkpoints, while overall Spearman falls to 0.7729. All five eligible proteins improve, but about 87% of the average nDCG increase comes from two proteins. Mean precision@30 remains 0.1133. This is an interesting retrieval tradeoff, not broad accuracy improvement.
- Reverse/background augmentations branch from D01 and score 0.8348/0.8253 overall, below D01's 0.8392. They were not combined with E02 in this screen.
- Strong head/adapter dropout and weight decay produce only small changes around the original control. The best regularization score is 0.8133 with head decay 1.0. These settings were not screened on top of E02.

## Final comparison now running

Frozen protocol: three fresh seeds each for A00 and E02, all 116 development proteins. The control uses three exposures (2,472 updates); E02 uses six (4,944 updates). Both use MSE, head LR 0.001, adapter LR 0.0003, AdamW decay 0.01, no dropout/augmentation, cosine decay and 5% warmup, with 64 variants per protein microbatch and four protein microbatches per update.

Final weights are saved at the predeclared endpoint. No early stopping, warm-starting from tuned checkpoints, checkpoint searches, or test-dependent calibration occurs. These are the original screened total exposure budgets; they are not an attempt to transfer each validation-best checkpoint step. Moving to all training proteins and final weights may change performance.

Control seeds run sequentially on GPU 0; E02 seeds run sequentially on GPU 2. The driver waits for all six fits and then evaluates every frozen model on all five panels. A training failure blocks evaluation and records an error. GPU 1 belongs to an unrelated job.

Run root: `/home/andrew/mutant_cache_scratch/runs/mutate_everything_full_comparison_v1/`.

- `plan.json`: exact initialization configurations.
- `status.json`, `driver.log`, `lane_gpu0.log`, `lane_gpu2.log`: live pipeline progress.
- `runs/<recipe>_s<seed>/`: training logs, progress, W&B URLs, checkpoint and result metadata.
- `source_snapshot/`, `source_sha256.json`, `data_provenance.json`: frozen implementation and inputs.
- `comparison.json`: fixed panel definitions and comparator identities.
- `matched_report/`: final output after training; `headline.csv`, `summary.csv`, per-protein comparisons, reproduction checks and paired protein-bootstrap intervals.

W&B entity/project: `amdickson128-usda/mutant_cache`; final-run group: `mutate_everything_full_comparison_v1`.

## What is operational, and what remains open

Operational: real-weight ESM-2 sparse scoring; variant/shared adapters; joint and additive prediction; upstream comparator reproduction; strict same-row multi-panel reports; seed summaries and paired protein intervals; W&B loss/metric, gradient/update, activation and sampling diagnostics; frozen source/data/configuration provenance; automatic final training-to-evaluation pipeline. The refit implementation passed seven synthetic training checks, including full development-set membership and final-weight saving. The configurable evaluator passed three focused checks and reproduced all five historical three-seed mean Spearman scores.

Open: external generalization of E02; whether its ranking gain offsets worse regression error; final full-training-set comparisons; reverse-direction accuracy; sensitivity to held-out protein selection; and a controlled accuracy-versus-compute comparison. The paper finetunes backbone layers and uses a different decoder, whereas our base remains frozen with LoRA. A system comparison can quantify that result, but causal claims about the cache alone require additional controls. The current runner does not produce a controlled same-hardware dense-versus-sparse speed comparison.

## Reading and reproducing

## TODO: next accuracy experiments

Prioritize these experiments from the E02 recipe. Select checkpoints using internal validation only; confirm promising changes across additional protein holdouts before using external panels.

1. [ ] **Add an additive baseline plus a learned interaction residual.** Predict each multi-mutant as the sum of its predicted single-mutant effects plus a jointly computed residual. Constrain the residual to zero for singles and initialize it near zero. Train with direct supervision on measured singles and doubles; where constituent singles are measured, evaluate the residual against measured epistasis. Compare against E02 on singles and doubles ranking and RMSE.
2. [ ] **Train E02 with a within-protein ranking objective.** Compare MSE alone against MSE plus pairwise ranking loss at a small set of ranking weights. Keep pairs within the same protein/assay, and report singles and doubles metrics separately alongside RMSE and retrieval metrics.
3. [ ] **Test whether more shared adaptation improves the reference stream.** Compare E02 with shared adapter rank 8 against rank 16 or 32, then separately test adding shared key/value projection targets. In a separate treatment, compare partial fine-tuning of the last 2–4 backbone blocks against LoRA. Track accuracy gains and training/inference cost.

- [Final comparison protocol](../experiments/mutate_everything_full_comparison/README.md)
- [Drop-in matched comparison runner](../experiments/mutate_everything_matched/README.md)
- [Main tuning results](../experiments/mutate_everything_tuning/STATUS.md)
- [Regularization results](../experiments/mutate_everything_regularization/STATUS.md)
- [Comparator reproduction](../experiments/MUTATE_EVERYTHING_BASELINE.md)
- [Historical breadth details](../experiments/MUTATE_EVERYTHING_BREADTH_RESULTS.md)
- [Diagnostics guide](../experiments/DIAGNOSTICS.md)
- [Experiment index](../experiments/MUTATE_EVERYTHING_EXPERIMENTS.md)

A complete historical matched report generated by the new runner is at `/home/andrew/mutant_cache_scratch/runs/mutate_everything_matched_runner_historical_v1/`. It is a reproducibility check on earlier predictions, not an evaluation of E02.
