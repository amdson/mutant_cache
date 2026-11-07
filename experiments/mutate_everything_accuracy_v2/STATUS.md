# Mutate Everything tuning — live status

Updated 2026-11-06T22:32:43.455033+00:00.

Continuation of `/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2`: 14 completed fits retained; unfinished fits start fresh with W&B and sampled diagnostics. Imported histories have no retrospective gradient/activation measurements.

Selection uses internal validation only. All trials save separate RMSE, Spearman, double-Spearman and nDCG@30 checkpoints. Screening seed is 0; confirmation seeds are 1/2.

| ID | Study | Status | Seed | GPU | Step | Val rho | Double RMSE | Artifacts |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A00 | E02 control | completed | 0 | 2 | 4500 | 0.8542 | 0.8786 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/A00/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/A00/attempt_1/train.log) |
| R01 | MSE plus within-depth rank weight 0.1 | completed | 0 | 0 | 4500 | 0.8549 | 0.8770 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/R01/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/R01/attempt_1/train.log) |
| R02 | MSE plus within-depth rank weight 0.3 | completed | 0 | 2 | 4500 | 0.8541 | 0.8694 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/R02/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/R02/attempt_1/train.log) |
| R03 | MSE plus within-depth rank weight 1.0 | completed | 0 | 0 | 4500 | 0.8543 | 0.8854 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/R03/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/R03/attempt_1/train.log) |
| I01 | Additive singles plus zero-initialized interaction residual | completed | 0 | 2 | 4500 | 0.8479 | 0.8973 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix/runs/I01/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix/runs/I01/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/959ab58197b34e37) |
| S01 | Shared rank 16, scaling 2 | completed | 0 | 2 | 4500 | 0.8561 | 0.8598 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/S01/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/S01/attempt_1/train.log) |
| S02 | Shared rank 32, scaling 2 | completed | 0 | 0 | 4500 | 0.8562 | 0.8860 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/S02/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/S02/attempt_1/train.log) |
| K01 | Shared adapters also adapt K/V | completed | 0 | 2 | 4500 | 0.8544 | 0.8700 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/K01/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/K01/attempt_1/train.log) |
| P00 | E02 fp32 precision control | completed | 0 | 0 | 4500 | 0.8552 | 0.8740 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/P00/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/P00/attempt_1/train.log) |
| P01 | E02 plus last 2 blocks trained at LR 1e-5, fp32 | completed | 0 | 0 | 4500 | 0.8527 | 0.8619 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/P01/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/P01/attempt_1/train.log) |
| P02 | E02 plus last 4 blocks trained at LR 1e-5, fp32 | completed | 0 | 2 | 4500 | 0.8520 | 0.8784 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/P02/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/P02/attempt_1/train.log) |
| G_control_s1 | control confirmation seed 1 | completed | 1 | 2 | 4500 | 0.8521 | 0.9090 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/G_control_s1/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/G_control_s1/attempt_1/train.log) |
| G_control_s2 | control confirmation seed 2 | completed | 2 | 0 | 4500 | 0.8533 | 0.8843 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/G_control_s2/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/G_control_s2/attempt_1/train.log) |
| G_selected_s1 | selected confirmation seed 1 | completed | 1 | 0 | 4500 | 0.8549 | 0.8961 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/G_selected_s1/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/G_selected_s1/attempt_1/train.log) |
| G_selected_s2 | selected confirmation seed 2 | completed | 2 | 2 | 4500 | 0.8527 | 0.8678 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/G_selected_s2/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/G_selected_s2/attempt_1/train.log) |

## Promoted controls

- A: S02

[Compare checkpoint-selection rules](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix/checkpoint_choices.csv) · [Full configuration/results CSV](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix/summary.csv)

Each stage includes its inherited control as a candidate; a new treatment is carried forward only if its validation score improves. Adapter and augmentation studies branch from the same head-study control. No external benchmark is run by this queue.

Raw registry: `/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix/registry.json`; configurations: `runs/*/attempt_*/resolved_experiment.json`.
