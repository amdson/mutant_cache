# Mutate Everything tuning — live status

Updated 2026-11-06T00:57:41.113256+00:00.

Selection uses internal validation only. All trials save separate RMSE, Spearman, double-Spearman and nDCG@30 checkpoints. Screening seed is 0; confirmation seeds are 1/2.

| ID | Study | Status | Seed | GPU | Step | Val rho | Double RMSE | Artifacts |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A00 | E02 control | completed | 0 | 2 | 4500 | 0.8542 | 0.8786 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/A00/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/A00/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/4e09c83527744911) |
| R01 | MSE plus within-depth rank weight 0.1 | completed | 0 | 0 | 4500 | 0.8549 | 0.8770 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/R01/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/R01/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/a02ebec6c88744b7) |
| R02 | MSE plus within-depth rank weight 0.3 | running | 0 | 2 | 3750 | — | — | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/R02/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/R02/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/c922694e011d4c8b) |
| R03 | MSE plus within-depth rank weight 1.0 | running | 0 | 0 | 3650 | — | — | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/R03/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/runs/R03/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/e4e680d9598a4278) |
| I01 | Additive singles plus zero-initialized interaction residual | planned | 0 |  |  | — | — |  |
| S01 | Shared rank 16, scaling 2 | planned | 0 |  |  | — | — |  |
| S02 | Shared rank 32, scaling 2 | planned | 0 |  |  | — | — |  |
| K01 | Shared adapters also adapt K/V | planned | 0 |  |  | — | — |  |
| P00 | E02 fp32 precision control | planned | 0 |  |  | — | — |  |
| P01 | E02 plus last 2 blocks trained at LR 1e-5, fp32 | planned | 0 |  |  | — | — |  |
| P02 | E02 plus last 4 blocks trained at LR 1e-5, fp32 | planned | 0 |  |  | — | — |  |
| G_control_s1 | control confirmation seed 1 | planned | 1 |  |  | — | — |  |
| G_control_s2 | control confirmation seed 2 | planned | 2 |  |  | — | — |  |
| G_selected_s1 | selected confirmation seed 1 | planned | 1 |  |  | — | — |  |
| G_selected_s2 | selected confirmation seed 2 | planned | 2 |  |  | — | — |  |

## Promoted controls


[Compare checkpoint-selection rules](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/checkpoint_choices.csv) · [Full configuration/results CSV](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/summary.csv)

Each stage includes its inherited control as a candidate; a new treatment is carried forward only if its validation score improves. Adapter and augmentation studies branch from the same head-study control. No external benchmark is run by this queue.

Raw registry: `/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2/registry.json`; configurations: `runs/*/attempt_*/resolved_experiment.json`.
