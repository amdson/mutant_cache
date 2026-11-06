# Mutate Everything tuning — live status

Updated 2026-11-05T07:45:35.651626+00:00.

Selection uses internal validation only. All trials save separate RMSE, Spearman, double-Spearman and nDCG@30 checkpoints. Screening seed is 0; confirmation seeds are 1/2.

| ID | Study | Status | Seed | GPU | Step | Val rho | Double RMSE | Artifacts |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| R00 | Control | completed | 0 | 2 | 2250 | 0.8119 | 0.8444 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R00/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R00/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/dfb2837c3dea453a) |
| R01 | Head dropout 0.2 | completed | 0 | 2 | 2250 | 0.8098 | 0.8608 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R01/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R01/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/389d6fea21a64e75) |
| R02 | Head dropout 0.5 | completed | 0 | 2 | 2250 | 0.8115 | 0.8595 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R02/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R02/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/8cd4aff1895c40f1) |
| R03 | Adapter dropout 0.1 | completed | 0 | 2 | 2250 | 0.8124 | 0.8550 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R03/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R03/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/7a3449cb25de4b24) |
| R04 | Adapter dropout 0.3 | completed | 0 | 2 | 2250 | 0.8093 | 0.8727 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R04/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R04/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/2b8d6031e5a145d6) |
| R05 | Head decay 0.1 | completed | 0 | 2 | 2250 | 0.8124 | 0.8455 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R05/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R05/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/e20ac0a23e994566) |
| R06 | Head decay 1.0 | completed | 0 | 2 | 2250 | 0.8133 | 0.8510 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R06/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R06/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/dd128241cce949fc) |
| R07 | Adapter decay 0.1 | completed | 0 | 2 | 2250 | 0.8109 | 0.8590 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R07/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R07/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/0000e541e11c469a) |
| R08 | Adapter decay 1.0 | completed | 0 | 2 | 2250 | 0.8100 | 0.8695 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R08/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R08/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/ae8e7c3737de49c8) |
| R09 | Combined strong regularization | completed | 0 | 2 | 2250 | 0.8125 | 0.8792 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R09/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/runs/R09/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/e48f4123103941fc) |

## Promoted controls

- R: R06

[Compare checkpoint-selection rules](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/checkpoint_choices.csv) · [Full configuration/results CSV](/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/summary.csv)

Each stage includes its inherited control as a candidate; a new treatment is carried forward only if its validation score improves. Adapter and augmentation studies branch from the same head-study control. No external benchmark is run by this queue.

Raw registry: `/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1/registry.json`; configurations: `runs/*/attempt_*/resolved_experiment.json`.
