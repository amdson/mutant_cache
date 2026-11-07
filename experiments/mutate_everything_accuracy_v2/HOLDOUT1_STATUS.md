# Mutate Everything tuning — live status

Updated 2026-11-07T00:02:51.911505+00:00.

Selection uses internal validation only. All trials save separate RMSE, Spearman, double-Spearman and nDCG@30 checkpoints. Screening seed is 0; confirmation seeds are 1/2.

| ID | Study | Status | Seed | GPU | Step | Val rho | Double RMSE | Artifacts |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A00 | E02 on alternate protein holdout | completed | 0 | 0 | 4350 | 0.7997 | 0.7972 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout1/runs/A00/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout1/runs/A00/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/609d7883501a4ce0) |
| selected | Frozen S02 on alternate protein holdout | completed | 0 | 2 | 4350 | 0.8017 | 0.7417 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout1/runs/selected/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout1/runs/selected/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/9305d493507742fe) |

## Promoted controls

- A: selected

[Compare checkpoint-selection rules](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout1/checkpoint_choices.csv) · [Full configuration/results CSV](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout1/summary.csv)

Each stage includes its inherited control as a candidate; a new treatment is carried forward only if its validation score improves. Adapter and augmentation studies branch from the same head-study control. No external benchmark is run by this queue.

Raw registry: `/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout1/registry.json`; configurations: `runs/*/attempt_*/resolved_experiment.json`.
