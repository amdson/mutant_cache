# Mutate Everything tuning — live status

Updated 2026-11-07T01:34:38.786765+00:00.

Selection uses internal validation only. All trials save separate RMSE, Spearman, double-Spearman and nDCG@30 checkpoints. Screening seed is 0; confirmation seeds are 1/2.

| ID | Study | Status | Seed | GPU | Step | Val rho | Double RMSE | Artifacts |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A00 | E02 on alternate protein holdout | completed | 0 | 2 | 4350 | 0.8014 | 0.7335 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout2/runs/A00/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout2/runs/A00/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/5183545e2c6049d1) |
| selected | Frozen S02 on alternate protein holdout | completed | 0 | 0 | 4350 | 0.8017 | 0.7387 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout2/runs/selected/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout2/runs/selected/attempt_1/train.log) · [W&B](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/e392de1fd73646e4) |

## Promoted controls

- A: selected

[Compare checkpoint-selection rules](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout2/checkpoint_choices.csv) · [Full configuration/results CSV](/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout2/summary.csv)

Each stage includes its inherited control as a candidate; a new treatment is carried forward only if its validation score improves. Adapter and augmentation studies branch from the same head-study control. No external benchmark is run by this queue.

Raw registry: `/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_diagnostics_fix_holdout2/registry.json`; configurations: `runs/*/attempt_*/resolved_experiment.json`.
