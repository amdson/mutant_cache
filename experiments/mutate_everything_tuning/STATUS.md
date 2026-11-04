# Mutate Everything tuning — live status

Updated 2026-11-04T23:03:45.261208+00:00.

Selection uses internal validation only. All trials save separate RMSE, Spearman, double-Spearman and nDCG@30 checkpoints. Screening seed is 0; confirmation seeds are 1/2.

| ID | Study | Status | Seed | GPU | Step | Val rho | Double RMSE | Artifacts |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A00 | Fixed-budget control | completed | 0 | 0 | 2250 | 0.8119 | 0.8444 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1/runs/A00/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1/runs/A00/attempt_1/train.log) |
| A01 | Lower head LR only | completed | 0 | 2 | 2250 | 0.8082 | 0.8686 | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1/runs/A01/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1/runs/A01/attempt_1/train.log) |
| A02 | Lower adapter LR only | running | 0 | 0 |  | — | — | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1/runs/A02/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1/runs/A02/attempt_1/train.log) |
| A03 | Both learning rates lower | running | 0 | 2 |  | — | — | [run](/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1/runs/A03/attempt_1) · [log](/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1/runs/A03/attempt_1/train.log) |
| B01 | Huber loss only | planned | 0 |  |  | — | — |  |
| B02 | 4x non-destabilizing doubles only | planned | 0 |  |  | — | — |  |
| B03 | 8:1 double subsampling only | planned | 0 |  |  | — | — |  |
| B04 | Separate tasks; singles weight 0.1 | planned | 0 |  |  | — | — |  |
| B05 | Combined Huber and balancing recipe | planned | 0 |  |  | — | — |  |
| C01 | 1 singles + 2 mixed exposures | planned | 0 |  |  | — | — |  |
| C02 | 6 mixed exposures; budget control | planned | 0 |  |  | — | — |  |
| C03 | 2 singles + 4 mixed exposures | planned | 0 |  |  | — | — |  |
| D01 | WT-relative mutation readout | planned | 0 |  |  | — | — |  |
| D02 | Mutation projection control | planned | 0 |  |  | — | — |  |
| E01 | Variant LoRA rank 16 | planned | 0 |  |  | — | — |  |
| E02 | Variant rank 8 + shared rank 8 | planned | 0 |  |  | — | — |  |
| F01 | Replace 25% with measured reversals | planned | 0 |  |  | — | — |  |
| F02 | Re-reference 50% of microbatches | planned | 0 |  |  | — | — |  |
| G_control_s1 | Fixed-budget control seed 1 | planned | 1 |  |  | — | — |  |
| G_control_s2 | Fixed-budget control seed 2 | planned | 2 |  |  | — | — |  |
| G_selected_s1 | Selected recipe seed 1 | planned | 1 |  |  | — | — |  |
| G_selected_s2 | Selected recipe seed 2 | planned | 2 |  |  | — | — |  |

## Promoted controls


[Compare checkpoint-selection rules](/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1/checkpoint_choices.csv) · [Full configuration/results CSV](/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1/summary.csv)

Each stage includes its inherited control as a candidate; a new treatment is carried forward only if its validation score improves. Adapter and augmentation studies branch from the same head-study control. No external benchmark is run by this queue.

Raw registry: `/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1/registry.json`; configurations: `runs/*/attempt_*/resolved_experiment.json`.
