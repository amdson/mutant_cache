Start here for the Mutate Everything comparison experiments.

| Program | Purpose | Results / status |
| --- | --- | --- |
| Baseline reproduction | Released ESM-2 checkpoints and identical-row evaluator checks | [Baseline](MUTATE_EVERYTHING_BASELINE.md) |
| Initial pilot | First joint fit, selected by validation double RMSE | [Pilot comparison](MUTATE_EVERYTHING_PILOT_COMPARISON.md) |
| Untuned seed repeats and breadth | Three joint and three additive fits; external generalization | [Breadth notes](MUTATE_EVERYTHING_BREADTH_RESULTS.md) · [three-seed summary](/home/andrew/mutant_cache_scratch/runs/mutate_everything_breadth_20261104/summary.csv) |
| **Systematic tuning v1** | Learning rates, objectives, curriculum, head, adapters and augmentation | **[Plan](mutate_everything_tuning/README.md)** · **[Live results](mutate_everything_tuning/STATUS.md)** |

The untuned seed/breadth program is complete. Its historical results are kept
separate from the new validation-only tuning program. Raw artifacts and logs
remain under `/home/andrew/mutant_cache_scratch/`; checked-in experiment scripts
record the exact work performed.

Future runs support [W&B tracking](WANDB_TRACKING.md) with the checked-in
[project preset](wandb_tracking.json).
