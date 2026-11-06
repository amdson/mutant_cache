[Research journal](RESEARCH_JOURNAL.md) — concise chronological record of experiments, results and decisions.

# Current project snapshot

[Model, benchmark results, tuning findings and final comparison state — 2026-11-05](../docs/MUTATE_EVERYTHING_STATE_2026-11-05.md).

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

The [diagnostics guide](DIAGNOSTICS.md) lists the sampled train/validation,
optimization, activation and exposure measurements and their default rates.

## Regularization screen

[Protocol and exact plan](mutate_everything_regularization/README.md) · [Live status](mutate_everything_regularization/STATUS.md). Ten controlled fits on GPU 2 compare head and adapter dropout and separate weight decay against a fresh original control; tracked in W&B group `mutate_everything_regularization_v1`.

## Configurable matched comparisons

[Runner instructions and drop-in model configurations](mutate_everything_matched/README.md): frozen checkpoint or prediction inputs, strict complete-row matching across five panels, seed summaries and paired protein intervals.

## Full-116-protein final fits

[Final comparison protocol and monitoring](mutate_everything_full_comparison/README.md): three fresh seeds each for control and E02, fixed final weights, then automatic matched-panel evaluation.

## E02 accuracy round 2

[Plan and monitoring](mutate_everything_accuracy_v2/README.md): ranking, additive interaction residual, shared capacity/KV, partial backbone, seed repeats and two extra protein holdouts.
