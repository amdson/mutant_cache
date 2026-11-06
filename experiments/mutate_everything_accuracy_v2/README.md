# E02 accuracy experiments, round 2

This round is separate from the completed full-116-protein matched comparison. Treatments start fresh from pretrained ESM2; no final-fit or tuned weights are loaded. All screen treatments use the original 104/12 protein split, seed 0, six exposures, and E02's other settings. No external panel is scored by this queue.

[Exact plan](plan.json) · [Live status](STATUS.md)

| IDs | Treatment |
| --- | --- |
| A00 | Repeated E02 control |
| R01/R02/R03 | MSE plus RankNet weight 0.1/0.3/1.0; comparisons confined to the same protein, assay and mutation depth; mean over eligible depths |
| I01 | Sum of predicted single effects plus joint interaction residual; residual exactly zero for singles/WT and zero-initialized for multi-mutants |
| S01/S02 | Shared adapter rank 16/32, shared alpha 32/64; variant rank/alpha remain 8/16, preserving scaling 2 |
| K01 | Add shared K/V projection targets, retaining q/o/fc1/fc2 |
| P00 | E02 fp32 precision control |
| P01/P02 | E02 plus last 2/4 base blocks and final norm trained, backbone LR 1e-5, fp32 |
| G_* | Seed-1/2 confirmation of E02 and the automatically selected screen recipe |

Partial backbone fits retain E02's adapters and add trainable base blocks. They are fp32 to avoid bf16 master-weight updates; compare them to P00 to distinguish precision effects. These tests do not reproduce the paper's much broader backbone finetuning recipe.

The residual head shares the WT-relative token projection with the single-effect head. It scores each constituent single independently, adds those predictions, and adds a separately parameterized joint residual for multi-mutants. Single measurements supervise the single-effect branch directly; double labels supervise the total, without measured-single teacher forcing. Existing measured-epistasis reports evaluate complete constituent-single groups. This adds computation; it is not assumed to be faster than E02.

The ranking coefficient multiplies an auxiliary dimensionless logistic objective alongside raw-stability MSE. Validation loss remains the configured regression objective; training loss includes the auxiliary ranking term. Ranking pairs are formed within each sampled protein microbatch, including across any union-position sub-batches.

Screen promotion uses macro overall validation Spearman, then doubles RMSE. All trials also save doubles-RMSE, doubles-Spearman and doubles-nDCG checkpoints; promotion is not a claim of improvement on those metrics. Seeds 1/2 confirm the frozen selected recipe on the original protein split.

After the primary queue completes without failures, an automatic coordinator compares E02 against the frozen winner on two additional protein holdouts (split seeds 1 and 2, training seed 0). Identical reference sequences stay together; released test assignments stay unchanged. These are additional validation sensitivity checks on the same 116 development proteins, not new independent external test sets. Each holdout gets its own run root and configuration; comparing scores between different holdouts is not a treatment comparison. If E02 wins the screen, the additional fits are repeated E02 controls.

The main screen is 11 seed-0 fits plus up to four seed confirmations, then four additional holdout fits. GPUs 0 and 2 process the queue; GPU 1 is unrelated. W&B group is each run root name.

Artifacts:

- `/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2`: primary registry, exact resolved configurations, source/data hashes, histories, diagnostics, model checkpoints and live CSV summaries.
- Same path with `_holdout1` / `_holdout2`: extra protein-split confirmations, launched automatically after the primary queue.
- `launch.json`, `worker_gpu*.log`, `holdout_coordinator.log`: worker/coordinator monitoring.
- `experiments/start_mutate_everything_accuracy_v2.py` and `mutate_everything_accuracy_v2_holdouts.py`: exact launch/coordinator scripts. The executed coordinator is copied to the source snapshot.

The completed full-training-set comparison remains unchanged. New findings need confirmation before any separately declared external evaluation round; these experiments do not retrofit changes into the completed results.
