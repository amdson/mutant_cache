# Full-training-set matched comparison v1

Frozen 2026-11-05, before these final models are evaluated. Two recipes, three fresh seeds (0/1/2) each:

- Original control A00: sum_mean_count head, variant LoRA rank 8, three exposures.
- Selected E02: WT-relative head, variant rank 8 plus shared rank 8, six exposures.

Both use pretrained ESM2-650M, MSE, head LR 0.001, adapter LR 0.0003, AdamW decay 0.01, no dropout or augmentation, cosine schedule, 5% warmup, 64 rows per protein microbatch and four microbatches per update. This compares frozen complete recipes; it does not isolate one architecture variable or match training compute.

**Training uses all 116 released development proteins (210,842 canonical measured variants), merging the previous 104/12 subdivision.** Neither test labels nor test predictions select checkpoints. Each fit saves the weights at its last scheduled update. The original screened budgets of 3/6 exposures are retained; validation-best checkpoint numbers from screening do not specify the refit endpoint. Consequently, these final fixed-budget fits may differ from screened checkpoint performance.

The former 12-protein validation frame is retained only for final in-sample diagnostics. **`val/*` in W&B and `predictions_val.csv` are in-sample in these runs.** They must not be reported as held-out results. The run record identifies `validation_role=in_sample_former_validation` and checkpoint rule `fixed_final`.

Exact frozen configurations, training membership counts, source hashes, data hashes, checkpoints, and logs:

`/home/andrew/mutant_cache_scratch/runs/mutate_everything_full_comparison_v1/`

- `plan.json`: all six complete initialization configurations.
- `status.json`: driver phase (`training`, `evaluating`, `complete`, `failed`).
- `lane_gpu0.log` / `lane_gpu2.log`: sequential control/E02 progress.
- `runs/<recipe>_s<seed>/train.log`, `progress.json`, `tracking.json`, `results.json`, `model.pt`.
- `comparison.json`: fixed panel/comparator manifest.
- `matched_report/`: automatically generated after all six training jobs complete; headline/summary CSVs, paired protein intervals, coverage and reproduction checks.

GPU 0 runs control seeds; GPU 2 runs E02 seeds. GPU 1 is unrelated. W&B group: `mutate_everything_full_comparison_v1`. The frozen driver waits for both lanes, then evaluates all six models on cDNA2, cDNA1, S669, ProTherm Multiple and uncalibrated S669 reverse. A lane failure blocks automatic evaluation and records failure; existing runs are never overwritten.

S669 uses the released single-only comparator; other panels use the joint comparator. cDNA2 nDCG and S669 Spearman have published-result reproduction checks. Reverse results remain diagnostics rather than reproduction of the calibrated paper table. The full-training-set match addresses the development-data discrepancy; frozen-base LoRA, sparse decoder and training schedule remain intentional system differences from Mutate Everything.

Launch scripts are retained exactly as run in `experiments/start_mutate_everything_full_comparison.py`, `mutate_everything_full_comparison_driver.py`, and `mutate_everything_full_comparison_lane.py`; executed driver/lane and library copies are frozen under the run root's `source_snapshot/`.
