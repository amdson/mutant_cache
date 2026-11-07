# Frozen E02 on ESM-MSR

Three fresh ESM-2 650M fits (seeds 0/1/2), frozen E02 rank-8 variant/shared adapters, WT-relative zero-anchored head, six exposures. The study-local recipe preserves the settings regardless of later edits to generic recipes.

Use the released `esm_msr_hyperopt` protein partition: 183,379 training, 35,373 validation and 43,700 test variants. Train on train only; select each checkpoint using validation macro overall Spearman grouped by `code_wt`, minimum two rows. Evaluate each selected checkpoint once on test, reporting all seeds. No hyperparameter screen, best-seed selection or train+validation refit. The test has 32,968 singles and 10,732 doubles. These counts are metadata checks, not a model-selection criterion.

Primary metric: macro per-protein overall Spearman. Secondary metrics: singles/doubles Spearman and RMSE, nDCG@96 and threshold-zero retrieval, and additive scoring of the joint model where available. Existing Megascale evaluation writes per-protein and pooled diagnostics separately.

Launch: `python experiments/start_esm_msr_e02_test.py`. The raw launcher freezes code/configs/data hashes in `scratch/runs/esm_msr_e02_test_v1`, validates all jobs, then waits for the accuracy-v2 diagnostics-fixed coordinator and successful additional holdout runs before using GPUs 0/2. A failed dependency stops the coordinator. The snapshot is used to execute training. No processed data are rebuilt.

Monitor `coordinator.log`, `queue_state.json` and `logs/msr_esm2_test_v1_s*.log` under that root. Each seed saves predictions, checkpoints and metrics under `runs/`; after all finish, the coordinator writes `test_summary_by_seed.csv` and `COMPLETE.json`.

## Comparator work remaining

These runs measure our model on ESM-MSR's released internal split. A head-to-head paper result requires identifying and pinning the exact paper checkpoint (the bundled small model is not a substitute), obtaining ESM3 weights, reproducing the pinned upstream inference modes, and checking its training membership. Preserve original upstream rows, map predictions by reference and canonical substitutions, reject duplicate/missing predictions, audit upstream constituent filtering and engineered backgrounds, and evaluate both systems on identical rows with verified metric definitions. Save full coverage and any exclusions before looking at comparative scores. The local importer count reconciliation alone does not establish identical upstream scoring rows.

External panels such as S461, SSYM and PTMUL-D need their own canonical import and comparison protocol. This study does not reproduce those panels, their structure inputs, or the paper's training augmentation. Do not feed this experiment to the Mutate Everything-specific matched-suite evaluator.
