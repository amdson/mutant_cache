# Matched comparison runner

Run frozen models against released Mutate Everything ESM-2 predictions on cDNA2, S669, ProTherm Multiple, cDNA1, and an explicitly labelled uncalibrated S669 reverse diagnostic. ProTherm reports include doubles and higher-order mutations separately. S669 uses the released single-only comparator checkpoint; other panels use the joint checkpoint. This is the available ESM-2 comparison suite, not every model or assay in the paper.

## Drop in checkpoints

Copy [checkpoints.json](checkpoints.json), replace the model ID and checkpoint paths, and describe the actual training protocol. Each model has one `fits` entry per seed. Fits within a model must have distinct seed IDs; use separate model entries for different recipes. Checkpoints use this repository's `model.pt` format (`args` and `trainable_state`). The checkpoint seed must match the config. Paths resolve relative to the JSON file.

```bash
python -m mutant_cache.matched_suite \
  --config experiments/mutate_everything_matched/checkpoints.json \
  --output /home/andrew/mutant_cache_scratch/runs/my_matched_comparison \
  --device cuda:0 --batch-size 512
```

Choose an available GPU before scoring. Existing tuning workers use GPUs 0 and 2. This command scores supplied frozen checkpoints; it does not train, select checkpoints, calibrate predictions or change queued jobs.

The example is the historical pilot trained on 104 proteins, explicitly labelled in its provenance. A final full-training-set comparison requires checkpoints trained on all 116 released training proteins, with recipe and duration fixed before test evaluation. Evaluation matching alone does not establish matching training conditions. The `training_protocol` field is descriptive, not an automated training-data certification. Checkpoint arguments are also copied into report provenance.

## Other model formats / existing predictions

Use a `predictions` mapping instead of `checkpoint` in each fit. [historical_predictions.json](historical_predictions.json) is a runnable three-seed example covering all five panels. Each CSV must contain `target_id,mutant,n_mut,score,pred`, using canonical positive-stabilizing scores in kcal/mol. An optional `seed` column must agree with the config. Different models may use checkpoints or supplied predictions in the same report.

```bash
python -m mutant_cache.matched_suite \
  --config experiments/mutate_everything_matched/historical_predictions.json \
  --output /home/andrew/mutant_cache_scratch/runs/historical_matched_report
```

`--validate-only` checks all panel inputs, exact coverage, labels, and supplied prediction seeds without scoring models or writing a report. For checkpoints it validates file structure/seed but does not execute the model. Output directories must be empty; use a fresh directory for each report.

## What to read

- `headline.csv`: each seed's pooled Spearman/RMSE and protein-macro nDCG@30/precision@30 on each full panel, with detection eligibility counts.
- `summary.csv`: mean and standard deviation across supplied seeds for each metric/depth. A one-seed SD is undefined. Comparator scores repeat across seeds because the comparator checkpoint is fixed.
- `reports/<model>/<panel>/`: exact prediction joins, per-protein results, macro regression metrics, paired differences, protein-bootstrap intervals and published-result reproduction gates. Bootstrap intervals condition on supplied fits; they do not capture every source of training uncertainty.
- `provenance.json`: hashes of configuration, labels, baseline predictions and checkpoints or supplied predictions, checkpoint arguments, comparator identities and declared training protocol.
- `status.json`: completion/failure and per-panel reproduction/coverage gates.

Missing or extra rows, duplicate canonical variants, mismatched labels/signs or inconsistent seeds cause failure rather than silently evaluating a subset. Do not change test panels or tune model choices from these scores. Adding a new benchmark requires canonical labels and complete baseline predictions in a new panel entry.

Only cDNA2 nDCG@30 and S669 pooled Spearman have pinned published targets in the default configuration. Other panels still give matched system comparisons, with reproduction explicitly unverified. The reverse panel is **uncalibrated** and must not be compared with the paper's calibrated reverse table. Training adaptation (frozen base + LoRA versus backbone finetuning) and decoder differences remain part of the system comparison. This runner does not measure matched inference cost or implement full-dataset training.
