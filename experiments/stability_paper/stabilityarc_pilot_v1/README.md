# Seven-assay StabilityArc pilot

Operational updates: concurrent scheduling was authorized after launch. The
pilot no longer waits for accuracy jobs and shares GPUs 0/2. Per-assay sampled
training-objective loss logging was added for the four still-pending fits
(SQSTM_MOUSE, DN7A_SACS2, ESTA_BACSU, A4_HUMAN), using a separate frozen
`assay_logging_snapshot`. BCHB completed and CBX4/SPTN1 started before that
change; their original training histories remain pooled. Active fits were
adopted without restart. New W&B keys are `train_assay/<assay>/loss` and
`train_assay/<assay>/window_rows`, emitted every 50 optimizer steps. These
are training losses on sampled batches, not held-out assay measurements.

Question: can the sparse ESMC-600M E02 methodology rank held-out ProteinGym stability variants competitively with the published standalone StabilityArc scores?

Preselected targets, before our comparative predictions: BCHB_CHLTE (first pilot), CBX4_HUMAN, SPTN1_CHICK, SQSTM_MOUSE, DN7A_SACS2, ESTA_BACSU and A4_HUMAN. These include mixed single/double assays, singles-only assays and the two non-Tsuboyama assays. This selected subset is a pilot, not an estimate of the full benchmark's average improvement.

## Data and holdouts

Source: ProteinGym v1.3 `DMS_ProteinGym_substitutions.zip`, the existing ProteinGym reference metadata, and StabilityArc revision `94fc12dc9b73e48e48e818a489155feccafea94a`. All 66 assays match the released per-assay total/single/double counts exactly (134,794 rows total). Every mutant is reconstructed against its reference; duplicates, nonfinite labels and depth mismatches fail. Identical reference sequences under different protein names also fail. No homology filtering beyond the original protein grouping is imposed.

The private packed StabilityArc data and row-level predictions are unavailable. Count reconciliation and mutation reconstruction do not prove identical private preprocessing. Comparisons are to published per-assay summaries, not paired row-level upstream predictions. Input hashes and complete counts live in `scratch/stabilityarc/pilot_v1/manifest.json` and `assay_inventory.csv`.

Each fold holds out the entire target protein. Six other proteins get deterministic inner-validation assignments; `full_train_refit` merges them back before the fixed-budget fit, so **all 65 other proteins train**, with no early stopping. Former validation metrics are explicitly in-sample diagnostics. Outer target labels never select checkpoints, budgets or recipes.

For training proteins only, targets are transformed by subtracting the assay's single-mutant median and dividing by its single-mutant MAD (floor 1e-6), using the same affine transform for singles and doubles. Outer test labels stay raw ProteinGym DMS_score. This is an explicit choice of our method, not a claim of reproducing StabilityArc's loss normalization. Report Spearman only for comparative claims: test RMSE and energy/epistasis quantities emitted by the generic harness do not have a matched physical scale here. All assays have fewer than 40,000 total variants, so no cap removes rows.

## Fits and reports

Frozen starting recipe: ESMC-600M bf16 frozen base, fp32 variant/shared rank-8 LoRA (alpha 16), WT-relative zero-anchored head, MSE, six exposures, protein-uniform sampling, head LR 1e-3, adapter LR 3e-4, seed 0. Each starts fresh from pretrained weights. This differs from StabilityArc's frozen feature encoder, decoder, loss and 40-epoch budget; it compares full training methods with the same pretrained backbone and source assay membership. Save timing/memory/parameter records rather than asserting matched compute.

`experiments/start_stabilityarc_pilot.py` freezes source/config and data hashes in `scratch/runs/stabilityarc_pilot_v1`. It waits for the already-running accuracy-v2 continuation and both additional holdouts to succeed, then runs BCHB first and verifies complete finite test predictions with unchanged labels. After that integrity check, it runs the remaining six on GPUs 0/2 regardless of whether the first score wins or loses. Each fit scores its outer test once at the fixed final update. Failures stop the coordinator.

The report `comparison.csv` shows published StabilityArc versus our Spearman for all, single and double subsets of each assay, including counts and differences. Singles-only panels have no doubles score. Do not claim a seven-assay mean matches the paper's 66-assay mean, or treat one seed per fold as a measure of training uncertainty.

Build isolated datasets with `python -m mutant_cache.data.stabilityarc --scratch /home/andrew/mutant_cache_scratch` after fetching the inputs (refuses existing destinations). `experiments/smoke_stabilityarc_pilot.py` is the recorded one-update training/inner-validation smoke, with no outer test evaluation. Launch once with `python experiments/start_stabilityarc_pilot.py`; monitor the scratch coordinator log, `queue_state.json`, per-run logs, and `STATUS.json`.
