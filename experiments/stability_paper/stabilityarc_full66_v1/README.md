# Full 66-assay StabilityArc comparison, seed 0

Extend the seven-assay pilot to all 66 protein holdouts without changing the
scientific recipe or selecting targets by performance. Import all seven completed
seed-0 fits, with checkpoint/config/prediction hashes. Train the remaining 59
fresh fits on the other 65 proteins each. The sources, labels, normalization,
protein-uniform sampler, rank-8 shared/variant ESMC-600M adapters, WT-relative
head, MSE objective and fixed six-exposure final checkpoint follow the pilot.
Training settings differ from StabilityArc's own method, as documented in the
[pilot protocol](../stabilityarc_pilot_v1/README.md). This evaluates complete methods.

All raw inputs retain their pinned hashes; counts match all 66 public assay
inventories. Exact reproduction of StabilityArc's private packed data is not
claimed. The comparator is its published per-assay scores, not newly run baseline
checkpoints. Those held-out checkpoints and row-level predictions are not released.
The same raw ProteinGym scoring labels and complete coverage are checked for
every fit. No assay's outer labels select hyperparameters, budget or checkpoints.

Run `python experiments/start_stabilityarc_full66.py` once after preparing data
with `python -m mutant_cache.data.stabilityarc --scratch /home/andrew/mutant_cache_scratch --remaining`.
Both refuse overwrite. The launcher freezes source/configs/data in
`scratch/runs/stabilityarc_full66_v1`. Two queue lanes each schedule GPUs 0/2,
allowing four simultaneous fits (two per GPU). GPU 1 is reserved for unrelated
work. Each lane has its own queue state, logs and run directories.

W&B group: `stabilityarc_full66_v1` for new fits; imported fits retain their pilot
group and original histories. Per-assay training losses are logged for every new
fit. The seven imported fits are not rerun to manufacture additional replicates.

`comparison.csv` and `summary.json` refresh every 30 seconds as fits finish.
Until all 66 finish, the macro average describes only the completed subset.
The final report includes overall and singles macro Spearman over all 66, doubles
over eligible assays, per-assay paired differences and wins, and paired-assay
bootstrap intervals. These intervals condition on seed-0 fits and do not measure
training randomness or private-preprocessing uncertainty. Every result is kept;
no best-seed or best-assay selection is performed. The 59 newly evaluated folds
can also be analyzed separately from the seven observed pilot folds.

`COMPLETE.json` is written only after all jobs and coverage checks succeed.
Failures remain visible in lane queue states; report generation rejects changed
labels/depths, missing rows, duplicate predictions and nonfinite predictions.
This run makes no inference-speed superiority claim. Multi-seed confirmation,
matched full-mutant/reference-only controls and a locally reproducible baseline
are separate follow-ups.
