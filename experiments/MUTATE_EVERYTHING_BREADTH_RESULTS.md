Expanded frozen-checkpoint tests, started 2026-11-04. The seed-0 external
evaluations below are complete. Additional joint seeds 1/2 and independently
trained additive seeds 0/1/2 are running with the original recipe unchanged.
An automatic finalizer will score all five checkpoints and generate three-seed
matched reports after training completes.

Pooled Spearman on identical complete test rows:

| Test | Rows | Released ESM-2 comparator | Our joint seed 0 |
| --- | ---: | ---: | ---: |
| cDNA2 doubles, previous result | 21,933 | 0.4539 | 0.5427 |
| S669 forward | 669 | 0.4736 (single-only checkpoint) | 0.4744 |
| ProTherm Multiple, all | 846 | 0.3918 | 0.3178 |
| ProTherm doubles | 536 | 0.3722 | 0.3149 |
| ProTherm 3–10 substitutions | 310 | 0.4422 | 0.4198 |
| Released cDNA1 singles, extra control | 19,645 | 0.7478 | 0.7288 |
| S669 reverse, unadjusted diagnostic | 669 | 0.1655 | -0.0368 |

Every join has zero unmatched rows. All new source rows pass WT, mutation and
finite-label checks; duplicate aggregation is the same median convention.
The reverse diagnostic builds each mutant sequence as its own reference,
applies the reverse substitution and negates the observed label. Original
protein IDs are retained for macro metrics. No native-AA bias correction or
reverse-label augmentation is applied. This is not presented as reproduction
of the paper's calibrated reverse-results table.

The first cDNA2 result does not imply broad superiority. S669 pooled Spearman
is effectively tied, and this joint checkpoint trails the comparator on
ProTherm and cDNA1. Macro metrics can differ: on S669 our macro Spearman is
0.4262 versus 0.2744; on ProTherm all-depth it is 0.2890 versus 0.2434.
These macro means include only proteins with a defined correlation, so their
eligible-protein counts accompany the per-protein reports. The overall table
uses pooled Spearman consistently with the paper's regression summaries.

Our joint checkpoint scored by summing predicted singles reaches ProTherm
pooled Spearman 0.4045 overall and 0.5862 for 3–10 substitutions. Its regression
error is larger than the joint model's, and it is not a separately trained
additive model. This motivates retaining both compositions and the separately
trained additive controls rather than selecting a composition on these tests.

The S669 comparator discrepancy is resolved. The joint checkpoint previously
used for cDNA2 scores 0.4529 on S669; the separately released single-only model
scores 0.4736 and reproduces the README's 0.47 within rounding. The primary
S669 row and future repeated-seed reports therefore use that documented
single-only checkpoint. Both comparator rows remain in `external/headline.csv`.
ProTherm, cDNA1, cDNA2 and the unadjusted reverse diagnostic use the joint
comparator checkpoint. We do not substitute the paper's AlphaFold results for
locally measured ESM-2 numbers.

Scoring correctness:

- Requested-library decoding uses the original upstream backbone and trained
  modules. It was compared against the original dense decoder and the original
  higher-order path for 1, 2, 3, 5 and 10 substitutions at two sequence lengths,
  including the upstream chunked path.
- On all 21,933 real cDNA2 rows, requested decoding matches the prior dense
  checkpoint inference to fp32 tolerance. The maximum difference is recorded
  in `external/parity_checks.json`.
- Original and local regression evaluators agree on every new prediction
  frame. Canonical signs are explicit; reverse labels are not passed through
  upstream's majority-sign-flip heuristic.
- Eight loader/evaluator/training/reverse-reference tests passed. No test
  predictions are used to change the training recipe or checkpoint criterion.

The MMseqs2 sensitivity audit used all 116 released training proteins,
`-s 7.5 -c 0.3 --seq-id-mode 1`. It found above-threshold alignments for three
S669 proteins (>30%), two ProTherm proteins (>35%), and one cDNA1 protein
(>36%). These settings are explicit and are not asserted to reproduce every
detail of the paper's sequence filtering. The original released sets remain
the primary tests. Applying identical audit-based exclusions to both methods
does not change the pooled conclusions:

| Audit subset | Rows | Comparator | Ours |
| --- | ---: | ---: | ---: |
| S669 | 624 | 0.4848 | 0.4856 |
| ProTherm | 835 | 0.3923 | 0.3170 |
| cDNA1 | 18,297 | 0.7467 | 0.7271 |

Artifacts:

- `/home/andrew/mutant_cache_scratch/mutate_everything/external/`: normalized
  labels, raw CSV hashes, exclusions, baseline predictions, seed-0 predictions,
  `headline.csv`, evaluator parity and per-test matched reports.
- `external/homology/`: raw MMseqs2 alignments, settings and sensitivity table.
- `/home/andrew/mutant_cache_scratch/runs/mutate_everything_breadth_20261104/`:
  repeated training runs, `joint_worker.log`, finalizer log/status and source
  snapshot. The additive worker's training output is currently in
  `external/driver.log`.
- `finalizer_status.json` tracks waiting/scoring/completion. Once complete,
  `summary.csv`, `all_seed_metrics.csv` and `reports/` will contain all three
  seeds per arm, paired differences and protein-bootstrap intervals.

GPU 2 handles joint seeds 1/2. GPU 0 performed external evaluations and then
started additive seeds 0/1/2. The finalizer waits for training to finish before
scoring the remaining checkpoints. Only the source-code oracle and protocol
checks determine whether evaluation runs; external accuracy never determines
which training configuration is used.

ProteinGym and T2837 have not been run in this batch. We located the historical
87-assay ProteinGym manifest and accessible score archive, preserving their
resource metadata under `mutate_everything/proteingym/`. The exact seven-protein
stability subset still requires reconciliation; it is not replaced by a
phenotype keyword guess or the modern expanded ProteinGym release. T2837 needs
its separate data preparation and matching the paper's exclusions.
