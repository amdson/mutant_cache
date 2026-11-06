The released Mutate Everything ESM-2 650M baseline was run locally on cDNA2
on 2026-11-04. These are the comparator values for the eventual matched-row
test comparison:

| Metric | Released ESM-2 checkpoint |
| --- | ---: |
| Pooled Spearman | 0.4539070323 |
| Macro per-protein Spearman | 0.5325919741 |
| Pooled Pearson | 0.4660443081 |
| Pooled RMSE, kcal/mol | 1.2918136607 |
| Macro per-protein RMSE, kcal/mol | 1.1243400064 |
| nDCG@30 (upstream detection definition) | 0.3792157619 |
| Detection precision@30 | 0.1242424242 |

Coverage is complete: 21,933 canonical double-mutant rows across all 18 cDNA2
proteins, after median aggregation of the 22,913 released measurements.
Detection averages over the 11 proteins with at least one experimentally
stabilizing variant at the upstream 0.5 kcal/mol threshold; correlations use
all 18 proteins. The released raw CSV and the canonical prediction frame yield
the same row set in the original upstream evaluator.

Verification has two independent parts. First, our pooled Spearman, pooled
RMSE, nDCG@30 and detection precision match the original pinned `metrics.py`
on these real predictions, to absolute tolerance 1e-10. Second, local nDCG
0.3792157619 reproduces the [official ESM-2 result of 0.38](https://github.com/jozhang97/MutateEverything/tree/034dd91e7a449e68b0f0993043bd32ce281f056c#main-results)
within rounding. The reproduction gate passes. The local Spearman values
above are measured from the released checkpoint, not inferred from S669.

Artifacts are under
`/home/andrew/mutant_cache_scratch/mutate_everything/baseline/`:

- `predictions_cdna2.csv`: complete canonical predictions with labels.
- `predictions_cdna2.json`: checkpoint/label hashes, source revision, timings
  and strict loading record.
- `baseline_metrics.json`: original metrics, local parity checks, macro
  summaries and published-nDCG reproduction gate.
- `upstream_joined.csv`, `upstream_detection.csv` and
  `upstream_per_protein_detection.csv`: original evaluator outputs.
- `local_table.csv`, `local_macro.csv`, `local_per_protein.csv` and
  `local_detection.csv`: our evaluator outputs on the same predictions.
- `predict.log`: inference log with every target and row count.

The checkpoint is the file linked by upstream's ESM evaluation instructions,
Box file 1336050981541, 2,631,746,777 bytes. SHA-256:
`a14792436712feec5a69bd09a37a7aa14cbacc620580ec877ab30aeecb292f25`.
Upstream source remains unmodified at
`034dd91e7a449e68b0f0993043bd32ce281f056c`.

Inference used an isolated venv with fair-esm 2.0.0, the installed PyTorch
2.11.0 runtime, fp32 arithmetic and physical GPU 0 (RTX A6000). The bridge
constructed the original fair-esm ESM-2 architecture without fetching redundant
initialization weights, then strictly loaded all 613 checkpoint tensors.
It ran upstream's model and decoder unmodified, extracting every requested
double from the dense output. This verifies the real checkpoint, rather than
substituting our ESM-2 port for the comparator implementation.

Reproduction commands, from the project environment unless otherwise shown:

```bash
python experiments/mutate_everything_checkpoint_download.py
bash experiments/mutate_everything_baseline.sh
python experiments/mutate_everything_baseline_eval.py
```

`mutate_everything_baseline.sh` invokes the isolated inference interpreter at
`scratch/mutate_everything/venv/bin/python`; that environment was created with
`python -m venv --system-site-packages` and `pip install --no-deps fair-esm==2.0.0`.
Here scratch is `/home/andrew/mutant_cache_scratch`.

For our completed, frozen training configurations, feed their cDNA2 predictions
and `baseline/predictions_cdna2.csv` to `python -m mutant_cache.comparison`,
using `experiments/mutate_everything_expected_cdna2.json` as `--expected` and
the official README URL above as `--expected-source`. Its outer join audits
missing rows before evaluating both methods on identical variants.
Our internal validation Spearman is not comparable to these test values.
Three seeds and paired protein intervals remain necessary for a claim of an
accuracy advantage. The pilot has now completed, and its frozen checkpoint
was evaluated once on these exact rows; see
[first matched comparison](MUTATE_EVERYTHING_PILOT_COMPARISON.md).
