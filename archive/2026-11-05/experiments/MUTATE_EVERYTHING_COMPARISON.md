The comparison implementation is ready for cDNA. The data have been downloaded
and audited, a real-weight smoke passed, and the first validation-only ESM-2
pilot has completed; see [launch details](MUTATE_EVERYTHING_PILOT_LAUNCH.md).
The released ESM-2 comparator has also been run on all cDNA2 rows, and its
published nDCG reproduced; see [baseline](MUTATE_EVERYTHING_BASELINE.md).
A preliminary [matched seed-0 comparison](MUTATE_EVERYTHING_PILOT_COMPARISON.md)
is complete. Three-seed fits and the final scientific comparison remain pending.
S669 and ProTherm are optional follow-up benchmarks, not part of this recipe.

Use `~/anaconda3/envs/mutant_cache/bin/python` for our code. Set
`MUTANT_CACHE_SCRATCH` to a writable directory outside the repository.

1. Obtain and unpack the release described in the [upstream README](https://github.com/jozhang97/MutateEverything/tree/034dd91e7a449e68b0f0993043bd32ce281f056c):

   ```bash
   gdown 1akfHt1WLbC345DgZwZ5kLeMUADq9XB7y -O "$MUTANT_CACHE_SCRATCH/mutate_everything_data.zip"
   unzip "$MUTANT_CACHE_SCRATCH/mutate_everything_data.zip" -d "$MUTANT_CACHE_SCRATCH"
   ```

   Download the ESM checkpoint linked there separately. Keep it in scratch.
   Clone their repository into scratch and check out
   `034dd91e7a449e68b0f0993043bd32ce281f056c`.

2. Build canonical tables, for example (adjust the unpacked paths):

   ```bash
   python -m mutant_cache.data.mutate_everything \
     --train-csv "$MUTANT_CACHE_SCRATCH/mutate_everything_data/cdna/mutations/cdna_train.csv" \
     --test-csv "$MUTANT_CACHE_SCRATCH/mutate_everything_data/cdna/mutations/cdna2_test.csv" \
     --checkpoint "$MUTANT_CACHE_SCRATCH/mutate_everything_esm.pth"
   ```

   Outputs under `processed/mutate_everything_cdna`: canonical parquet tables,
   the `mutate_everything_cdna` protein split, raw-row exclusion CSVs, canonical
   label CSVs, and `manifest.json` with SHA-256 hashes, released and processed
   shapes, depth counts, sign convention, validation seed and overlap checks.
   Replicates use upstream evaluation's median. Validation holds out 10% of
   training sequence groups, seed 0. Identical sequences stay together;
   identical sequences crossing released train/test cause failure. This is
   our validation policy, not an upstream validation split.

   `--audit-dataset-dir PATH` additionally audits exact-sequence overlap against
   another processed dataset's local splits in both directions and compares
   joined labels to detect sign disagreements. Use the Tsuboyama or ESM-MSR
   processed directory for that check. The loader does not claim that exact
   sequence checks verify the reported MMseqs2 identity ceiling. Review the
   raw CSV shapes, excluded rows, and sign audit before training. Default
   upstream `ddg` is positive destabilizing; `--ddg-sign` explicitly changes
   this when necessary. There is no majority-sign guess.

3. In the upstream environment, run our bridge (absolute path recommended):

   ```bash
   python /home/andrew/mutant_cache/experiments/mutate_everything_predict.py \
     --repo "$MUTANT_CACHE_SCRATCH/MutateEverything" \
     --labels "$MUTANT_CACHE_SCRATCH/processed/mutate_everything_cdna/comparison_labels_test.csv" \
     --predictions "$MUTANT_CACHE_SCRATCH/me_predictions.csv" \
     --resume "$MUTANT_CACHE_SCRATCH/mutate_everything_esm.pth" --device cuda:2
   ```

   This needs upstream's **fair-esm** environment, distinct from this project's
   EvolutionaryScale `esm`. The bridge requires the pinned, clean upstream
   source, loads the checkpoint strictly, and extracts all requested singles
   and doubles from the model output. Upstream `test.py`'s double CSV contains
   only predicted stabilizing variants, so it cannot be used for this join.
   Dense double decoding can require substantial GPU memory for long proteins;
   the supplied bridge follows upstream's inference path and does not implement
   an alternative decoder. It writes a companion JSON recording source and
   checkpoint hashes, GPU and synchronized forward times. The bridge has now
   been exercised with the released checkpoint; `--checkpoint-only-backbone`
   avoids downloading initialization weights that strict checkpoint loading
   immediately overwrites.

4. Train our joint and additive models with the existing training driver:

   ```bash
   python -m mutant_cache.megascale --dataset mutate_everything_cdna \
     --split mutate_everything_cdna --backbone esm2_t33_650M_UR50D \
     --composition joint --seed 0 --device cuda:2 --save-model
   ```

   Choose exposures and hyperparameters using validation only. Freeze them
   before enabling `--final-test --benchmark-inference`. Repeat seeds 0, 1, 2
   for both compositions. Every joint model also emits `pred_add`, its
   additive evaluation using the same learned parameters. Repeat the frozen
   recipe with `esmc_300m` / `esmc_600m` for separate backbone rows. No earlier
   ESM-MSR split or checkpoint is reused. This command is a starting recipe,
   not a tuned or completed experiment.

   ESM-2 uses `q o fc1 fc2` LoRA defaults; ESM-C retains `q o gate up down`.
   Incompatible targets fail clearly. Full ESM-2 forwards implement token
   dropout, including masked marginals. Sparse cache references must be one
   unmasked, unpadded sequence; sparse mask-token variants are rejected because
   mask-ratio scaling changes the entire reference stream.

5. Reproduce their numbers before interpreting differences:

   ```bash
   python -m mutant_cache.comparison --theirs "$MUTANT_CACHE_SCRATCH/me_predictions.csv" \
     --ours seed0/predictions_test.csv seed1/predictions_test.csv seed2/predictions_test.csv \
     --out "$MUTANT_CACHE_SCRATCH/me_comparison"
   ```

   The report writes full-comparator metrics before any join, one joined frame
   per seed, unmatched rows, pooled regression, per-protein macro regression,
   detection summaries, seed-specific differences, and paired protein-bootstrap
   intervals. Detection uses score >= 0.5 (upstream ddg <= -0.5); nDCG uses
   positive rectified score, linear gain, default sklearn tie averaging, K=30.
   Both detection metrics exclude proteins lacking a threshold-positive variant.
   These details were checked against upstream `metrics.py` at the pinned
   commit; they differ from our ESM-MSR retrieval definition.

   `status.json` defaults to an unverified reproduction. Supply `--expected`
   with a JSON mapping such as `{"all": {"ndcg30": VALUE, "n": COUNT}}` and
   `--expected-source` with the citation **after verifying the exact table and
   dataset**. The gate compares their complete predictions with those numbers
   (absolute tolerance 0.005, adjustable with `--atol`). Never fill a cDNA2
   target from S669. Claim eligibility additionally requires three distinct
   seeds and complete row coverage. A gate pass is not a statistical win.

   Protein-bootstrap intervals average paired seed differences within each
   protein and resample proteins, not mutations. Incomplete seed/protein cells
   have no interval; seed-specific tables retain the evidence. Signs of error
   metrics are ours minus theirs (negative means lower error).

   `--cost-csv FILE` optionally renders `accuracy_cost.png`; columns are
   `label,seconds_per_mutant,accuracy,regime,cost_source`. Fill these from the
   measured inference JSONs, including the dense vs supplied-library regimes.
   Compare identical accuracy metrics. Our cold timing includes reference
   construction, warm timing measures repeated scoring, and neither includes
   model loading. The plot labels timings as an uncontrolled comparison.

Validation: small fp32 HF equivalence (masked and padded), nonzero shared and
variant LoRA cache/oracle checks, ragged cache, full-position exact oracle,
loader/report tests, and synthetic training/timing integration are automated.
The cached 650M fp32 equivalence test is opt-in:

```bash
MUTANT_CACHE_TEST_ESM2_650M=1 MUTANT_CACHE_TEST_DEVICE=cuda:2 python -m pytest tests/test_esm2.py -q
```

The published HF checkpoint stores rounded rotary frequencies, which the port
preserves. Recomputing ideal frequencies causes measurable fp32 drift. In
Transformers 4.57.6, `EsmModel.forward` omits the embedding attention mask;
the padded parity reference supplies that mask directly to `EsmEmbeddings`.
Unpadded substitution scoring is unaffected by that reference discrepancy.

Implementation verification: 85 tests passed in the final GPU-2 integration
run, including the cached 650M fp32 test, fused sparse/ragged cache oracles,
adapter audits, training, timing and reporting. A separate ESM-C regression
selection passed 49 tests. Direct synthetic comparison with the pinned
upstream evaluator matched pooled Spearman, RMSE, nDCG@30 and detection
precision to 1e-12, including prediction ties. These are code checks, not
reproduction of a published result.

Any eventual result must say: both methods use ESM-2 650M pretrained weights as
their initialization, but their trained weights and adaptation policies differ;
our sparse attention/LoRA decoder and their learned representation aggregation
are different; they decode a dense landscape and we score a supplied library.
The released comparator fine-tunes backbone layers, so matching the pretrained
backbone is not equality of final weights or training protocols. ESM-C rows
use a different backbone, and AlphaFold results must remain separate labelled
published rows unless reproduced. The ESM-2 checkpoint's cDNA2 nDCG is now
reproduced. We have not run MMseqs2 or produced the final scientific comparison.
