First matched cDNA2 comparison, completed 2026-11-04. This is a preliminary
single-seed result. The reproduction gate passes; the three-seed requirement
for a claim of an accuracy advantage is not met.

| Method | Pooled Spearman | Macro Spearman | nDCG@30 | Detection precision@30 | Pooled RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mutate Everything ESM-2, released checkpoint | 0.4539 | 0.5326 | 0.3792 | 0.1242 | 1.2918 |
| Our joint ESM-2 pilot, seed 0 | 0.5427 | 0.6388 | 0.3600 | 0.1212 | 0.9374 |
| Our pilot summed singles, same checkpoint | 0.5303 | 0.5877 | 0.2368 | 0.0939 | 1.1203 |

All rows use exactly the same 21,933 canonical cDNA2 double mutants across 18
proteins; the outer join finds zero unmatched rows. Macro Spearman gives each
protein equal weight; pooled Spearman ranks all mutations together. Detection
metrics use upstream's definition and the 11 eligible proteins. Both models
use positive-stabilizing canonical scores. The released model reproduces its
published nDCG of 0.38, and the original upstream evaluator agrees with ours
on its real predictions. [Comparator verification](MUTATE_EVERYTHING_BASELINE.md).

The pilot completed at step 1,425 by the predeclared early-stopping rule. The
checkpoint at step 675 was selected exclusively by validation double RMSE
(0.8385), before seeing its cDNA2 predictions. Hyperparameters, checkpoint and
composition were then frozen for a single cDNA2 evaluation. The scoring run
loaded that checkpoint with zero training steps. Its validation predictions
reproduce the training run's selected checkpoint metrics.

This first fit has higher overall rank correlation and lower regression error,
but lower top-30 stabilization retrieval quality than the released comparator.
This is not evidence that our method wins across metrics, seeds, or protocols.
Further model choices must use internal validation, not these cDNA2 results.
Three-seed fits, paired protein uncertainty and a documented selection protocol
remain necessary for the final claim.

Both methods initialize from ESM-2 650M, but their trained weights and decoders
differ: ours uses a frozen base and variant LoRA, theirs fine-tunes the base
and decodes the dense landscape. Our pilot trains on 104 proteins with 12 held
out for validation; the released comparator uses the full 116-protein training
release. Its results are therefore a comparison of these systems, not an
isolation of decoder architecture alone. The summed-singles row uses our joint
checkpoint, not a separately trained additive model.

Artifacts under
`/home/andrew/mutant_cache_scratch/runs/mutate_everything_esm2_pilot_20261104/`:

- `joint_s0/`: training output and selected model.
- `joint_s0_cdna2_fixed/`: fixed-checkpoint test predictions and measured
  inference cost; no retraining.
- `comparison_seed0/headline.csv`: the numerical table above.
- `comparison_seed0/joined_seed0.csv`: identical-row prediction join.
- `comparison_seed0/unmatched.csv`: empty unmatched-row audit.
- `comparison_seed0/paired_differences.csv` and `protein_bootstrap.csv`:
  protein-level differences and conditional single-seed intervals.
- `comparison_seed0/status.json`: passed reproduction gate, zero unmatched
  rows, and `superiority_claim_eligible: false` because there is only one seed.
- `comparison_seed0/pilot_protocol.json`: checkpoint selection and scope.

The raw scripts are `mutate_everything_pilot_test.py` and
`mutate_everything_pilot_compare.py`. The fixed scoring run used physical GPU 0.
Warm scoring of the supplied 21,933-row library measured 2.59 s (about 8,467
mutants/s), with a 2.95 s cold pass including reference construction and
1.68 GB peak allocated memory. Comparator timing is forward-only dense
decoding; these timing scopes are not a controlled throughput comparison.
