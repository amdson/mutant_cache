First training pilot launched 2026-11-04, 21:14 UTC, on physical GPU 2
(RTX A6000). This is a seed-0 validation pilot, not a completed head-to-head.

Update: training finished at step 1,425, selected step 675 using validation,
and the frozen checkpoint has now had one cDNA2 evaluation. The comparator
was reproduced and both were evaluated on identical rows; see
[preliminary comparison](MUTATE_EVERYTHING_PILOT_COMPARISON.md).

Run root: `/home/andrew/mutant_cache_scratch/runs/mutate_everything_esm2_pilot_20261104`.
Process and launch metadata: `launch.json`; live output: `driver.log`;
eventual predictions, selected model and metrics: `joint_s0/`.
Launch command is preserved in `mutate_everything_train.sh`. The detached
process continues after the interactive session ends.

Confirmed active after step 75/2,250: finite training loss 1.2330, validation
double RMSE 0.9914 and Spearman 0.5436, single RMSE 0.7654 and Spearman 0.6689.
These are the first exploratory validation results, not final test estimates.

The run uses ESM-2 650M, a frozen bf16 base, fp32 variant LoRA r8/alpha16
on q/o/fc1/fc2, and a zero-anchored sum/mean/count head. Composition is joint;
validation also evaluates additive scoring using the same parameters.
Head LR 1e-3, adapter LR 3e-4, AdamW weight decay 0.01, gradient clip 1,
cosine schedule with 5% warmup. Budget: three exposures, 2,250 optimizer
steps, 4 proteins x up to 64 variants per step, validation every 75 steps,
early stopping after 10 validations without improvement in macro double
RMSE (single RMSE breaks ties). These hyperparameters are an initial recipe,
not selected on this split. No prior ESM-MSR adapters or heads are loaded.

Released data and audit:

| Partition | Unique variants | Proteins |
| --- | ---: | ---: |
| Train | 191,922 | 104 |
| Validation | 18,920 | 12 |
| cDNA2 test | 21,933 | 18 |

The release has 213,798 training measurements, which become 210,842 unique
protein/mutation keys, and 22,913 test measurements, which become 21,933 keys.
The count reductions are duplicate keys aggregated by median, matching the
upstream evaluator. No rows failed mutation/WT/finite-label checks. Validation
holds out 12 released training proteins using split seed 0; exact sequence
groups stay together. Released train/test have no identical reference sequences
or protein identifiers. The reported homology identity ceiling has not been
recomputed. cDNA2 labels were inspected only for ingestion/sign/overlap audits;
the training process does not evaluate cDNA2.

Archive SHA-256:
`4e4a579902db8308196f4c1bf0fed25ee31b1573c28613525ff2f3def2350f9f`.
Train CSV SHA-256:
`1d1c901f940a207872f0c554237cb24c077118dad81c55ca6b979bc7d401bc5b`.
Test CSV SHA-256:
`e82b937c0d9f2d05c01951a95221b5494086b13aa341e0e31237c5d3c8afdb25`.
Upstream source is pinned at `034dd91e7a449e68b0f0993043bd32ce281f056c`.

Negating release ddG yields the project's positive-stabilizing score.
On 134,090 labels joined to the local ESM-MSR frame by exact sequence and
mutation, Pearson is 0.9981 and Spearman 0.9987. Same-sign MAE is 0.0443
kcal/mol versus 2.8232 for the flipped sign. These support the sign direction;
the release values are not identical, so training retains the released labels.
The manifest's strict equality flag is consequently false. The audit also
finds one cDNA2 protein in ESM-MSR training and three cDNA training proteins
in ESM-MSR test. Those splits must not be mixed or earlier checkpoints reused.

Preparation and provenance live in:

- `scratch/mutate_everything/`: downloaded archive, its hash, extracted CSVs,
  and pinned upstream checkout (here scratch is `/home/andrew/mutant_cache_scratch`).
- `scratch/processed/mutate_everything_cdna/`: tables, immutable-for-this-run
  split, exclusions, manifest, and `sign_and_overlap_audit.json`.
- Run root: copies of manifests/split, `source_snapshot/`, source hashes,
  working-tree diff and `backbone_provenance.json` (cached HF revision,
  safetensors SHA-256, and package versions).

Before launch, the real-weight smoke run completed eight optimizer steps,
saved a finite checkpoint with nonzero LoRA updates and finite validation
predictions, and used 2.01 GiB peak allocated GPU memory. Its outputs are in
`scratch/runs/mutate_everything_smoke_20261104/joint_s0/`. The first attempt
exposed a CUDA allocator initialization error for explicit `cuda:0`; the
training driver now selects/initializes that device before resetting memory
statistics. The successful standalone smoke verified the fix.

The comparator checkpoint has now been downloaded/reproduced and the pilot
evaluated once on cDNA2. Validation-only hyperparameter selection, three-seed
joint/additive fits and the final scientific comparison remain future work.
