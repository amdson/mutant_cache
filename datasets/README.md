# Datasets

All data lives under `$MUTANT_CACHE_SCRATCH` (default `~/mutant_cache_scratch`), never in the repo. This directory holds the build rules (`Makefile`), raw-download checksums (`checksums.md5`) and the protocol files that training resolves against (`protocols/`).

The root Makefile forwards `make data DATASET=<target>` to `make -C datasets <target>`. Always preview with `make -n` first.

## Processed stability benchmarks

| Target | Status | Produces | Protocol |
| --- | --- | --- | --- |
| `mutate-everything` | ready | `processed/mutate_everything_cdna/` (incl. holdout1/2 splits), `mutate_everything/external/` | [`mutate_everything_v1.json`](protocols/mutate_everything_v1.json) |
| `esm-msr` | ready for training; matched external comparison unresolved | `processed/esm_msr_mega/` | [`esm_msr_v1.json`](protocols/esm_msr_v1.json) |
| `stabilityarc` | **planned**: the target fails and lists what is missing | nothing | [`stabilityarc_v1.json`](protocols/stabilityarc_v1.json) |

```bash
make -n -C datasets mutate-everything     # should report "Nothing to be done" on the current scratch tree
make -n -C datasets esm-msr
make -C datasets stabilityarc             # exits nonzero while planned
make -C datasets verify                   # content hashes vs processed_hashes.json
```

### Rebuild safety

Each processed rule is a file target, so existing outputs are up to date and skipped. A processed rule that does need to run first checks its output directory. If that directory is non-empty, the rule refuses to write unless `FORCE=1` is set. Live tuning runs read these directories, so stop dependent jobs before forcing a rebuild. A rebuild does not change the processed split for the same inputs and seed. It does rewrite every table, and it does not regenerate the derived files listed below.

All paths come from `$(SCRATCH)` (`MUTANT_CACHE_SCRATCH`); `PYTHON` defaults to `~/anaconda3/envs/mutant_cache/bin/python` if it exists, else `python`. Do not depend on code mtimes: if `mutant_cache/data/*.py` changes, make does not see it, so rebuilding is a deliberate `FORCE=1` decision.

### Mutate Everything (`mutate-everything`)

Upstream [jozhang97/MutateEverything](https://github.com/jozhang97/MutateEverything) @ `034dd91e7a449e68b0f0993043bd32ce281f056c`. The data archive is Google Drive id `1akfHt1WLbC345DgZwZ5kLeMUADq9XB7y`.

1. `mutate-everything-raw`: `python -m mutant_cache.data.mutate_everything_raw fetch` downloads `mutate_everything/mutate_everything_data.zip` only if the zip is absent, from `https://drive.usercontent.google.com/download?id=1akfHt1WLbC345DgZwZ5kLeMUADq9XB7y&export=download&confirm=t` (the request the original script made on 2026-11-04). If Drive serves an HTML page instead, fetch it by hand with `gdown 1akfHt1WLbC345DgZwZ5kLeMUADq9XB7y` (the upstream README command) into `mutate_everything/`. `extract` checks the archive sha256 `4e4a5799…0f9f` and writes `mutate_everything/data/{cdna_train,cdna2_test}.csv` (sha256-checked as well). `mutate-everything-check` verifies the archive only.
2. `processed/mutate_everything_cdna/manifest.json`: runs `python -m mutant_cache.data.mutate_everything --train-csv …/cdna_train.csv --test-csv …/cdna2_test.csv --audit-dataset-dir processed/esm_msr_mega`. These are the arguments recorded in the existing `manifest.json`: no checkpoint, seed 0, val fraction 0.1, `destabilizing_positive`. The step writes canonical parquet tables, the `splits/mutate_everything_cdna.parquet` protein split (104 train / 12 val / 18 test), exclusion CSVs, `comparison_labels_{train,test}.csv` and `external_overlap.csv`. The ESM-MSR processed directory is an order-only audit input.
3. `mutate_everything/external/inventory.json`: `mutate_everything_raw external` writes the S669, ProTherm-multiple, cDNA1 and S669-reverse label/exclusion CSVs.
4. `mutate-everything-holdouts`: `mutate_everything_raw holdouts` writes `splits/mutate_everything_cdna_holdout{1,2}.parquet`: the sorted unique development (non-test) sequences shuffled with `numpy.random.default_rng(seed)`, seed 1 / 2, the first `round(0.1 n)` val, test unchanged. Existing files are compared and kept if identical (the step fails if they differ); `FORCE=1` rewrites them. `mutate-everything` includes this step.

`mutant_cache/data/mutate_everything_raw.py` is the portable equivalent of `experiments/mutate_everything_prepare.py`, `experiments/mutate_everything_external_prepare.py` and the split block of `experiments/start_mutate_everything_accuracy_v2.py`. Those raw scripts hardcode the original scratch root and are kept as run; the Makefile does not call them.

The following files are **not** produced by these rules. They were written by other scripts and remain in place:

- `processed/mutate_everything_cdna/sign_and_overlap_audit.json` and `largest_external_label_differences.csv`. No recorded generating script was found.
- The comparator checkpoints `mutate_everything/mutate_everything_esm*.pth`, the upstream clone `mutate_everything/repo`, and the baseline predictions `mutate_everything/baseline/` and `external/baseline/`.

### ESM-MSR (`esm-msr`)

Upstream [SKTeamLab/esm-msr](https://github.com/SKTeamLab/esm-msr) @ `c34c315913b539070095b8af63b4bd7fbcd503b7`. Data: Zenodo [10.5281/zenodo.21539277](https://doi.org/10.5281/zenodo.21539277) v0.1, `preprocessed_data.tar.gz` (md5 `56ae9af3f800065b7a7ff4f4a42b7b92`).

- `esm-msr-raw` covers two inputs. The first is `esm_msr/preprocessed/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv`, extracted from the archive after an md5 check. The second is `esm_msr/repo/data/hyperopt_splits.csv`, from a git clone checked out at the pinned commit (sha256 `e3fccc65…959d` checked; set `MSR_REPO_URL=file:///path/to/clone` to clone from an existing copy). The archive must be fetched manually from the DOI page: the original fetch command was not recorded, and a guessed Zenodo API URL returned 404. `esm-msr-check` verifies the archive's md5 and sha256 (`9c36e94d…2086`); extraction also checks the CSV md5 `5810ab15…c040`.
- `esm-msr` runs `python -m mutant_cache.data.esm_msr`. It writes `processed/esm_msr_mega/{targets,variants,score_sets,wt_report}.parquet`, `splits/esm_msr_hyperopt.parquet` and `audit/` (exclusions, provenance, split summary, overlap, homology ≥30%).

Training on the released development partition is ready. A **matched external comparison is unresolved**. Four pieces are missing:

- **Checkpoint provenance:** the paper's HuggingFace checkpoints are not downloaded, and the bundled `esm-msr-small` is not the paper model.
- **Row mapping:** upstream silently drops mismatching constituents and groups by `code_wt`; we group by `target_id`, which gives 34 vs 36 test groups.
- **Metrics:** the per-protein pooled Spearman, nDCG@96 and `ndcg_t0` mirrors have not yet been verified against upstream reported numbers.
- **Overlap audit:** what their checkpoint trained on has not been audited.

`mutant_cache.matched_suite` is specialized to Mutate Everything and is not an ESM-MSR evaluator.

### StabilityArc (`stabilityarc`)

This target is planned only. No importer, release pin, splits or evaluator exist. `make -C datasets stabilityarc` exits nonzero, listing six missing pieces:

1. a pinned source release
2. an importer
3. materialized released outer protein holdouts
4. a development-only model-selection protocol
5. an overlap/label audit
6. reproduction of upstream evaluation on identical rows

`resolve_config` rejects the protocol because its status is `planned`.

## Verification (`verify`)

`make -C datasets verify` runs `python -m mutant_cache.data.verify_processed`, which recomputes a content hash for every processed table and JSON listed in [`processed_hashes.json`](processed_hashes.json) and a byte sha256 for the raw inputs (archives, cDNA CSVs, ESM-MSR CSV, `hyperopt_splits.csv`). Expected values were recorded from the original server's scratch on 2026-11-05. Restrict with `VERIFY_GROUPS="esm-msr mutate-everything"`.

- Tables (parquet/CSV): columns sorted by name, types reduced to `int<bits>`/`float<bits>`/`bool`/`str`, rows serialized as JSON (floats exact via `repr`, NaN/None as null) and sorted. The hash is therefore independent of the parquet writer, compression, row groups and row order. Row-order differences are reported as a note.
- JSON: re-serialized with sorted keys.
- Paths: strings under the scratch root are rewritten to `$MUTANT_CACHE_SCRATCH/...`. That covers `manifest.json` `files.*.path` and `external_dataset_dir`, `audit/provenance.json` `archive`/`source_csv`/`split_file`, and the `split_file` column of `external_overlap.csv`. No field is ignored, and none of these files contain timestamps.
- A file whose exact hash differs but whose floats agree to 10 significant digits is reported as `ROUNDED`, which still fails unless `--accept-rounded` is passed. It flags float noise from different library versions; investigate before accepting.

Not verified: `sign_and_overlap_audit.json` and `largest_external_label_differences.csv` (no generating script), and evaluation outputs under `mutate_everything/external/`.

## Protocol files

`protocols/*.json` are consumed by `mutant_cache.training.resolve_config`. Required fields:

```json
{"id": "...", "status": "ready" | "planned",
 "training": {"dataset": "<processed dir name, megascale --dataset>", "split": "<splits/<name>.parquet, megascale --split>"}}
```

Only `status: "ready"` protocols can train. `training.dataset` / `training.split` are fixed by the protocol; recipes and experiment overrides cannot change them. The remaining fields (source pins, partitions, eval panels, metrics, notes) are descriptive.

| Protocol | `dataset` | `split` |
| --- | --- | --- |
| `mutate_everything_v1` | `mutate_everything_cdna` | `mutate_everything_cdna` (internal 104/12 development split used for E02) |
| `esm_msr_v1` | `esm_msr_mega` | `esm_msr_hyperopt` (released hyperopt partition) |
| `stabilityarc_v1` | — (`training: null`) | — |

## Raw downloads

The older targets download raw files into `$(SCRATCH)/raw/<dataset>/` with resumable `curl -C -`. `make -C datasets check` verifies `checksums.md5`.

| Target | Source |
| --- | --- |
| `tsuboyama` | Zenodo 7844779 v2_230420: `Processed_K50_dG_datasets.zip`, `K50_dG_tables.zip` (`tsuboyama-raw-ngs` adds the 243 MB NGS counts) |
| `mavedb` | Zenodo 14172004 v2 (2024-11-14) bulk dump, 1.4 GB |
| `flip_gb1` | FLIP repo `splits/gb1` (the `data.bioembeddings.com` host 404s) |
| `proteingym` | `reference_files/DMS_substitutions.csv` only |
| `combingym` | shallow clone of `sitonglab/CombinGym` |

`make -C datasets all` covers tsuboyama, mavedb, flip_gb1 and proteingym.

## History

The original dataset shortlist and rationale (MaveDB pool, multi-mutant sets, held-out benchmarks, baselines) is archived at [`archive/2026-11-05/DATASETS.md`](../archive/2026-11-05/DATASETS.md).
