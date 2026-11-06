# Setting up a new server

Rebuild this repository's processed data from raw and run parallel training on another machine. Raw downloads and model weights are fetched on the new machine; nothing has to be copied from the original server. Paths come from `$MUTANT_CACHE_SCRATCH`, so the home directory can differ.

## 1. Code and environment

```bash
git clone git@github.com:amdson/mutant_cache.git && cd mutant_cache
conda env create -f env/environment.yml && conda activate mutant_cache
pip install -e .
export MUTANT_CACHE_SCRATCH=/path/to/scratch        # put it in your shell profile
CUDA_VISIBLE_DEVICES="" python -c "import torch, mutant_cache; print(torch.__version__, torch.version.cuda)"   # 2.11.0+cu130 13.0
```

You need NVIDIA driver 580 or newer (the CUDA 13.0 wheels). [env/README.md](../env/README.md) lists the full lock, the packages you can drop, and fallbacks without conda. If the env is not at `~/anaconda3/envs/mutant_cache`, pass `PYTHON=$(which python)` to the make commands below.

## 2. Model weights

Fetch the pinned revisions so results match the original server:

```bash
huggingface-cli download facebook/esm2_t33_650M_UR50D --revision 08e4846e537177426273712802403f7ba8261b6c
huggingface-cli download EvolutionaryScale/esmc-600m-2024-12 --revision d11cc14d44078eaecbc6a843d5eb20f4eecc1e7e   # only for ESMC recipes
huggingface-cli download EvolutionaryScale/esmc-300m-2024-12 --revision a19d363f07313a10a64d08a2d6b41376a73df5c8   # tests / harness default
```

The loaders read `main` from the cache, so download before running anything. The weight sha256s are in [env/README.md](../env/README.md). ESMC-600M is under a non-commercial licence. If you get a 401/403, accept the licence on its model page and set `HF_TOKEN`.

## 3. Data

The raw step for each dataset downloads or extracts it, then checks its checksums.

```bash
# ESM-MSR archive: manual fetch (no direct URL recorded)
#   https://doi.org/10.5281/zenodo.21539277 -> preprocessed_data.tar.gz
mkdir -p $MUTANT_CACHE_SCRATCH/esm_msr && mv preprocessed_data.tar.gz $MUTANT_CACHE_SCRATCH/esm_msr/
make -C datasets esm-msr-check                 # md5 + sha256 of the archive

make -n -C datasets esm-msr mutate-everything  # preview
make -C datasets esm-msr mutate-everything     # ME build audits against ESM-MSR, so both
make -C datasets verify                        # content hashes vs datasets/processed_hashes.json
```

- **Mutate Everything:** the zip comes from Google Drive. If the automatic fetch gets an HTML page, run `gdown 1akfHt1WLbC345DgZwZ5kLeMUADq9XB7y` inside `$MUTANT_CACHE_SCRATCH/mutate_everything/` and rerun.
- **Holdouts:** `mutate-everything` also builds the round-2 holdout1/2 splits, deterministically.
- **`verify`:** it should report every file as matching. `esm_msr_mega/audit/homology_ge30.csv` is marked UNSTABLE: an existing row-order nondeterminism in `targets.parquet` changes which background represents each wild type in this audit only. Splits and training tables are unaffected.
- **Rebuilds:** existing outputs are never overwritten without `FORCE=1`.
- **Not rebuilt:** the comparator checkpoints, baseline predictions and the two hand-made ME audit files (`sign_and_overlap_audit.json`, `largest_external_label_differences.csv`). `experiments/stability_paper/me_esm2_v1/comparison.json` uses absolute paths on the original server, so matched comparisons need their own config there. See [datasets/README.md](../datasets/README.md).
- **Optional:** `MMSEQS=/path/to/mmseqs`. These targets don't use it. The binary on the original server is commit `85fb9493`. Note that `cluster()` silently returns nothing if the binary is missing.

## 4. Run experiments in parallel

```bash
python -m mutant_cache.train --experiment experiments/stability_paper/me_esm2_v1/experiment.json --seed 0 --dry-run
python -m mutant_cache.queue_runner run experiments/stability_paper/jobs_example.json --dry-run
python -m mutant_cache.queue_runner run experiments/stability_paper/jobs_example.json      # all visible GPUs; --gpus 0,1 to restrict
python -m mutant_cache.queue_runner status experiments/stability_paper/jobs_example.json
```

- **What the runner does:**
  - Every job is checked before any launch.
  - Each GPU runs one job at a time.
  - Outputs go to `$MUTANT_CACHE_SCRATCH/queues/<jobs file>/runs/<job>`.
- **Rerunning:**
  - Finished jobs are skipped.
  - A failed job's directory is never overwritten. `--retry-failed` moves it aside.
- **Stopping:** workers outlive the runner, so stop them by pid from `queue_state.json`.
- **Code changes:** workers use the live checkout, so don't edit code while a queue runs. Each run still saves its own copy of the code.
- **Your own jobs:** write a jobs file with a list of `{name, experiment | protocol+recipe, seed, args}` entries, or `experiments: [{experiment, seeds}]`.
- **W&B:** off by default (`--wandb-mode online` to enable, with `WANDB_API_KEY`).
- **Before the first real queue:** run one short smoke job and check its output, for example with `--exposures 1` via `args`.

## 5. Check the install

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_train_cli.py tests/test_queue_runner.py tests/test_verify_processed.py
python -m pytest -q tests/test_backbone_equivalence.py   # GPU; compares backbone ports to the official esm package
```
