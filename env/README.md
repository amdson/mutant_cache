# Environment and dependency inventory (snapshot 2026-11-05)

Taken from the source machine: Linux x86_64, 3x RTX A6000, NVIDIA driver
580.105.08 (CUDA 13.0), Xeon Platinum 8352Y. Repo HEAD was `4f74a31`.

Files in this directory:
- `environment.yml`: `conda env export --no-builds`, with the prefix line removed.
- `requirements-lock.txt`: `pip freeze`. Entries pointing at local paths are replaced by comments.

## Install

```bash
conda env create -f env/environment.yml          # python 3.12.15 + pip lock (PyPI only)
conda activate mutant_cache
pip install -e .                                  # this repo; add ".[tracking]" for the wandb extra
# or, without conda: python3.12 -m venv .venv && pip install -r env/requirements-lock.txt && pip install -e .
CUDA_VISIBLE_DEVICES="" python -c "import torch, mutant_cache; print(torch.__version__, torch.version.cuda)"  # expect 2.11.0+cu130 13.0
```

- **Python** 3.12.15 (conda-forge). `pyproject.toml` asks for `>=3.10`.
- **The env was created fresh, not cloned from svar.** `conda-meta/history`
  shows only `conda create -n mutant_cache python=3.12`. The conda layer is
  python and base libraries. The other 163 packages came from pip/PyPI.
- **torch 2.11.0** is the default PyPI Linux wheel, built for CUDA 13.0 (it
  reports `2.11.0+cu130`, cuDNN 9.19, triton 3.6.0). It pulls in
  `cuda-toolkit==13.0.2` and the `nvidia-*-cu13` wheels, so the system needs
  no CUDA toolkit. It does need an **NVIDIA driver >= 580** (the CUDA 13.0
  minimum). If pip on the new machine resolves torch to a different CUDA
  build, add `--index-url https://download.pytorch.org/whl/cu130`. bf16
  recipes need an Ampere-or-newer GPU.
- **The repo itself** is an editable install (`direct_url.json` ->
  `file:///home/andrew/mutant_cache`). On the new machine run `pip install -e .`
  from the clone. `pyproject.toml` declares **no runtime dependencies**, so
  `pip install -e .` on its own does not install torch and the rest. Use the
  lock.
- **Entries that would not resolve elsewhere, and how they were handled:**
  - `-e git+ssh://...mutant_cache@4f74a31`: replaced with a comment.
  - `packaging @ file:///home/conda/feedstock_root/...`: a conda build path,
    replaced with `packaging==26.3`.
  - No other git or local-path installs exist.
- **What the code imports at runtime:**
  - training: torch, numpy, pandas, pyarrow (parquet), scipy, scikit-learn,
    transformers (loads ESM-2), huggingface_hub (loads ESMC), safetensors.
  - data builders: polars, biopython (`Bio`).
  - plots: matplotlib.
  - tests: pytest and the `esm` package (`esm` is used only by
    `tests/test_backbone_equivalence.py`).
- **Heavy packages that training does not need.** All are pulled in by
  `esm==3.4.1.post1`, and all are kept in the lock:
  - rdkit
  - cuequivariance / cuequivariance-ops(-torch)-cu13 (about 280 MB, CUDA-13 specific)
  - boto3/botocore
  - biotite, pydssp, py3Dmol, dna_features_viewer, accelerate, ipython/ipywidgets

  wandb pulls in opentelemetry-* and cwevals. Jupyter tooling
  (ipykernel, nbconvert, ...) was installed directly.
  Nothing from svar (CARBON/DNABERT) is in this env.

## Model weights

All weights come from the Hugging Face Hub (default cache `~/.cache/huggingface/hub`).
Nothing is fetched through torch hub (`~/.cache/torch` does not exist). The
ESMC loader tries `local_files_only=True` first, then downloads.
The ESM-2 loader calls `from_pretrained(f"facebook/{name}")`, which downloads
on a cache miss. Neither loader pins a revision: both use `main`. Pin with
`huggingface-cli download <repo> --revision <commit>`.

| Used by | Loader | HF repo | Revision (cached `refs/main`) | File | Bytes | sha256 |
|---|---|---|---|---|---|---|
| `esm2_650m_shared.json` recipe (`esm2_t33_650M_UR50D`) | `mutant_cache/backbone/esm2.py:load_esm2` via `transformers.EsmForMaskedLM.from_pretrained(..., attn_implementation="eager")`, converted to the local ESM2 | `facebook/esm2_t33_650M_UR50D` | `08e4846e537177426273712802403f7ba8261b6c` | `model.safetensors` | 2609506392 | `a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0` |
| | | | | `config.json` | 724 | `539095c22efc52a09d6147074ba4ca119f76a890df5901213b2b55f7d2f96b2b` |
| | | | | `vocab.txt` | 93 | `0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03` |
| | | | | `tokenizer_config.json` | 95 | `7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad` |
| | | | | `special_tokens_map.json` | 125 | `3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1` |
| `esmc_600m_shared.json` recipe (`esmc_600m`) | `mutant_cache/backbone/esmc.py:load_esmc`: `snapshot_download(hf_repo)` + `torch.load(<snap>/data/weights/...pth, weights_only=True)`, strict load into the local ESMC. The local tokenizer is hardcoded, so no tokenizer files are needed. | `EvolutionaryScale/esmc-600m-2024-12` | `d11cc14d44078eaecbc6a843d5eb20f4eecc1e7e` | `data/weights/esmc_600m_2024_12_v0.pth` | 2300275866 | `8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324` |
| harness default `--backbone esmc_300m`; all cache/equivalence tests | same as the row above | `EvolutionaryScale/esmc-300m-2024-12` | `a19d363f07313a10a64d08a2d6b41376a73df5c8` | `data/weights/esmc_300m_2024_12_v0.pth` | 1332095738 | `323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea` |

Other models are in the HF cache but **are not referenced** by `mutant_cache/`,
`tests/`, recipes or protocols. They are left over from svar and other
projects, and are not needed:
- `facebook/esm2_t6_8M_UR50D` (config and tokenizer only)
- `biohub/ESMC-300M`
- `Synthyra/{ESM2-650M,ESMplusplus_small,FastESM2_650}`
- `GrimSqueaker/proteinBERT`
- `HuggingFaceBio/Carbon-*`
- `Qwen/Qwen3-4B-Base`
- `songlab/gpn-brassicales`
- `zhihan1996/DNABERT-2-117M`

The tests compare against the official `esm` package. They do not download
weights through it: they rebuild `esm`'s ESMC classes and load the same
`esmc_300m` `.pth` shown above.

**Licensing and gating.**
- `esmc-300m-2024-12` is under the EvolutionaryScale *Cambrian Open License*.
- `esmc-600m-2024-12` is under the *Cambrian Non-Commercial License*.
- The cached model cards show no gating flag, and this machine has no stored
  HF token (`~/.cache/huggingface/token` is absent). The files were therefore
  probably downloaded anonymously.
- This is not verified online. If a download returns 401/403, accept the
  terms on the model page and run `huggingface-cli login` (or set `HF_TOKEN`).
- `facebook/esm2_*` is MIT and ungated.

Verify a reconstruction with:
`sha256sum $(python -c "from mutant_cache.backbone import esmc_weights_path as p; print(p('esmc_600m'))")`.

## External tools and services

- **MMseqs2** (used for sequence clustering in the exclusion list and splits,
  `mutant_cache/data/cluster.py`):
  - **The binary path is hardcoded:** `MMSEQS = "/home/andrew/MMseqs2/build/bin/mmseqs"`.
  - Source build of `github.com/soedinglab/MMseqs2`, commit
    `85fb949328613918c6ce715c3ac2b5a97e2e85c8`. `git describe` gives
    `18-8cc5c-2-g85fb9493` (release 18 plus 2 commits), and `mmseqs version`
    prints the same hash.
  - Built with CMake `RELEASE`, `NATIVE_ARCH=1` (host AVX-512 machine),
    `ENABLE_CUDA=1`, no MPI, using the conda `nvcc` env compiler.
  - The binary is tuned to this CPU, so **rebuild it on the new machine**
    rather than copying it. Release 18 binaries (`mmseqs-linux-avx2.tar.gz`)
    should match the clustering algorithm. The 2 extra commits are not
    checked.
  - Called as `easy-cluster --min-seq-id S -c C --cov-mode M -s SENS --cluster-mode 0`.
  - If the binary is missing, `cluster()` returns `None` and callers fall back
    silently. **Splits built without it can differ.**
- **git** must be installed:
  - `training.py`, `harness.py` and `data/esm_msr.py` call `git rev-parse HEAD`
    for provenance. Failure is tolerated: the commit is recorded as `null`.
  - `datasets/Makefile` clones upstream repos:
    - CombinGym
    - `SKTeamLab/esm-msr` @ `c34c3159...`
    - MutateEverything @ `034dd91e...`
- **wget, curl, unzip, tar, gunzip**: used by `datasets/Makefile` for
  downloads.
- **git-lfs**: not installed here and not referenced.
- **W&B is optional.** The `--wandb-mode` flag (in `mutant_cache/tracking.py`,
  and so also in `python -m mutant_cache.train` and `megascale`) **defaults to
  `disabled`**. The SDK is imported only when the mode is `online` or
  `offline`. If wandb is missing in those modes, the run raises with an
  install hint.
  - **Train CLI runs do not use the preset by default.** The preset
    `experiments/wandb_tracking.json` (`online`, entity `amdickson128-usda`,
    diagnostics on) is applied only through
    `python -m mutant_cache.tuning init --tracking-config ...`.
  - To enable tracking: `pip install -e ".[tracking]"`, then `wandb login` or
    set `WANDB_API_KEY`. wandb>=0.31 defaults to the CoreWeave Forge endpoint
    `api.forge.coreweave.com`.
  - To force it off: omit `--wandb-mode`, pass `--wandb-mode disabled`, or set
    `WANDB_MODE=disabled`.
  - If logging fails mid-run, training carries on.
- **Network at train time:** only the HF downloads on the first load of each
  backbone. After that, set `HF_HUB_OFFLINE=1` to guarantee cached weights are
  used.

## Environment variables

| Var | Effect |
|---|---|
| `MUTANT_CACHE_SCRATCH` | Root for raw, processed and runs (`processed/<dataset>/`, `runs/`, reference caches). Default `~/mutant_cache_scratch`. Read in `harness.scratch_root`, `data/paths.py`, `cache/reference_cache.py`, `models/data_compat.py`. |
| `HF_HOME` / `HF_HUB_CACHE` | Where the backbone weights are cached/looked up. Default `~/.cache/huggingface`. Unset on the source machine. |
| `HF_TOKEN` | Only if ESMC turns out to be gated. |
| `HF_HUB_OFFLINE=1` | Optional. Stops the loaders from reaching the network. |
| `TORCH_HOME` | Unused. |
| `WANDB_PROJECT`, `WANDB_ENTITY` | Defaults for `--wandb-project/--wandb-entity`. |
| `WANDB_API_KEY`, `WANDB_MODE`, `WANDB_DIR` | Standard SDK variables. A key is exported in the source shell profile. Do not copy it into the repo. |
| `CUDA_VISIBLE_DEVICES` | GPU selection. `--device` defaults to `cuda` if available, otherwise `cpu`. |

## Portability issues (new entry points)

- `mutant_cache/train.py`, `training.py`, harness data resolution: **no
  hardcoded machine paths.**
  - Data comes from `$MUTANT_CACHE_SCRATCH/processed/<protocol.training.dataset>/splits/<split>.parquet`.
  - Output goes to `--output`, or `--out-root`, or `$MUTANT_CACHE_SCRATCH/runs/<dataset>_<label>_s<seed>_<uuid>`.
  - Provenance records absolute paths, a git SHA, and sha256 hashes of the
    code and data. The data hashes cover every file under the dataset dir, so
    a large dir makes run start-up slow.
- `experiments/recipes/*.json`, `datasets/protocols/*.json`,
  `experiments/stability_paper/*/experiment.json`: **portable.** Protocol and
  recipe paths are relative to the experiment file.
- `experiments/stability_paper/me_esm2_v1/comparison.json`: **16 absolute
  `/home/andrew/mutant_cache_scratch/...` paths.**
  - Labels and baselines under `processed/mutate_everything_cdna/` and
    `mutate_everything/external/...`.
  - Frozen checkpoints under
    `runs/mutate_everything_full_comparison_v1/runs/{control,E02}_s{0,1,2}/model.pt`.
  - These must be rewritten for the new scratch root. The checkpoints exist
    only on the source machine; regenerate them or treat them as unavailable.
- `mutant_cache/data/cluster.py`: hardcoded MMseqs2 path (see above). It
  fails silently if the binary is absent.
- `experiments/WANDB_TRACKING.md` refers to a source-only venv at
  `/home/andrew/mutant_cache_scratch/wandb_validation_venv`. This doesn't
  matter, since the main env already has wandb 0.31.0.
- `README.md` / `CLAUDE.md` refer to `~/anaconda3/envs/mutant_cache/bin/python`.
  Any env made from this lock works.
- Several docstrings refer to `~/svar/...` as the method's origin. These are
  comments only, with no code dependency.
