Future supervised Megascale/tuning runs can stream progress to
[amdickson128-usda/mutant_cache](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache).
Tracking is optional and defaults to `disabled`.

Install and authenticate in the environment that will launch those runs:

```bash
python -m pip install -e '.[tracking]'
wandb login
```

The SDK has also been installed and tested in an isolated environment at
`/home/andrew/mutant_cache_scratch/wandb_validation_venv`, using the main
environment's existing training dependencies. To use that environment now:

```bash
source /home/andrew/mutant_cache_scratch/wandb_validation_venv/bin/activate
wandb login
```

Login is local: use the W&B CLI prompt or the `WANDB_API_KEY` environment
variable. Credentials do not belong in experiment plans. The optional SDK
dependency starts at 0.31, whose default API endpoint is
`https://api.forge.coreweave.com`; its dashboard URLs use the Forge host.

For a future queue, add this argument to its initialization command:

```bash
python -m mutant_cache.tuning init --root NEW_RUN_ROOT --plan NEW_PLAN_JSON \
  --template TEMPLATE_CHECKPOINT --tracking-config experiments/wandb_tracking.json
```

The readable [tracking preset](wandb_tracking.json) selects the project above.
Initialization merges it into the base training arguments, freezes a copy beside
the plan, and records its hash. Tracking options can alternatively be specified
in the plan's `base_overrides` as `wandb_*` fields. An explicit tracking preset
overrides those base fields; per-trial overrides still apply afterward.

For a standalone `python -m mutant_cache.megascale` run, append:

```bash
--wandb-mode online --wandb-project mutant_cache --wandb-entity amdickson128-usda \
--wandb-group STUDY_NAME --wandb-name RUN_NAME --wandb-log-every 50
```

The queue automatically supplies the sweep folder name as the W&B group, names
such as `B01-s0-attempt1`, and a stage tag. Each attempt has its own W&B run ID;
retries start fresh. Resolved hyperparameters, parent ID, seed, plan hash and
source/data hashes are recorded as configuration. `WANDB_PROJECT` and
`WANDB_ENTITY` provide CLI defaults; explicit settings override them.

What to inspect in W&B:

| Charts / fields | Meaning |
| --- | --- |
| `train/loss_running`, `train/loss` | Running loss and loss at validation checkpoints |
| `val/all_spearman`, `val/double_spearman`, `val/single_spearman` | Macro per-protein validation correlations |
| `val/double_rmse`, `val/ndcg30_double`, `val/precision30_double` | Validation error and retrieval quality |
| `val/add_*` | Additive readout from the same joint model |
| `lr/head`, `lr/variant_lora`, `lr/shared_lora` | Actual learning rates of active parameter groups |
| `progress/*` | Exposure, rows, fraction complete, elapsed time and curriculum phase |
| `selection/*/step` | Best checkpoint step for each saved selection criterion |
| `selected/val/*` | Final selected checkpoint metrics, with macro/pooled labels |
| `timing/*`, `memory/*`, `model/*` | Final timing, memory and parameter counts |

All charts use `trainer/step` as the x-axis. Undefined metrics are omitted.
Group/filter by the sweep, stage, seed or configuration fields to compare runs.
The local status page links online runs directly; `tracking.json` in each run
directory records its ID, URL, mode, SDK version and completion status.
Local histories and checkpoints continue to be saved. Tracking uploads scalar
metrics and configuration; it does not upload prediction tables or model files.

Offline testing and later synchronization are also supported:

```bash
# Use --wandb-mode offline in the CLI or "wandb_mode": "offline" in the preset.
wandb sync /absolute/path/to/run/wandb/offline-run-*
```

An unavailable SDK or failed initialization stops a tracking-enabled run before
training begins. A logging error after initialization is reported locally and
training continues; the dashboard may then be incomplete. Training failures
finish the W&B run with a failure exit code.

The already-launched `mutate_everything_tuning_v1` sweep uses its frozen source
and plan. These hooks apply to future initializations. Local authentication is
still needed before an online launch.

The exact integration preflight is [wandb_tracking_smoke.py](wandb_tracking_smoke.py).
Implementation follows the official SDK's
[initialization API](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py)
and [run logging API](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py).
