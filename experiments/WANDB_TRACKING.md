Future supervised Megascale/tuning runs can stream progress to
[amdickson128-usda/mutant_cache](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache).
Tracking is optional and defaults to `disabled`.

Online access was verified on 2026-11-04. The SDK's saved login works in a
fresh process without sourcing `.bashrc`. The isolated environment below is
ready to launch future runs with the checked-in tracking preset.

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
--wandb-group STUDY_NAME --wandb-name RUN_NAME --wandb-log-every 50 --diagnostics
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
metrics and configuration. The [diagnostic preset](DIAGNOSTICS.md) additionally
uploads bounded scatter-plot data and per-protein tables; it does not upload
sequences, model files or full prediction tables.

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
and plan. These hooks apply to future initializations; its workers were not
restarted or modified during authentication setup.

The exact integration preflight is [wandb_tracking_smoke.py](wandb_tracking_smoke.py).
Validation on 2026-11-04: 21 tracking/queue/training tests passed, including exact
prediction parity with tracking disabled. A complete CPU training run passed
with W&B 0.31.0 offline; inspection of its transaction log confirmed validation
metrics, learning rates, selected metrics and a successful exit. Its local
[smoke result](/home/andrew/mutant_cache_scratch/runs/wandb_tracking_preflight_20261104/smoke_result.json)
records the SDK version and artifact paths.

The [online training check](wandb_tracking_online_smoke.py) authenticated and
uploaded a tiny synthetic CPU run. Its initial completion poll saw a cached
running state; the [fresh-process verification](wandb_tracking_online_verify.py)
refreshes the server response and verifies the saved login, finished state,
configuration, validation metrics and selected checkpoint. See the
[dashboard test run](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/f456e5764fda4d13)
and [online verification record](/home/andrew/mutant_cache_scratch/runs/wandb_tracking_online_preflight_20261104/smoke_result.json).
This is tagged `integration-test`/`synthetic` and grouped under
`tracking-verification`, separate from scientific experiments.

Implementation follows the official SDK's
[initialization API](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py)
and [run logging API](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py).
