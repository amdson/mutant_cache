The [W&B project preset](wandb_tracking.json) now enables sampled diagnostics
for future Megascale/tuning runs. Standalone runs enable them with
`--diagnostics`. Diagnostics can also be recorded locally with W&B disabled.
The active `mutate_everything_tuning_v1` sweep retains its frozen code and plan.

| Measurement | Default schedule / bound |
| --- | --- |
| Training loss and actual learning rates | Every 50 optimizer steps |
| Gradient norms, clipping, actual parameter updates | Every 50 steps and the last budget step; clipping statistics cover the intervening steps |
| Clean training-probe loss and all evaluation metrics | Alongside every validation |
| Validation objective, calibration and per-protein summaries | Every validation; existing validation metrics continue at their original cadence |
| Per-layer/head activation profiles and adapter/base ratios | At initialization, every fifth validation, and the last budget validation |
| Prediction/interaction plots and protein tables | Same sparse schedule as activation profiles |
| Metrics for the selected checkpoint | Once after restoring it, labelled `selected/*` |

The fixed training probe uses at most **8 proteins × 128 rows = 1,024 measured
training rows**. Its private seed is 0 for every training seed. Protein selection
and row ordering are deterministic. Up to a quarter of each protein's row
budget selects doubles with both measured singles; their constituent rows are
included. Remaining space is balanced by mutation depth and score >= 0. The
exact manifest and its SHA-256 are saved. This is a deliberately stratified
diagnostic probe, not a full-training-set performance estimate. In particular,
its retrieval scores rank a smaller candidate library than the full dataset.

Activation profiling uses at most **2 training proteins × 32 rows**. Hooks are
installed only for those evaluation forwards and removed afterward. RMS values
exclude dense padding and the packed attention implementation's dummy tokens.
Profiles follow the model's composition: an additive model's head sees the
constituent singles. Each adapter/projection ratio compares its output with the
corresponding base linear projection on the same inputs. The reference pass is
primed before installing variant-stream hooks, including with shared adapters.

Actual update sizes use fixed, evenly spaced coordinates: at most **8,192 per
optimizer group**, sampled before and after that optimizer step. These are real
optimizer updates, not learning-rate-times-gradient estimates. RMS updates,
relative updates, and distance from initialization are explicitly labelled
`*_sampled`. Group gradient/parameter norms use the full tensors. Frozen base
weights are not copied for this instrumentation. LoRA A/B gradient norms are
separate; an initially zero B naturally gives A no initial gradient.

Useful chart names:

| Prefix / field | Interpretation |
| --- | --- |
| `train/loss_running`, `train/loss` | Actual sampled/augmented training objective, running and validation-interval averages |
| `train_probe/loss`, `val/loss` | Same configured objective on clean references, averaged equally over targets |
| `train_probe/all_spearman`, `val/all_spearman` | Macro per-protein correlations |
| `*/all_spearman_pooled`, `*/double_rmse_pooled` | Explicitly labelled pooled metrics |
| `*/single_*`, `*/double_*`, `*/epistasis_*` | Depth-specific error and correlation metrics |
| `*/add_*` | Additive readout of the same joint checkpoint |
| `*/ndcg30_*`, `*/precision30_*`, `*/retrieval96/*` | Existing benchmark detection/retrieval definitions |
| `*/*_protein_median`, `*/*_protein_p10`, `*/*_protein_p90` | Distribution across proteins; counts report how many metrics are defined |
| `*/calibration/*` | Prediction/target mean and standard deviation, bias, scale ratio and observed-on-predicted fit |
| `*/class/*` | Errors and counts for score >= 0 versus score < 0 |
| `*/interaction/*` | Nonadditivity, complete measured groups and paired macro double-RMSE gain over the additive readout |
| `optimization/*/grad_norm_pre_clip` | Full gradient norm per head/variant/shared optimizer group |
| `optimization/clipped_fraction` | Fraction of finite pre-clip norms exceeding the clip threshold in the latest window |
| `optimization/*/update_rms_sampled`, `*/relative_update_sampled` | Actual sampled parameter changes |
| `optimization/*/missing_gradient_fraction`, `*/zero_gradient_fraction`, `*/nonfinite_gradient_elements` | Gradient coverage and numerical failures |
| `activation/layer_*/pre_attention/rms`, `*/post_attention/rms` | Residual activation scales by layer |
| `activation/head/*`, `activation/final_hidden/*` | Readout input/hidden scales and final token representations |
| `activation/layer_*/*/delta_to_base_rms` | Relative effect of each variant/shared adapter projection |
| `sampling/*` | Actual row/depth/class/reference exposure; both cumulative and recent-window fractions |
| `performance/*` | Training row/token rates and measured diagnostic-evaluation time |
| `memory/*` | Allocated/reserved GPU memory at sampled steps, plus final peaks |

The clean-reference loss uses the configured MSE/Huber weights and Huber delta.
It is evaluated across all clean probe/validation depths even during a
singles-only phase. It does not include stochastic class subsampling or
reference augmentation. Thus `train_probe/loss` versus `val/loss` is the clean
comparison; sampled `train/loss` additionally reflects the active curriculum
and augmentation. Unweighted RMSE/MAE remain comparable across objective choices.
For ranking objectives, diagnostic loss uses bounded per-target chunks because
the comparison group defines the loss.

Prediction scatter plots contain at most **512 points per split**, sampled
across proteins/depths with the private seed. Per-protein W&B tables have the
same row cap; all per-protein metrics are preserved in the local CSV. Complete
interaction groups also produce an observed-versus-predicted residual plot.
The media contain observed/predicted values and identifiers, not sequences or
checkpoints. W&B stores the backing tables as table artifacts.

Diagnostics preserve Python, NumPy and PyTorch RNG states, model mode and
reference caches. Extra evaluations run without gradients. They do not read
external test data or feed checkpoint selection, stage promotion or stopping.
No fixed overhead percentage is assumed: the extra train forwards are bounded,
and diagnostic time is recorded for the actual workload.

Every `diagnostics/` folder contains:

- `settings.json` and `train_probe.csv`: exact budgets, seed, members and hash.
- `history.jsonl`: scalar records with optimizer step and measurement kind.
- `per_protein.csv`: all train-probe/validation protein metrics by step.
- `*_scatter_step_*.csv`: bounded plotted prediction samples.
- `reference_exposure.csv`: actual cumulative rows per reference.

The preset records all settings explicitly. Adjust them through `diagnostic_*`
JSON fields or the corresponding `--diagnostic-*` CLI flags. A media checkpoint
also evaluates the training probe if it falls between configured probe checks.

Validation: 47 relevant tests passed, including numerical oracles, scheduling checks and exact training
parity with diagnostics off, including dropout, curriculum, shared adapters and
background augmentation. A real 33-layer ESM-2 650M bf16 preflight passed:
[script](diagnostics_esm2_preflight.py) ·
[result](/home/andrew/mutant_cache_scratch/runs/diagnostics_esm2_preflight_20261104/smoke_result.json).
The full online scalar/media path also passed:
[script](wandb_diagnostics_preflight.py) ·
[W&B test run](https://forge.coreweave.com/wandb/amdickson128-usda/mutant_cache/runs/b8152affc70c4d24).
