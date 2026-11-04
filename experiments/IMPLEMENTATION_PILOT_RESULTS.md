# Implementation pilot (TODO.md, 2026-11-01)

Status (2026-11-02): **E0-E4 complete** (E2 reference V_1e-3; E3 head H2 sum_mean_count;
E4 transfer run). bf16 deployment comparison and E5 not run. All of this is pilot scale
(320/1,280 labels, 3 replicates) and provisional until checked along a scaling curve. Every Spearman below is on a DEVELOPMENT holdout
(`*_tune_ds{N}_r{r}` manifests, stored by the harness as split `test`), never on a METL
supertest, unless a row says "historical, supertest".

## E4 — H0 mean vs H2 sum_mean_count across data regimes (2026-11-02)

Protocol (`e4_transfer.sh`, `REF=V_1e-3 HEAD=sum_mean_count`; driver log
`e4_transfer_driver.log`; paired tables `e4_transfer_summary.txt` from
`summarize_e4_transfer.py`): E3's setup copied without retuning (400 steps, warmup 40, val
every 20) to avGFP-320, GB1-320, GB1-1280, replicates 0-2, both heads rerun (18 runs, all rc=0,
≈ 29 GPU-minutes). Train/val 256/64 or 1,024/256, dev 5,000 everywhere. GB1 dev is all doubles
(+8 singles), so its overall ρ is the depth-2 ρ.

| task | H0 dev ρ (0/1/2) | H2 dev ρ (0/1/2) | paired H2 − H0 per rep | mean | wins | best steps |
|---|---|---|---|---|---|---|
| avGFP-320 | 0.458 / 0.399 / 0.396 | 0.639 / 0.614 / 0.635 | +0.181 / +0.215 / +0.238 | +0.211 | 3/3 | 20 (all) |
| GB1-320 | 0.757 / 0.698 / 0.818 | 0.774 / 0.722 / 0.756 | +0.017 / +0.024 / −0.063 | −0.007 | 2/3 | 20-40 |
| GB1-1280 | 0.913 / 0.914 / 0.892 | 0.895 / 0.912 / 0.909 | −0.019 / −0.002 / +0.017 | −0.001 | 1/3 | 40-60 |

avGFP-320 within depth (paired mean): d1 −0.020, d2 +0.003, d3 +0.024, d4 +0.013, d5+ +0.059.
Cost identical between heads (avGFP 135 s / 5.4 GB; GB1 71 s / 3.9 GB).

Readings (pilot scale: 320-1,280 labels, 3 reps, untuned schedule; read nothing here as
settled until checked along a scaling curve):
- **avGFP-320: the +0.21 gain is almost entirely between-depth.** Within depth it is ≈ 0 at d1-d2
  and modest at d3+. Every run selected step 20, the first eval and still inside warmup.
- **GB1: neutral.** Mean differences −0.007 / −0.001, inside replicate spread (H0 alone spans
  0.70-0.82 at 320). Two paired losses exceed the 0.01 review flag (GB1-320 rep 2 −0.063,
  GB1-1280 rep 0 −0.019), with gains of similar size on other reps. GB1 dev is single-depth,
  so sum = 2·mean and count is constant: H2 can only change GB1 through optimization, making
  GB1 a weak test of H2 rather than a confirmation or refutation.
- Combined with E3: H2's benefit is largest where depth varies and labels are scarce, and is
  mostly (avGFP-320) or partly (avGFP-1280, +0.03-0.04 within depth) a depth-ordering effect.
- Not run: the bf16-base deployment comparison and E5 controls.

**Status: investigate, not keep.** H2 passes the provisional triage on avGFP (≥ +0.02, 3/3)
without a material mean GB1 regression, but the evidence is small-scale and largely
between-depth.

## E3 — prediction head, avGFP ds1280, reference V_1e-3 (2026-11-02)

Protocol (`e3_avgfp_heads.sh`, `REF=V_1e-3`; driver logs `e3_avgfp_heads_driver.log`,
`e3_mut_proj_driver.log`): identical to E2's V_1e-3 arm (variant LoRA r8/α16, q,o,gate,up,down,
adapter LR 1e-3, head LR 3e-3, fp32, 400 steps, cosine, resolved warmup 40 — checked in every
record — val every 20, restore lowest val rank loss), same three dev manifests, `--seed r`.
Only `--head` differs. H0 = the three E2 V_1e-3 runs, reused: same working-tree diff hash
(3368dc81fff9) as every E3 run, so same code. Train/val/dev 1,024/256/5,000 in every run.
Paired tables: `e3_heads_summary.txt` (`summarize_e3_heads.py`), curves `e3_val_curves.csv`.

| head | dev ρ per rep (0/1/2) | mean | paired − H0 per rep | wins | d1 / d2 / d3 / d4 / d5+ paired − H0 (mean) | head params |
|---|---|---|---|---|---|---|
| H0 mean | 0.707 / 0.677 / 0.680 | 0.688 | — | — | — | 123,137 |
| H1 mean_count | 0.728 / 0.732 / 0.723 | 0.728 | +0.020 / +0.055 / +0.043 | 3/3 | −0.008 / +0.016 / +0.016 / +0.016 / +0.016 | 123,265 |
| H2 sum_mean_count | 0.784 / 0.767 / 0.767 | 0.773 | +0.076 / +0.090 / +0.087 | 3/3 | +0.005 / +0.038 / +0.038 / +0.034 / +0.029 | 246,145 |
| H3 wt_relative | 0.780 / 0.779 / 0.770 | 0.776 | +0.073 / +0.102 / +0.090 | 3/3 | +0.048 / +0.046 / +0.051 / +0.045 / +0.033 | 406,353 |
| control mut_proj (w 325) | 0.788 / 0.777 / 0.773 | 0.779 | +0.081 / +0.100 / +0.093 | 3/3 | +0.040 / +0.054 / +0.055 / +0.045 / +0.033 | 406,646 |

Every run selects step 40 (end of warmup). Cost is indistinguishable across heads: cold wall
132-138 s, 873-914 train variants/s, peak 5.6-5.9 GB allocated / 6.2-6.6 GB reserved
(H0 135 s, 5.8 GB). Mean best val loss: H0 0.545, H1 0.511, H2 0.485, H3 0.474, control 0.485.
Depth-1 dev bin has n = 129, so d1 differences are the noisiest. Total E3 cost: 12 runs, ≈ 28
GPU-minutes.

Head-vs-head (paired, mean over reps; per-rep in parentheses):
- H3 − H2: +0.004 overall (−0.004 / +0.011 / +0.003), within-depth d1 +0.043, d2-d4 +0.008 to
  +0.013, d5+ +0.004.
- H3 − control: −0.003 overall (−0.008 / +0.002 / −0.003); within depth −0.008 to +0.009.
- control − H2: +0.006 overall (+0.004 / +0.009 / +0.006), within-depth +0.004 to +0.035.

Readings (pilot, three data/training replicates — not a significance test):
- **Count alone (H1) helps a little and mostly between depths**: +0.040 overall, +0.016 within
  each depth ≥ 2, slightly negative at d1.
- **H2 (sum + mean + count) is a large, repeatable gain that is not only between-depth**: +0.085
  overall on all three reps, positive within every depth ≥ 2 on every rep (+0.014 to +0.053,
  mean +0.03-0.04); d1 is mixed (+0.040 / −0.023 / −0.002).
  Sum = k·mean, so this is a useful explicit computation, not new information.
- **WT information does not explain H3's gain.** The parameter-matched mutation-only projection
  (no WT states, same ref/alt embeddings and pooling) is as good or slightly better (−0.003).
  The extra over H2 (≈ +0.004-0.006 overall, mostly at d1) comes from the per-token projection /
  capacity and ref/alt embeddings, and is below the +0.02 triage threshold.
- All heads still select at the end of warmup and then overfit (val loss rises from step 60
  while train loss → 0.01); val ρ for H2/H3 stays flat around 0.75-0.77 after step 40. The head
  change does not fix the early-overfitting pattern seen in E2.

**Gate decision (pre-registered rule: simplest head with a repeatable benefit): H2
`sum_mean_count`.** H3 → reject (WT-relative inductive bias not supported by its control);
token-projection capacity → investigate later (small, consistent +0.006 at 1.65× head params).
E4 run with `REF=V_1e-3 HEAD=sum_mean_count`.

## E2 — avGFP training/adapter scope (original minimal mean head fixed)

Protocol (`e2_avgfp_scope_screen.sh`, `e2_avgfp_scope_reps.sh`): avGFP ds1280 dev manifests,
ESM-C 300M, sparse ragged cache, fp32 base and adapters, uniform RankNet on raw `score`, head
LR 3e-3, AdamW wd 0.01, clip 1, batch 256, exactly 400 optimizer steps (cosine, warmup 40),
val every 20 steps + final, restore lowest val rank loss. Train/val/dev = 1,024 / 256 / 5,000
in every run (asserted from the record). LoRA r 8, alpha 16, targets q,o,gate,up,down. Per-run
table: `experiments/pilot_runs.csv` (`summarize_pilot.py`). The anchor F_1e-5 is the
historical avGFP setting (100 epochs × 4 batches = 400 steps then too); differences from
history: val every 20 steps instead of every 4, raw `score` instead of global `score_rank`
(same ordering), dev holdout instead of supertest — so it is a re-run, not an exact
reproduction.

**Screen, replicate 0** (dev Spearman; best val loss in brackets; selected = lowest val loss):

| arm | LR | dev ρ | best val loss | best step |
|---|---|---|---|---|
| F | 3e-6 | 0.566 | 0.630 | 60 |
| F | **1e-5** (selected) | 0.533 | 0.605 | 40 |
| F | 3e-5 | 0.586 | 0.632 | 40 |
| V | 1e-4 | 0.548 | 0.612 | 60 |
| V | 3e-4 | 0.676 | 0.593 | 60 |
| V | **1e-3** (selected) | 0.707 | 0.546 | 40 |
| V | 3e-3 | 0.736 | 0.582 | 100 |
| S | 1e-4 | 0.548 | 0.628 | 60 |
| S | 3e-4 | 0.527 | 0.610 | 40 |
| S | **1e-3** (selected) | 0.609 | 0.598 | 40 |
| S | 3e-3 | 0.627 | 0.615 | 40 |

Within F and S the val-loss ranking does not match the dev ranking (F_1e-5 has the lowest val
loss and the lowest dev ρ); the pre-registered criterion was kept anyway.

**Selected settings, replicates 0 / 1 / 2:**

| arm | dev ρ per rep | mean | paired vs F | d1 / d2 / d3 / d4 / d5+ (means) | best step | cold wall | train var/s | peak alloc / reserved | trainable params |
|---|---|---|---|---|---|---|---|---|---|
| F 1e-5 | 0.533 / 0.584 / 0.588 | 0.568 | — | 0.582 / 0.481 / 0.584 / 0.654 / 0.392 | 40-60 | 211 s | 611 | 8.7 / 9.5 GB | 333.1 M |
| S 1e-3 | 0.609 / 0.576 / 0.630 | 0.605 | +0.076 / −0.008 / +0.042 | 0.603 / 0.513 / 0.621 / 0.687 / 0.409 | 40-60 | 166 s | 710 | 6.4 / 7.1 GB | 3.58 M |
| V 1e-3 | 0.707 / 0.677 / 0.680 | 0.688 | +0.174 / +0.093 / +0.092 | 0.619 / 0.522 / 0.634 / 0.693 / 0.433 | 40 | 136 s | 882 | 5.8 / 6.4 GB | 3.58 M |

(Reference cache 80.5 MB in every run. Same A6000, GPU 0, no competing load. Total E2 cost:
17 runs, ≈ 50 GPU-minutes.)

Readings (pilot, three data/training replicates — not a significance test):
- **V > F on all three replicates**, overall +0.12 mean, at ~65 % of the wall time and ~67 % of
  the peak memory. V > S on all three (+0.05 to +0.10). S vs F mixed (2 of 3, mean +0.04).
- **Much of V's overall gain is between-depth.** Paired within-depth gains V − F are large
  on replicate 0 (+0.07 to +0.11 at every depth) but small on replicates 1-2 (d2 +0.014 /
  +0.004, d3 +0.041 / +0.021, d4 +0.014 / +0.002, d5+ +0.023 / +0.025, d1 +0.034 / −0.001).
  Overall ρ (0.68-0.71) is above every within-depth ρ, i.e. V orders genotypes across
  mutation depths better; within a depth the improvement is modest and replicate-dependent.
- **Every arm selects a checkpoint at step 40-60** (end of the 40-step warmup or just after);
  validation loss rises afterwards while training loss keeps falling (final train loss
  0.01-0.09). Arms are not "still improving near the endpoint", so per the TODO no 800-step
  extension was run, and overfitting is not to be fixed by longer schedules or more rank.
  Because selection lands at the end of warmup, warmup length / peak LR effectively decide
  the selected model; this is uncontrolled here and is the obvious next optimization variable.
- A shared-LoRA gain over F (mixed here) would say adapting reference representations
  helps; it would not say frozen-reference LoRA is buggy (E1 found no defect).
- Single selection on replicate 0 with a criterion that disagreed with dev within F and S:
  F's selected LR is not its best-on-dev LR (F_3e-5 was 0.586 on rep 0). The F-vs-V gap on
  rep 0 is +0.12 against F's best LR, so the ordering does not hinge on it, but its size does.

**Gate:** reference configuration for E3 = **V 1e-3** (variant LoRA r8, fp32), as recommended
in TODO.md on 2026-11-02; F 1e-5 retained as the historical anchor.

## Not done / next

- bf16-base deployment comparison of V_1e-3 (+H2) on the E4 cases.
- E5 conditional controls (exact backend, WT-feature / one-hot ridge baselines): not started.
- Warmup / peak-LR sensitivity: every run selects at step 20-60, so the schedule effectively
  picks the model; uncontrolled so far.
- Scaling: re-test head (H0/H2) × training scope (V/F) along a label-budget curve before
  treating any pilot choice as settled.
- Keep/reject/investigate: LoRA audit → no defect (keep implementation); restore-cache guard →
  keep (correctness); fp32 adapter arithmetic → reject as default, keep opt-in; variant LoRA vs
  full backbone → investigate (repeatable at 1,280, mostly between-depth); H2 sum_mean_count →
  investigate (E3/E4); H3 wt_relative → reject (control matches it); token-projection capacity →
  investigate later (+0.006 over H2).

## E0 — results reconciled, development protocol frozen

**Summaries refreshed** from completed `results.json` (old CSVs kept in
`experiments/snapshots/pre_2026-11-01/`). The checked-in avGFP summary had 1 replicate per
budget; scratch had all of them (ds320 11 reps, ds1280 7 reps). Inventory with replicate
counts, medians AND means, and protocol fields: `experiments/e0_run_inventory.csv`
(`e0_run_inventory.py`).

Historical context only (minimal head, full backbone fp32, supertest, NOT development
targets): GB1 0.8294 / 0.9176 and avGFP 0.4418 / 0.5606 median at budgets 320 / 1,280
(means 0.8301 / 0.9163 and 0.4264 / 0.5483).

**Reusability.** No existing development result matches the pilot protocol: the GB1
`dms_tune` minimal runs (mF = full backbone fp32, mL = variant LoRA r8 bf16 at lr 3e-3) used
the global `score_rank` target, epoch-clamped schedules (40 epochs ≈ 160 steps at ds1280, 300
at ds320) and per-epoch validation. They stay as context; E4's GB1 baselines are rerun under
the frozen protocol. No avGFP development results existed.

**Manifests** (`build_avgfp_tune_splits.py`, checked by `e0_protocol_checks.py` →
`e0_manifests.csv`, `e0_distributions.csv`):

| task | budgets | reps | train/val (asserted) | dev holdout |
|---|---|---|---|---|
| avGFP `metl_avgfp_dms_tune_ds{320,1280}_r{0,1,2}` | 320, 1,280 | 0, 1, 2 | 256/64, 1,024/256 | 5,000, seed 0, from 42,799 genotypes outside the supertest (5,172) and the union of queue train/val (3,743) — new |
| GB1 `metl_gb1_dms_tune_ds{320,1280}_r{0,1,2}` | 320, 1,280 | 0, 1, 2 | 256/64, 1,024/256 | 5,000, reused (outside supertest and ds80/320/1280 reps 0-2) |

Assertions passing for all 12 manifests: unique genotypes per table; no train/val/dev
overlap; train/val identical to METL's released replicate; one dev set per task, disjoint
from the supertest and from every queue train/val; every mutant of both tables reconstructs
against the WT sequence. Not filtered by outcome.

Distribution notes that matter for reading results: GB1 dev is 4,992 doubles + 8 singles, so
GB1 "depth-stratified" Spearman is the depth-2 number (depth 1 undefined, n < 20). avGFP dev
depth counts 129 / 1,223 / 1,163 / 929 / 1,556 (1/2/3/4/5+); its scores have a hard floor
(q10 = −2.418, heavily tied).

Protocol: checkpoints chosen by released-validation rank loss only (logged next to val
Spearman; criterion not changed after viewing); dev scored at init and once after selection;
replicate IDs change sampled data AND training seed (data/training replicates); no supertest
used anywhere in the pilot.

## E1 — LoRA correctness and numerical diagnostics

**Unit-level audit** (`tests/test_lora_audit.py`, toy ESM-C, fp32; 46 tests): every logical
projection alone and all seven together, for variant / shared / both adapter sets, through
ragged, padded and clamped-recompute (oracle): forward values agree to 1e-4 relative and every
adapter's A and B gradients to 1e-3 relative with nonzero adapters. Nonzero adapters change
every target's output and never the padded (reference) slots. Zero-B first step: dL/dA = 0
exactly, dL/dB ≠ 0; after B moves A gets gradient. Optimizer groups: each trainable parameter
exactly once, requested head / adapter / backbone LRs, frozen base in no group (variant-only,
shared-only, combined, full, full+variant, head-only). Reference cache: valid after optimizer
steps for all four modes; memoized only when it cannot move.

**One latent defect found and guarded:** restoring a checkpoint with `load_state_dict` while
staying in eval mode reused references computed with the replaced weights (shared LoRA, both,
trainable base; variant-only is unaffected because its reference never moves). The harness
was NOT affected — its post-restore evaluation calls `predict()`, whose `eval()` switch
already drops the cache — but any other caller would have scored stale references.
`VariantEffectModel.load_state_dict` now clears the cache; `test_reference_cache_lifetime`
fails without the guard for 3 of 4 modes and passes with it.

**Real-model diagnostic** (`e1_lora_diagnostics.py` →
`e1_lora_diagnostics{.json,_steps.csv,_probes.csv,.log}`): ESM-C 300M, 48 fixed avGFP train
variants (16 singles, 32 at depth 2-9), original minimal head, uniform RankNet on raw score,
r 8 / alpha 16, default targets, AdamW (3e-3 head and adapters, 1e-5 backbone), clip 1,
constant LR, 20 steps.

| config | loss 1 → 20 | ref recomputed / step | frozen base bit-identical | cache vs oracle at step 20: pred max abs diff (scale ~10) / grad rel. err | ms / step |
|---|---|---|---|---|---|
| variant fp32 | 0.693 → 0.091 | 0 | yes | 2e-5 / 2e-5 | 103 |
| shared fp32 | 0.693 → 0.096 | 1 | yes | 8e-6 / 1e-5 | 189 |
| both fp32 | 0.693 → 0.107 | 1 | yes | 2e-5 / 2e-5 | 233 |
| full fp32 (bb 1e-5) | 0.693 → 0.191 | 1 | — | 8e-6 / 1e-5 (backbone) | 203 |
| variant bf16 | 0.693 → 0.088 | 0 | yes | 0.029 / 0.048 | 115 |
| shared bf16 | 0.693 → 0.103 | 1 | yes | 0.022 / 0.026 | 213 |
| both bf16 | 0.693 → 0.123 | 1 | yes | 0.021 / 0.029, 0.028 | 271 |
| variant bf16, fp32 adapter math | 0.693 → 0.086 | 0 | yes | 0.048 / 0.079 | 113 |
| shared bf16, fp32 adapter math | 0.693 → 0.107 | 1 | yes | 0.015 / 0.014 | 210 |
| both bf16, fp32 adapter math | 0.693 → 0.151 | 1 | yes | 0.067 / 0.045, 0.063 | 265 |

Readings:
- fp32 ragged cache = clamped oracle within fp32 numerics at init and after 20 updates, for
  predictions and every group's gradients. No missing-LoRA-path or stream defect.
- Intended parameters update (lora_B from 0; lora_A from step 2; head; backbone in F); the
  frozen base is bit-identical afterwards; eval after training equals a fresh-reference eval
  to ≤ 5e-7.
- bf16 cache-vs-oracle disagreement is 0.2-0.7 % of the prediction scale and 1-8 % in
  gradient (one 21 % reading, both/fp32-math at step 10): two different bf16 computations
  (flex with fp32 softmax vs full-sequence SDPA) — bf16 noise, the reason CLAUDE.md keeps
  oracles fp32.
- **fp32 adapter arithmetic** is bit-identical under an fp32 base (tested) and under bf16 does
  not reduce the oracle discrepancy, the loss trajectory or the gradient error in a consistent
  direction; per-step cost about equal. Not promoted; kept opt-in (`--lora-compute fp32`).
  Peak memory in this diagnostic is dominated by the oracle's full-sequence backward and is
  not a training-cost number.
- At the historical adapter LR 3e-3, after 20 steps the adapter deltas are 10-60 % of the base
  projection outputs on the variant stream (shared `down` ≈ 100 %). Not a defect, but a strong
  perturbation; it is what E2's adapter-LR screen probes.

**Gate: passed.** Intended parameters update, frozen ones do not, reference and gradient
semantics are correct.

## Implementation (TODO §1, §2, §4)

- `--head {mean, mean_count, sum_mean_count, wt_relative, mut_proj}` (needs `--minimal`
  unless `mean`), `--token-proj-dim` (128), `--residue-embed-dim` (16). `mean` is the original
  head, unchanged: same modules, order and init; HEAD-vs-working-tree runs of minimal-FT,
  minimal-LoRA and the full head agree to ≤ 1.2e-7, the size of same-code GPU rerun noise.
  New heads build only their own modules (no prior, assay terms, extra embeddings or
  reference summary); every backend reduces to one final-normed state per real substitution;
  WT states come from `ref.final` at the same token index (gradient kept when the reference
  trains, never forced for a frozen one). No extra mutant-sequence pass. Tests:
  `tests/test_models_heads.py` (30).
- `--lora-compute {activation, fp32}`.
- `build_param_groups` (named groups, each trainable parameter once, guard against trainable
  backbone parameters outside every group) and per-group diagnostics in every eval row:
  pre-clip gradient norm, last-step update norm, distance from init (absolute and relative);
  snapshots on CPU so they do not inflate reported GPU memory.
- `--max-steps` (exact optimizer-step budget, cadence-only evaluation + final step).
- No silent validation→test fallback (`SystemExit`); `--select-on-val` without val rows also
  fails. `--target-transform train_rank` (train-only ECDF ranks for magnitude-aware losses).
  `--eval-label` (e.g. `development`).
- Run records now carry: git diff hash when dirty, split/variants SHA-1, unique-genotype
  counts per partition, eval label, resolved LoRA settings (targets, alpha, arithmetic,
  dtype), optimizer groups, active head parameter count, cold total / prep / train / eval wall
  time, steady train throughput, peak allocated and reserved GPU memory, reference-cache size;
  predictions carry `split` and `seed`, plus `predictions_val.csv` at the selected checkpoint.
- Test suite: 319 passed (240 pre-existing + 79 new).
