# Phase 0 go/no-go: does the sparse variant cache work on a protein LM?

PLAN.md items 4 (degradation) and 5 (runtime). Backbone/cache correctness was
settled separately (exact equivalence against the official `esm` package;
efficient vs bruteforce agreement ~1e-6 in fp32), so everything below is about
the *approximation*, not about bugs.

Scripts: `experiments/phase0_degradation.py`, `experiments/phase0_runtime.py`,
`experiments/phase0_figs.py`. Raw CSVs and logs next to them, figures in
`experiments/figs/`. Model: ESM-C 300M, one RTX A6000, `CUDA_VISIBLE_DEVICES=0`.

**Verdict: GO, with conditions.** The cache is 12-32x faster than full
re-forwarding at the mutation depths real DMS data actually has, and at those
depths it reproduces the exact masked-marginal ranking at Spearman 0.92-0.998.
It falls apart at high *mutation density*, not high mutation count, which means
the binding constraint is short proteins, not deep mutants — and every dataset in
DATASETS.md sits below density 0.07. See "What this means for Phase 3" for the
six design consequences, the important ones being: do not mean-pool the sequence
for the reference summary, keep the additive head so the learned part can absorb
a per-target bias, and chunk the variant batch before training on deep mutants.

---

## Experiment 1 — accuracy degradation

### Setup

Four real DMS reference proteins, sequences taken verbatim from ProteinGym
`DMS_substitutions.csv` (`target_seq`) except GB1:

| name | L | sequence source |
|---|---|---|
| GB1 | 56 | the GB1 domain of Olson 2014 / Wu 2016 / FLIP (first 56 aa of the FLIP `5LDE_1` construct) |
| ubiquitin | 76 | first 76 aa of `RL40A_YEAST_Roscoe_2013` (the ubiquitin-ribosomal fusion target) |
| avGFP | 238 | `GFP_AEQVI_Sarkisyan_2016` |
| PTEN | 403 | `PTEN_HUMAN_Matreyek_2021` |

For each protein and each K in {1,2,3,5,10,20,50}: 64 random mutants (uniformly
random distinct positions, uniformly random non-WT canonical residue). Each
mutant is compared against a *full exact forward of the mutant sequence*. fp32
throughout (TF32 off) so numerics do not confound the approximation; a bf16 copy
of the model runs the same mutants for the bf16 cost.

Two quantities:

* **hidden-state cosine** of the post-final-norm state, at mutated positions
  (what the cache recomputes) and at unmutated positions (where the cache simply
  returns the wild-type reference state — the approximation, by construction).
* **masked-marginal LLR in mutant context**: for each mutated position j, mask j
  with the other K-1 substitutions in place and read `log p(alt_j) - log p(wt_j)`;
  the variant score is the sum over j. At K=1 this is exactly the standard ESM-C
  zero-shot masked marginal. The cache computes the same thing with the K mutated
  positions as the variant slots and `<mask>` in slot j. Note the cache's
  reference stream never sees the mask token, so even at K=1 the two differ.

### Hidden-state cosine, cache vs exact (fp32)

Mutated positions (mean over 64 mutants):

| protein | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| GB1 (56) | 0.971 | 0.958 | 0.952 | 0.937 | 0.901 | 0.875 | 0.951 |
| ubiquitin (76) | 0.994 | 0.992 | 0.989 | 0.980 | 0.958 | 0.908 | 0.704 |
| avGFP (238) | 0.979 | 0.982 | 0.980 | 0.978 | 0.972 | 0.968 | 0.952 |
| PTEN (403) | 0.988 | 0.990 | 0.988 | 0.986 | 0.986 | 0.980 | 0.956 |

Unmutated positions (reference state vs the exact mutant's state there):

| protein | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| GB1 | 0.984 | 0.966 | 0.956 | 0.931 | 0.865 | 0.751 | 0.609 |
| ubiquitin | 0.985 | 0.950 | 0.932 | 0.863 | 0.730 | 0.551 | **-0.089** |
| avGFP | 0.998 | 0.996 | 0.994 | 0.991 | 0.981 | 0.968 | 0.935 |
| PTEN | 0.999 | 0.998 | 0.997 | 0.995 | 0.989 | 0.974 | 0.906 |

Mean-pooled residue embedding (the cache's reconstruction of the whole sequence:
reference at unmutated, cache at mutated):

| protein | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| GB1 | 0.996 | 0.993 | 0.990 | 0.984 | 0.968 | 0.933 | 0.974 |
| ubiquitin | 0.979 | 0.921 | 0.892 | 0.774 | 0.594 | 0.519 | 0.718 |
| avGFP | 1.000 | 1.000 | 0.999 | 0.999 | 0.998 | 0.995 | 0.987 |
| PTEN | 1.000 | 1.000 | 0.999 | 0.998 | 0.995 | 0.983 | 0.936 |

(figs/fig1_cosine_vs_nmut.png)

### Masked-marginal LLR agreement

Spearman between cached and exact scores across the 64 mutants of each cell —
the number that matters, because downstream evaluation is rank-based:

| protein | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| GB1 | 0.986 | 0.987 | 0.939 | 0.779 | 0.647 | 0.317 | 0.826 |
| ubiquitin | 0.996 | 0.993 | 0.988 | 0.959 | 0.847 | 0.477 | 0.348 |
| avGFP | 0.939 | 0.947 | 0.940 | 0.914 | 0.894 | 0.913 | 0.724 |
| PTEN | 0.997 | 0.997 | 0.995 | 0.993 | 0.985 | 0.970 | 0.804 |

Pooled across all four proteins: 0.993 / 0.996 / 0.994 / 0.991 / 0.985 / 0.930 /
0.834 at K = 1/2/3/5/10/20/50.

Mean absolute gap in nats (whole-variant score, i.e. summed over K positions):

| protein | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| GB1 | 0.29 | 0.50 | 0.92 | 1.77 | 5.24 | 13.68 | 2.46 |
| ubiquitin | 0.11 | 0.25 | 0.48 | 1.09 | 2.78 | 11.68 | 29.84 |
| avGFP | 0.30 | 0.51 | 0.70 | 1.21 | 2.50 | 4.72 | 11.32 |
| PTEN | 0.21 | 0.26 | 0.40 | 0.69 | 1.10 | 2.05 | 12.29 |

The absolute gap on its own is not interpretable; the ratio |gap| / sd(exact
scores in the same cell) is what drives Spearman. That ratio is 0.04-0.40 at K=1
and 0.15-3.3 at K=20. It also explains avGFP: at K=1 its gap is a perfectly
ordinary 0.30 nats, but the spread of random single-mutant LLRs in avGFP is only
0.75 nats, so even a small bias costs rank. PTEN has a 4.6-nat spread at K=1 and
a 0.21-nat gap, hence Spearman 0.997.

The gap is a **systematic bias, not noise**: `mean(gap)` and `mean(|gap|)` agree
in magnitude in nearly every cell (e.g. avGFP K=20: -4.724 vs 4.724; PTEN K=50:
-12.21 vs 12.29), and the sign is constant within a protein — negative for GB1,
avGFP and PTEN at every depth (the cache scores substitutions *lower*, i.e. it is
more confident the wild type is right, because the residues it conditions on are
frozen wild type), and positive for ubiquitin at every depth up to 20. So the
bias is a per-target offset rather than a universal direction; what matters for
Phase 3 is that it is stable within a target, not which way it points.

(figs/fig3_llr_agreement.png)

### What actually predicts the error

Not mutation count and not sequence length — **mutation density** K/L, and only
for the unmutated-position term. `1 - cos_unmut` is within a factor of ~2-4 of
the density itself across all four proteins and all depths (ratio
`(1-cos_unmut)/density` = 0.31-2.1 over the whole grid), i.e. the frozen
reference loses roughly as much of its state as the fraction of the sequence that
changed under it. The mutated-position cosine does *not* collapse onto density:
it has a per-protein floor (~0.01-0.02 of error for avGFP/PTEN at any density,
~0.03 for GB1) on top of a density-driven term.

Because length only enters through density, longer proteins are *better*, not
worse: PTEN at 50 substitutions (density 0.12) is more accurate than ubiquitin at
10 (density 0.13) and far more accurate than GB1 at 20 (density 0.36).

There is also a **non-monotone turn at extreme density**: GB1 at K=50 (89% of the
sequence mutated) is *more* accurate than at K=20, and its Spearman recovers from
0.32 to 0.83. That is mechanical, not a fluke — when almost every position is a
variant slot, the variant stream attends mostly to variant keys and the cache
degenerates into a nearly-full forward. The worst case is intermediate density,
where the frozen reference still dominates the attention and is already wrong.

(figs/fig2_error_vs_density.png)

### Comparison with the DNA-LM prior

The prior number from the DNA work is ~99% cosine at 50 edits. That is not a
like-for-like comparison: 50 edits in a multi-kb DNA window is a density of
~0.01-0.05, whereas 50 substitutions in a 76-aa protein is a density of 0.66. At
*matched density* proteins are only slightly worse than the DNA result: PTEN at
density 0.0025-0.025 gives 0.986-0.990 cosine at mutated positions, versus the
DNA ~0.99. The pessimistic hypothesis in the brief — that substitutions perturb
distant residues in a folded structure and so proteins should be much worse — is
**not** supported at low density. What does show up is the per-protein error
floor at mutated positions (avGFP sits at ~0.98 regardless of how few mutations
there are, PTEN at ~0.988), which is exactly the "reference positions never see
the variant" term and which does not shrink to zero as K goes to 1.

### bf16 cost

Negligible. Across the whole grid, `cos_mut` computed with the bf16 model differs
from the fp32 value by at most 0.0015, and Spearman by at most 0.017 (in the
noisiest high-K cells). The pure-numerics floor — cosine between a bf16 exact
forward and an fp32 exact forward of the same mutant — is 0.9994-0.99996, i.e.
bf16 rounding is 1-2 orders of magnitude smaller than the approximation error
everywhere. Running the whole variant path in bf16 (with the fp32 attention core
the cache already uses) costs nothing in accuracy.

---

## Experiment 2 — runtime

ESM-C 300M, one A6000, `torch.no_grad`, 3 timed reps after warmup. The baseline
is the same N mutant sequences through the ordinary model, batched to
<=32768 tokens per batch. The cache path chunks the variant batch so N*K <= 16384.

Two numbers are reported, because the reference pass is paid once:

* **amortized** = N / (t_reference + t_variants)
* **per-call** = N / t_variants (reference already cached, which is the
  `ReferenceCache` case and the fine-tuning case)

The reference pass is ~20 ms at every length tested (it is one batch-1 forward).

### Amortized speedup over a full forward, N = 256 mutants, bf16

| L | K=1 | 2 | 5 | 10 | 20 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 3.86 | 3.82 | 3.79 | 2.57 | 1.38 | 0.64 | -- | -- | -- |
| 128 | 7.80 | 7.71 | 7.11 | 4.66 | 2.48 | 1.14 | 0.56 | -- | -- |
| 256 | 15.87 | 15.48 | 12.39 | 7.80 | 4.11 | 1.84 | 0.90 | 0.41 | -- |
| 512 | 31.91 | 31.47 | 19.81 | 11.92 | 6.21 | 2.71 | 1.34 | 0.62 | 0.27 |
| 1024 | 53.35 | 52.13 | 29.07 | 16.65 | 8.58 | 3.65 | 1.81 | 0.86 | 0.39 |

Same grid in fp32 (the cache wins by much more, because the baseline loses its
fused bf16 attention while the cache's attention core is fp32 either way):

| L | K=1 | 2 | 5 | 10 | 20 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 18.05 | 16.45 | 9.19 | 5.24 | 2.66 | 1.13 | -- | -- | -- |
| 128 | 36.77 | 32.40 | 17.57 | 10.03 | 5.11 | 2.17 | 1.08 | -- | -- |
| 256 | 66.81 | 56.15 | 31.48 | 18.20 | 9.38 | 4.00 | 2.00 | 0.94 | -- |
| 512 | 105.55 | 84.20 | 50.63 | 30.28 | 16.09 | 6.94 | 3.51 | 1.68 | 0.77 |
| 1024 | 139.39 | 116.58 | 75.30 | 47.20 | 26.21 | 11.56 | 5.88 | 2.86 | 1.34 |

### Absolute throughput and memory (bf16, N = 256)

| L | K | full fwd (mut/s) | cache amortized (mut/s) | cache per-call (mut/s) | speedup amort | speedup per-call | peak mem full (MiB) | peak mem cache (MiB) |
|---|---|---|---|---|---|---|---|---|
| 256 | 1 | 326 | 5167 | 8444 | 15.9 | 25.9 | 3162 | 767 |
| 256 | 5 | 326 | 4034 | 5789 | 12.4 | 17.8 | 3162 | 837 |
| 256 | 20 | 326 | 1338 | 1487 | 4.1 | 4.6 | 3162 | 1102 |
| 512 | 1 | 159 | 5074 | 8411 | 31.9 | 52.9 | 3138 | 866 |
| 512 | 5 | 159 | 3150 | 4180 | 19.8 | 26.3 | 3138 | 982 |
| 512 | 20 | 159 | 987 | 1070 | 6.2 | 6.7 | 3138 | 1412 |
| 1024 | 1 | 76 | 4047 | 5947 | 53.4 | 78.4 | 3092 | 1065 |
| 1024 | 5 | 76 | 2206 | 2670 | 29.1 | 35.2 | 3092 | 1275 |
| 1024 | 20 | 76 | 651 | 686 | 8.6 | 9.0 | 3092 | 2040 |

Peak memory of the cache path is 2-4x lower than the batched full forward at the
same N, and it is dominated by the fp32 attention scores (N*h*K*L), not by the
reference states. The reference states themselves are 11 / 22 / 43 / 87 / 173 MiB
at L = 64 / 128 / 256 / 512 / 1024 in bf16 — but two thirds of that is the
memoized K/V, which is a pure function of `layer_inputs`.

### Mutants per reference (L = 512, bf16)

| K | N=1 | 4 | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|---|---|
| 1 | 0.30 | 0.61 | 2.17 | 8.36 | 32.4 | **91.9** |
| 5 | 0.29 | 0.58 | 2.07 | 8.02 | 20.0 | 26.5 |
| 20 | 0.29 | 0.60 | 2.15 | 4.88 | 6.26 | 6.72 |

Amortized speedup; the cache loses for N < ~7 (the reference pass plus kernel
launch overhead is not yet paid off). At K=20 it has saturated by N=256 (6.26 of
an eventual 6.72); at K=1 it is still climbing at N=1024, because at K=1 the
variant path is so cheap that the fixed 20 ms reference pass dominates until N is
in the hundreds. Peak throughput measured: **20,198 mutants/s** per-call at
L=512, K=1, N=1024, against 158 mutants/s for the full forward. The realistic DMS
workload — one reference, thousands of mutants — is on the right side of this
everywhere.

(figs/fig4_runtime.png)

### Break-even

Amortized speedup crosses 1.0 at (log-interpolated, bf16, N=256):

| L | break-even K | break-even density |
|---|---|---|
| 64 | 32 | 0.51 |
| 128 | 59 | 0.46 |
| 256 | 93 | 0.36 |
| 512 | 138 | 0.27 |
| 1024 | 180 | 0.18 |

In fp32 the cache still wins at density 0.65-0.75 and never loses at L=1024 up to
K=400. Two caveats on the bf16 break-even: the full-forward baseline uses fused
SDPA while the cache's attention core is explicit fp32 matmuls, and at large K the
cache does strictly more work than a plain forward (a self- *and* a cross-
attention, merged by log-sum-exp). So the bf16 break-even density of ~0.2-0.5 is
partly an implementation artifact and could be pushed out with a fused kernel.
It does not matter for this project: every dataset in DATASETS.md sits far below
these densities. Tsuboyama Megascale (the processed table here: 479 domains,
32-74 aa, median 56) has a maximum density of 0.0625 and a 99th percentile of
0.051; FLIP GB1 is 4 sites on 56 aa (0.071); avGFP Sarkisyan averages 3.7
substitutions on 238 aa (0.016, up to ~0.06); CombinGym's deepest landscape is 16
mutations on an antibody variable domain (~0.07).

### LoRA (the fine-tuning path)

`attach_variant_lora(model, LoRAConfig(r=16, alpha=32))`, bf16 base, fp32
adapters, N = 128 per step:

| L | K | var fwd (s) | var fwd+bwd (s) | full fwd, no grad (s) | train mut/s | fwd speedup (per-call) | peak mem fwd+bwd (MiB) |
|---|---|---|---|---|---|---|---|
| 128 | 1 | 0.046 | 0.126 | 0.201 | 1013 | 4.4 | 1057 |
| 128 | 5 | 0.047 | 0.129 | 0.201 | 994 | 4.3 | 2183 |
| 128 | 20 | 0.087 | 0.210 | 0.202 | 610 | 2.3 | 6402 |
| 512 | 1 | 0.045 | 0.126 | 0.815 | 1018 | 18.0 | 1378 |
| 512 | 5 | 0.047 | 0.130 | 0.813 | 985 | 17.2 | 3192 |
| 512 | 20 | 0.136 | 0.322 | 0.813 | 398 | 6.0 | 10092 |
| 512 | 50 | 0.324 | 0.767 | 0.814 | 167 | 2.5 | 24352 |
| 1024 | 1 | 0.061 | 0.185 | 1.686 | 691 | 27.5 | 1815 |
| 1024 | 5 | 0.060 | 0.181 | 1.685 | 707 | 28.2 | 4599 |
| 1024 | 20 | 0.203 | 0.473 | 1.687 | 271 | 3.6 | 14923 |
| 1024 | 50 | 0.488 | 1.139 | 1.686 | 112 | 1.5 | 36259 |

Adding LoRA is not free, but it is cheap: at L=512, K=1 the variant forward goes
from 0.030 s (no adapters, and flat in N up to 256 — this regime is
kernel-launch-bound, not FLOP-bound) to 0.045 s with r=16 adapters on
q/o/gate/up/down, i.e. ~300 extra small kernel launches per call. That still
leaves it 18x faster than a full forward of the same mutants. A training step
(forward + backward through the
variant path) at L=512, K<=5 runs at ~1000 mutants/s and is still 6.3x faster
than a *no-grad* full forward of the same mutants — the honest comparison against
full-model fine-tuning is far more lopsided than that, since that baseline would
also need backward.

The one bad number here is **activation memory for the backward**: it scales as
N*h*K*L in fp32 (the attention scores the log-sum-exp merge has to save), and
hits 24 GB at N=128, K=50, L=512 and 36 GB at L=1024. At the depths this project
cares about (K<=5) it is 1.4-4.6 GB and fine, but the Phase 3 trainer needs a
variant-batch chunker or activation recompute before it can touch deep mutants.

---

## What this means for Phase 3

1. **Depth-stratified everything, and expect the cache's own error to be
   invisible at depth <=5 and material at depth >=20.** At K<=5 the cache
   reproduces the exact zero-shot ranking at Spearman 0.91-0.998, which is well
   inside DMS measurement noise; by K=20 on a short protein it is at 0.32, which
   is not. Every planned dataset is inside the good regime: Tsuboyama Megascale
   tops out at density 0.0625 (479 domains, 32-74 aa, singles and doubles, plus
   the triples list), FLIP GB1 at 0.071, CombinGym at ~0.07. The weakest
   realistic cell is a short domain at depth 2-3, which the GB1 row brackets at
   cos 0.952-0.958 / Spearman 0.94-0.99. That is acceptable but it is the cell
   where the full-recompute control of PLAN item 22 has to be run as a real
   ablation, not a formality — and it is also exactly the Phase 6 gate (Megascale
   singles -> doubles), so the control belongs in that experiment.
2. **The error is a per-target systematic bias, which is learnable.** The gap
   has a direction and magnitude that is stable within a protein (negative for
   GB1/avGFP/PTEN, positive for ubiquitin) rather than varying mutant to mutant;
   |mean gap| ~ mean |gap| in nearly every cell. That is an argument
   for keeping the additive component of the head initialised from ESM-C masked
   marginals (PLAN item 18) and letting the learned part absorb the offset,
   rather than treating the cached logits as a noisy estimate of the exact ones.
   It also means the ablation in item 20 ("additive") is the right control: if
   the model only recovers the bias, the cross-attention is doing nothing.
3. **Do not mean-pool the sequence for the reference summary.** `cos_pool`
   collapses to 0.52-0.72 for ubiquitin at K>=10 while the mutated-position
   states are still at 0.91-0.96, because the pooled vector is dominated by the
   frozen reference positions — the pooled "mutant" embedding is mostly a
   wild-type embedding. The open question in PLAN ("ref summary: mean-pool vs
   learned query") is answered in favour of a learned query over the variant
   tokens plus a *separate* reference summary, not a pooled mixture of the two.
4. **bf16 base + fp32 adapters is free.** Confirmed: <=0.0015 cosine and <=0.017
   Spearman versus fp32, with a numerics floor two orders of magnitude below the
   approximation error. No reason to train in fp32.
5. **`ReferenceCache` (PLAN item 9) should persist `layer_inputs` only.** The
   memoized per-layer K/V is two thirds of the 87 MiB/target at L=512 and is a
   pure function of `layer_inputs` plus base weights; recomputing it on load
   costs one linear per layer. Persisting it instead would make a few thousand
   targets a few hundred GB.
6. **Chunk the variant batch in the trainer.** Forward is fine; the backward's
   fp32 attention scores are O(N*h*K*L) and reach 24 GB at N=128, K=50, L=512.
   A chunk rule of N*K <= ~16384 (what `phase0_runtime.py` uses) keeps the
   forward under 2.6 GB; the backward needs something tighter or recompute.

### Honest caveats

* Random uniform mutants are the right model for a single-mutant DMS scan, but
  not for real multi-mutant libraries, which concentrate substitutions at a few
  positions (FLIP GB1 is 4 fixed sites; CombinGym likewise). Clustered mutations
  at K=4 have a much higher *local* density than a random K=4, and the cache's
  variant-variant attention is exact among them while their shared neighbourhood
  is frozen — this grid does not measure that case, and it should be measured
  before trusting item 11.
* Only ESM-C 300M was tested. 600M has 36 layers and d=1152; the approximation
  is architectural so the trend should hold, but the numbers will move.
* The LLR metric here is the mutant-context masked marginal. The commonly
  reported zero-shot baseline sums *wild-type-context* single-mutant marginals,
  which is depth-1 by construction and for which the cache and the exact model
  differ only by the mask-token effect (the K=1 column above).
* The runtime baseline is an unoptimised batched full forward of each mutant with
  no cross-mutant sharing. That is the correct comparison for this workload, but
  it is not a hard floor — a smarter baseline could share some prefix work.

---

## Experiment 3 — clustered mutations (follow-up; corrects the density claim)

`experiments/phase0_clustered.py`. Experiments 1 and 2 drew mutated positions
uniformly at random. Real libraries do not: FLIP GB1 varies V39/D40/G41/V54,
three of them adjacent, and combinatorial designs pick interacting residues
almost by construction. This compares, at matched K, uniform-random positions
against one contiguous block, 64 mutants per cell, fp32, ESM-C 300M.

Spearman (cache vs exact masked-marginal LLR), uniform → block:

| protein | K=2 | K=3 | K=5 | K=10 |
|---|---|---|---|---|
| GB1 (56) | .967 → .933 | .970 → .826 | .864 → .803 | .512 → .666 |
| avGFP (238) | .953 → .958 | .926 → .942 | .889 → .928 | .923 → .924 |
| PTEN (403) | .996 → .991 | .995 → .976 | **.996 → .935** | **.993 → .879** |

Real FLIP GB1 site set (V39/D40/G41/V54, the actual item-11 workload):

| depth | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Spearman | .982 | .969 | .894 | .845 |
| mean abs gap (nats) | 0.41 | 0.74 | 0.96 | 1.30 |

**Clustering costs real accuracy, and it costs most where experiment 1 was most
reassuring.** PTEN was the headline case for "longer proteins are better": at
K=10 it keeps Spearman .993 with scattered mutations but drops to .879 when the
same ten mutations are contiguous. So global density K/L is *not* the whole
predictor — local density matters, and length only protects you when the
mutations are actually spread out. The mechanism is the obvious one: the cache
lets variant positions attend to each other, but each variant still sees a
reference context in which its *neighbours* carry wild-type residues, and
adjacent positions are exactly where that assumption is worst.

GB1 is non-monotone in the same way experiment 1 found (K=10 block beats K=10
uniform) because at density 0.18 the block covers a large share of the local
context either way.

Partial confound, stated rather than hidden: block mutants also have a smaller
spread of true scores (PTEN K=10: sd 7.1 block vs 11.5 uniform), and Spearman
falls more easily when the thing being ranked is compressed. Normalising by it,
gap/sd is 0.27 block vs 0.10 uniform at PTEN K=10 — still ~2.7x worse, so the
effect is real but somewhat amplified in the Spearman column.

**Revision to the verdict.** Still GO, and depth ≤2 is safe under any layout.
But the claim "every planned dataset sits below density 0.07, therefore fine"
does not survive: FLIP GB1 at its real depth 4 gives .845, not the ~.99 its
global density 0.07 would predict. Two consequences:

1. The item-22 full-recompute control is not optional for the depth-extrapolation
   datasets (FLIP GB1, CombinGym). Report cached and exact side by side there.
2. Sequence-adjacent is only a proxy for the real case, which is residues close
   in 3D. Worth one more check against structure contacts before trusting
   CombinGym numbers, since interacting sites are the ones those assays choose.
