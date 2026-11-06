# Paper direction: cached fine-tuning of protein language models

## Central interpretation

The cache is a **fine-tuning architecture**. Its purpose is to learn useful
mutation-effect predictors efficiently from labeled mutation data. Strong
out-of-the-box performance, or interchangeable execution with an ordinary
full-sequence PLM, is not a requirement.

The central research question should therefore be:

> After task-specific fine-tuning, can the cached architecture achieve
> competitive predictive accuracy with substantially less training compute
> and memory than conventional PLM adaptation?

The most useful evidence is an accuracy–training-cost frontier comparing
separately trained models. A checkpoint switched between cache and exact
execution measures sensitivity to changing its architecture; it is a
diagnostic, not the primary success criterion. The model can legitimately
learn cache-specific representations.

## Architecture as implemented

This section is based on reading the implementation, not executing experiments
or tests. The main sources are
[variant_model.py](mutant_cache/models/variant_model.py),
[variant_cache.py](mutant_cache/cache/variant_cache.py),
[lora.py](mutant_cache/cache/lora.py),
[esmc.py](mutant_cache/backbone/esmc.py), and
[harness.py](mutant_cache/harness.py). Some introductory code comments describe
older interfaces; the explanation here follows the current function bodies.

### Shared computation: a reference stream and mutant streams

The backbone is a local ESM-C implementation: token embeddings, bidirectional
transformer blocks with normalized queries/keys and rotary position encoding,
SwiGLU feed-forward layers, final normalization, and a sequence-logit head.
The supervised predictor reads hidden states at substituted positions; it
does not use the sequence-logit head as its learned scalar prediction head.

For one reference sequence, a full forward records each block's input residual
states, the final states, and lazily reusable keys/values for each layer.
Many measured variants of that reference can then share those states.

Each mutant has its own evolving tokens at its substituted positions. At every
layer, a mutant token can attend to both:

- the reference keys/values at positions that this mutant did not substitute;
- the updated mutant keys/values at its own substituted positions.

The reference key at a substituted position is excluded, so the mutant and
reference versions of that residue are not counted twice. Different mutants
in a batch cannot attend to each other's tokens. Rotary encoding uses the
actual sequence positions, even when the computation contains only a few
tokens.

The reference stream does not receive information back from mutants. This is
the architectural restriction that saves computation: reference positions
remain reference states throughout the depth, while substituted positions
undergo the attention and feed-forward computations of every layer. The
learned model can adapt to that restriction during fine-tuning.

### Two prediction heads

**Full head (default).** Before the variant backbone pass, the pretrained
alternative-residue embedding receives a learned embedding delta containing
absolute position, alternative residue, reference residue, and optionally
assay identity. A learned scalar scales this delta. The added alternative
embedding is separate from the backbone's native amino-acid embedding.

The resulting mutant hidden states feed two branches:

```text
A = sum of learned per-token linear scores
    + optional learned-weight sum of fixed PLM log-likelihood ratios

I = MLP(pool(mutant hidden states), projected reference summary, assay embedding)

prediction = assay_bias + assay_scale * (A + I)
```

Pooling can be mean, sum, or max. The reference summary is either the mean
over reference residues (excluding boundary tokens) or a learned attention
pool, followed by a linear projection. The default interaction MLP has a
128-unit hidden layer and a scalar output; its depth and dropout are
configurable. Assay bias and scale start at zero and one respectively.

Importantly, **A is not necessarily mathematically additive in the full
model**: its token states already depend on co-mutations through attention.
Likewise, the output field called `interaction` is not constrained to equal
experimental epistasis, or to vanish for singles. The branches are a
parameterization, not an identifiable physical decomposition.

The fixed likelihood-ratio prior is optional. The harness computes it from
the original backbone before attaching adapters, using masked-reference
marginals by default or one unmasked reference pass with `--marginal-mode wt`.
It is not updated as adapters train. With the default zero-initialized token
projection and zero-initialized last interaction layer, initial predictions
equal the supplied prior. Without supplied marginals they initially equal
zero. This initialization is an option, not the purpose or success criterion
of the architecture.

**Minimal head (`--minimal`).** Native alternative-residue token embeddings
pass through the same variant backbone, the real substituted-token states
are pooled, and one MLP predicts the scalar. It has no added token embeddings,
likelihood prior, direct reference-summary input, assay embedding, or assay
affine output. The default pool is mean, but sum/max are supported. The MLP
uses ordinary random initialization. Reference context still reaches mutant
tokens through attention. The minimal model can be paired with adapters or
backbone training; it is not inherently a frozen-feature model.

### What can be trained

These choices are independent of the full/minimal head selection. The
head's active parameters train in every case.

| Training version | Reference computation | Mutant computation | Reuse during training |
| --- | --- | --- | --- |
| Frozen backbone, no adapters | Base weights | Base weights plus learned input deltas in the full model | Reference states reusable across steps |
| Variant-only LoRA (`--lora-r`) | Base weights | Base + variant adapters at real substituted positions | Reference states reusable across steps |
| Shared LoRA (`--shared-lora-r`) | Base + shared adapters | Base + shared adapters | Recompute reference with gradients per batch |
| Shared + variant LoRA | Base + shared adapters | Base + shared + variant adapters | Recompute reference with gradients per batch |
| Backbone training (`--train-backbone`), optionally with adapters | Trainable base, plus shared adapters if present | Trainable base, plus requested adapters | Recompute reference with gradients per batch |

LoRA is attached per transformer layer to logical projections `q`, `k`, `v`,
`o`, `gate`, `up`, and `down`; the default target set is `q,o,gate,up,down`.
The shared and variant sets can use different ranks and targets. Each delta
is `(alpha/r) * B*A*x`, with B initially zero. The harness defaults alpha to
twice the rank. Adapter parameters remain fp32 over the optionally bf16 base.

Variant-only LoRA really is position-specific: it never changes the ordinary
reference forward. Shared LoRA changes projections throughout both streams.
Backbone training also changes the reference, but **does not remove the sparse
architecture's restriction on reference-to-mutant feedback**. It is distinct
from choosing full-sequence execution.

Even with frozen backbone weights, the full head's learned input embeddings
are trained through the variant transformer computation. Thus calling that
version “head-only” should not imply that all mutant features are fixed.
For the minimal model without adapters or backbone training, those features
are fixed functions of sequence.

### Execution versions: sparse, padded, and controls

| Execution choice | Behavior |
| --- | --- |
| `cache`, ragged (default) | Flattens only real substitutions across variants; offsets identify each variant. One masked attention over reference and mutant keys implements the permitted connections. |
| `cache`, padded (`--no-ragged`) | Uses a batch-wide union of positions, with a mask for real substitutions. Separately computes reference and mutant attention, then merges them using log-sum-exp normalizers to produce one softmax over the union. |
| `recompute` | Full-length correctness oracle: recomputes blocks but reinstates reference states at non-mutant positions each layer. It realizes the same restricted architecture as the cache. |
| `exact` | Recomputes every position of every mutant without clamping. Non-mutant positions can respond to mutations. Reads substituted-position outputs into the same prediction head. |

Ragged and padded cache paths target the same model semantics; ragged avoids
processing every variant at every position in the batch-wide union. The
CUDA implementation uses compiled FlexAttention; a dense reference attention
implementation is also present. Small dummy padding to kernel block sizes
is an execution detail, not additional biological tokens.

In the padded path, absent substitutions are treated as ordinary reference
positions, clamped to reference states and excluded from mutant pooling.
This prevents predictions from depending on which unrelated variants share
the batch. A WT row has no real substitutions and pools to zero.

The exact path retains added input hooks and applies variant adapters only
at real substituted positions; shared adapters apply everywhere. Therefore
`exact` with variant-only adapters is not conventional LoRA applied at all
positions. Without those hooks/adapters and special masks, it reduces to a
full mutant forward read at substituted positions. Attention ablations in
the exact path restrict mutant query rows; non-mutant rows remain fully
contextual, so interpretation can differ from the sparse path.

Changing execution under fixed weights is optional diagnostic work. For the
paper, independently training and evaluating each architecture is the central
comparison, and a conventional all-position LoRA baseline is a separate arm.

### Ablation versions

| Preset | What the implementation changes |
| --- | --- |
| `additive` | Removes the interaction MLP from prediction and, by default, evaluates every substitution in its own single-mutant context before summing scores. This gives exact additivity relative to the WT prediction. |
| `no_cross_attn` | Blocks mutant queries from reference keys and removes the direct reference-summary head input. Mutant tokens can still attend to one another. |
| `random_ref` | Replaces reference layer/final activations with fixed random tensors per target, optionally matching activation mean/std, and discards memoized reference K/V. |
| `position_only` | Replaces the backbone's alternative residue tokens with mask or reference tokens; removes the added alternative-residue embedding and likelihood prior. Position and mutation presence remain. |

Separate single-mutant execution in the additive ablation matters: masking
mutant–mutant attention alone still lets the absence of other reference keys
signal co-mutation. A pooled nonlinear head would also reintroduce
interactions. The default decomposition avoids both. Programmatically turning
off `additive_decompose` loses that strict guarantee.

The ablation presets do not remove every possible information source:
`no_cross_attn` and `random_ref` leave the fixed PLM marginal prior enabled
unless separately disabled. The former also retains reference-residue input
embeddings. Thus neither, by itself, proves that all pretrained reference
information is irrelevant. Minimal mode rejects named ablation presets.

### Cache lifetime, batching, and objectives

The prediction model uses its own in-memory cache, keyed by target ID and
tokenized sequence length. That assumes a target ID uniquely specifies its
reference sequence. A separate `ReferenceCache` utility supports sequence
hash keys, an LRU, CPU storage, disk persistence, and a sampled fingerprint
of base/shared weights; the main harness does not automatically use this
disk-backed utility.

If shared/base weights train, the model recomputes reference states with
autograd rather than retaining them across training batches. It clears
reference caches on train/eval transitions; fixed evaluation weights allow
reuse within evaluation. An explicitly supplied `batch.ref` bypasses this
normal construction and must be valid for the caller's weights.

Batches contain one target and score set. The model supports pairwise RankNet,
ListNet, ListMLE, a soft-Spearman surrogate, MSE on rank-normalized labels,
and MSE on physical stability labels, including combined objectives. There
are optional pair-gap, sample, and measurement-uncertainty weights. These
loss choices train the same architectures; no objective directly guarantees
that the branch called `interaction` learns physical epistasis.

Assay embeddings and affine terms are indexed by score-set ID. The harness
allocates IDs over train and evaluation score sets, but does not implement an
explicit learned unknown-assay fallback. A previously unseen assay's
embedding receives no training examples. The minimal head removes this
assay-ID issue; the full head needs a deliberate conditioning policy for
cross-protein/cross-assay transfer claims.

Two small implementation details help interpret configurations: the
`per_assay_affine` config field is currently not consulted by the prediction
equation, and the full model constructs some head modules even when a switch
leaves them unused. Module/parameter counts should therefore distinguish
allocated parameters from those active in the selected forward path.

The practical architectural comparison is consequently **head family ×
trainable stream choice × execution path**, with ablations and loss choices
specified alongside it. “The cached model” is not one unique configuration.

## Literature findings

- **[Mutate Everything, NeurIPS 2023](https://arxiv.org/abs/2310.12979):**
  already uses shared wild-type representations to efficiently predict
  multi-mutant stability. This is a key comparator for trained stability
  predictors. The distinction to establish is the value of sparse,
  contextual computation during adaptation, rather than fast multi-mutant
  prediction alone.
- **[Protein stability models fail to capture epistatic interactions,
  Protein Science 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11659742/):**
  additive models are surprisingly competitive. Good double-mutant accuracy
  does not establish that a model predicts interactions. Evaluate additive
  baselines and interaction residuals if making an epistasis claim.
- **[ESM-MSR, 2026 preprint](https://www.biorxiv.org/content/10.64898/2026.06.04.730231v1.full):**
  fine-tunes distinct wild-type and mutant LoRAs for stability and epistasis.
  This makes contextual multi-mutant fine-tuning an established neighboring
  direction. It uses ESM3 and structure inputs, so it is a task comparator,
  rather than a controlled test of our cache architecture.
- **[FLIP2, ICML 2026](https://flip.protein.properties/):** simple models
  often compete with fine-tuned PLMs under realistic distribution shifts.
  One-hot ridge and frozen-feature predictors must be serious baselines.
- **[CombinGym](https://github.com/sitonglab/CombinGym):** supplies released
  higher-order extrapolation tasks and prediction/design metrics. It is a
  natural test of whether efficient fine-tuning remains useful beyond
  single-mutant scans. Pin its version and split files.
- **[METL, Nature Methods 2025](https://www.nature.com/articles/s41592-025-02776-2)
  and [ESMRank, 2026](https://link.springer.com/article/10.1038/s44320-026-00252-3):**
  strengthen the case for separating within-assay adaptation, biophysical
  pretraining, and cross-protein transfer. These are different experimental
  settings, not interchangeable leaderboard comparisons.

## Recommended experiments

### 1. Primary: predictive quality per training budget

Fine-tune the cache architecture and conventional full-sequence alternatives
independently, using the same pretrained backbone where possible. Match data,
validation access, losses where appropriate, and hyperparameter-search budget.
Allow each architecture to learn on its own representations.

Compare both equal training examples and equal GPU-hour budgets. Include
head-only and variant-LoRA cache models, conventional full-sequence LoRA,
one-hot ridge, and frozen-PLM ridge/MLP. Full fine-tuning is useful where
affordable, but should not be the only alternative.

Start with strict Megascale, a specified BRCA1 assay, avGFP, and GB1 as a
dense stress test. Use validation to choose settings, then freeze the test
protocol. Report per-protein accuracy, peak memory, total training time,
reference preparation, and subsequent screening cost. Show the dependence
on sequence length and mutation density.

A proposed practical target is accuracy within 0.02 macro Spearman of a
competitive conventional adaptation baseline at approximately 10× lower
training cost and 4× lower peak memory in a predefined sparse regime.
These are planning thresholds, not publication rules. Establish the accuracy
margin with paired uncertainty; a nonsignificant difference is insufficient.
Beating an expensive baseline is valuable only if the model also competes
with strong cheap alternatives.

### 2. Secondary: higher-order generalization after fine-tuning

Use released CombinGym 2-vs-rest and 3-vs-rest splits. Train on the designated
lower-order variants and test higher-order variants. Compare against additive,
global-epistasis, frozen-feature, and contextual baselines. Report Spearman
by mutation depth and the benchmark's design metric.

An improvement of roughly 0.03 macro Spearman over the strongest cheap
baseline, supported by protein-level uncertainty, would be a useful ambitious
goal. Matching conventional adaptation at much lower cost can also support
the primary paper even without superior accuracy.

### 3. Optional: interaction prediction on unseen proteins

Train on singles and doubles from training proteins; evaluate held-out
protein families using an established Megascale-D partition. Test measured
interaction residuals as well as total double-mutant effects. Keep external
PTMUL-D for confirmation after auditing training homology.

Singles-only training can test extrapolation, but does not by itself identify
interaction terms. Epistasis is an additional empirical claim, not a
prerequisite for a successful efficient fine-tuning architecture.

## What existing results tell us

The repository reports promising speed and memory reductions. Those timings
need a matched end-to-end evaluation for a paper. Existing cache-versus-exact
training comparisons are small and noisy; they do not yet establish either
equivalence or a decisive quality penalty.

Poor performance when cache-trained weights are executed through the exact
backend should not be interpreted as failure of the fine-tuning architecture.
The relevant comparison is each model trained and evaluated through its own
architecture. Out-of-the-box results likewise do not decide the main claim.

ThermoMPNN comparisons in the notes are validation-only. Existing METL
Rosetta comparisons differ in training objective, and the tiny avGFP pilot
is insufficient to establish a few-shot result. Treat these as development
evidence and avoid direct published superiority claims.

## Recommendation

Build the paper around **competitive prediction after fine-tuning at much
lower training cost**. Treat higher-order generalization as a supporting
application and epistasis prediction as an optional stronger result.

Run the focused benchmark panel before MaveDB-wide expansion. The next
deliverable should be a trained-model accuracy–GPU-hour plot with credible
cheap baselines, followed by a mutation-density failure map. Neither
zero-shot quality nor same-weight cache/exact agreement should be a gate.

See [PAPER_PLAN.md](PAPER_PLAN.md) for the longer experimental checklist.
This summary supersedes any reading of that plan that treats the cache
primarily as a drop-in approximation to an already trained full-sequence model.
