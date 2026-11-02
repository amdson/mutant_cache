# Research directions for a scalable mutation model

Literature synthesis: 2026-11-02. Three Luna agents reviewed cross-protein
supervision, few-shot adaptation, and multi-mutant prediction; the parent
reviewed efficient implementations and checked key references. This document
describes possible research programs, not completed experiments. Numerical
goals are proposed decision criteria, not publication thresholds or forecasts.

Related documents:

- [TODO.md](TODO.md): implementation, LoRA, prediction-head, and pilot tasks.
- [PAPER_FINDINGS.md](PAPER_FINDINGS.md): architecture and original paper framing.
- [PAPER_PLAN.md](PAPER_PLAN.md): original accuracy–compute experiment proposal.

## Recommendation

Prioritize a **mutation-supervised initialization that adapts well to unseen
proteins**, followed by **assay conditioning**. Efficient sparse execution is
the enabling infrastructure and a possible standalone methods contribution.

The decisive large-model question is not whether millions of measurements can
be fitted. It is whether supervision from other proteins reduces the data or
compute required to predict mutations in a new protein and experiment.

Start with a modest curated corpus and held-out protein families. Demonstrate
transfer before committing to all available mutation data or a larger backbone.
Initially scope the model to amino-acid substitutions; indels, synonymous
effects, and clinical labels require separate representation/supervision choices.

## Current evidence and its limits

The repository has substantial within-assay development evidence. Completed
minimal sparse full-backbone runs have median test Spearman approximately
0.918 on GB1 and 0.561 on avGFP at the nominal 1,280-label budget. These budgets
include validation labels. The avGFP run copied GB1 hyperparameters without
retuning. Some repository summaries are stale relative to raw scratch results.

These outcomes motivate implementation and training diagnostics. They do not
yet demonstrate cross-protein transfer, a general mutation model, or a matched
end-to-end accuracy–cost advantage. Do not repeatedly use these inspected test
sets to select changes; use the development protocol in TODO.md.

## 1. Efficient mutation adaptation

**Research question:** Can sparse mutation computation deliver useful supervised
adaptation substantially more cheaply than competitive conventional methods?

### Literature context

- [ESME: Efficient inference, training, and fine-tuning of protein language
  models (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/) demonstrates
  substantial benefits from attention kernels, packing, quantization, and
  training optimizations. Those results raise the implementation baseline;
  they are not mutation-cache comparisons.
- [PLMFit (2025)](https://doi.org/10.1093/bib/bbaf381) benchmarks transfer-learning
  strategies and identifies final-layer adaptation as a strong efficiency
  option in its evaluated tasks.
- [Fine-tuning protein language models boosts predictions across diverse
  tasks (2024)](https://doi.org/10.1038/s41467-024-51844-2) shows that efficiency
  and the value of LoRA depend on model size, task, and training implementation.

### First experiment

After the TODO correctness checks, use ESM-C 300M on avGFP and GB1 at nominal
1,280 labels, then expand to 320 and 5,120. Compare:

1. Sparse variant-only LoRA.
2. Sparse shared LoRA.
3. Sparse full-backbone training.
4. Optimized full-sequence LoRA, including a final-layer-only variant.
5. A frozen WT-feature predictor.

Match splits, validation access, hardware, and measurement boundaries. Give
each practical baseline reasonable tuning; use matched configurations as
additional diagnostic comparisons. Report cold total training cost, steady
throughput, peak memory, and total tuning cost.

### Goal and decision

Aim to remain within about 0.02 macro Spearman of competitive conventional
adaptation while reducing end-to-end training cost by severalfold. Establish
an accuracy benefit over cheap features where adaptation is claimed to matter.
For a formal noninferiority claim, use a prespecified margin and paired
uncertainty rather than interpreting a nonsignificant gap as equivalence.

**Risk:** Speedup over a weak or unnecessarily expensive baseline is not a
useful frontier improvement. LoRA projection preferences from other protein
tasks motivate experiments but do not prove the current targets are wrong.

**Role:** Near-term methods contribution and infrastructure for every route below.

## 2. Transferable mutation-trained initialization

**Research question:** Can mutation measurements from other proteins make a new
protein easier to learn?

### Literature context

- [FSFP, Nature Communications 2024](https://www.nature.com/articles/s41467-024-49798-6)
  combines ranking, LoRA, and meta-learning for few-shot fitness prediction.
  Its auxiliary-task selection and target-data use must be matched or disclosed
  in comparisons; do not assume every few-shot protocol is family-held-out.
- [Metalic, ICLR 2025](https://arxiv.org/abs/2410.08355) learns across fitness
  tasks and combines support-set conditioning with fine-tuning. It is a direct
  precedent for learning representations that transfer to new mutation tasks.

Consequently, mutation pretraining followed by fine-tuning is not itself novel.
The opportunity is better transfer, stronger data efficiency, or lower cost
through a scalable mutation representation.

### First experiment

Construct a modest curated cross-protein corpus with protein-family-held-out
training, validation, and evaluation groups. Compare:

1. Original pretrained ESM-C initialization.
2. Ordinary multitask mutation-supervised pretraining.
3. Episodic support/query training using the same supervision.

Keep the sparse architecture and target adaptation procedure fixed. Evaluate
on unseen proteins with 0, 16, 64, and 256 total target labels, including any
labels used for validation. Use fixed query sets across budgets and multiple
support draws. Model selection must use held-out development families, not
the final evaluation families.

Implement ordinary multitask training first; it may capture most of the value
without complicated meta-learning machinery. The zero-label arm must have a
declared shared readout or phenotype-conditioning policy, not an untrained
assay-ID embedding.

### Goal and decision

Seek a repeatable improvement in the label-efficiency curve across held-out
families. An ambitious practical result would be matching original-PLM
256-label performance with 64 labels. Also compare against strong frozen-feature
and support-based methods, not only the unadapted PLM.

**Risk:** Gains may depend on homologous training proteins or similar assays.
Report near-family and distant-family behavior separately. Do not confuse
more measured variants with more independent biological tasks.

**Role:** Recommended main research direction and bridge to a broad mutation model.

## 3. Assay-conditioned general mutation model

**Research question:** Can one representation predict distinct biological
consequences when told what experiment is being performed?

### Literature context

- [ESMRank, Molecular Systems Biology 2026](https://link.springer.com/article/10.1038/s44320-026-00252-3)
  combines overlapping measurements into a consensus ordinal tolerance signal
  and learns it with a boosted-tree ranker using PLM-derived and other features.
  Its signal is stability-enriched. It is not a fine-tuned universal PLM, and
  its success does not establish interchangeable targets across all phenotypes.
- [Supervised learning across large-scale mutagenesis datasets, Protein Science
  2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12967566/) examines cross-protein
  abundance modeling and experiment-specific mappings in VAMP-seq. It motivates
  retaining experimental context even within a relatively coherent phenotype.

Consensus prediction and context-conditioned prediction are different goals.
A mutation can improve stability while reducing activity or changing binding
to one partner. Rank normalization removes neither that contradiction nor
missing information about the experimental endpoint.

### First experiment

With the same training corpus and compute budget, compare:

1. One shared predictor trained with within-assay ranking.
2. A shared representation with transferable phenotype conditioning or heads
   for stability, abundance, binding, and activity.
3. The conditioned model with a small target-assay adaptation stage.

Use metadata with explicit missing/unknown handling rather than arbitrary
assay IDs. Evaluate unseen proteins separately from new assays on proteins
that had other training measurements. Compare single-phenotype training against
adding other phenotype families at matched variant count.

### Goal and decision

Broad training should improve multiple functional categories, and conditioning
should reduce negative transfer. A particularly useful result is a shared model
outperforming independently trained specialists in data-scarce categories.

**Risk:** An overall mean can hide gains confined to stability and losses in
binding or activity. Report category and protein-level results. Assay metadata
may still omit crucial conditions, so a few labeled examples may remain necessary.

**Role:** Closest to the broad-model ambition; pursue after basic transfer works.

## 4. Support-conditioned mutation predictor

**Research question:** Can a model infer the new assay from a few labeled
examples without extensive parameter updates?

### Literature context

- [ProteinNPT, NeurIPS 2023](https://papers.nips.cc/paper/2023/file/6a4d5d85f7a52f062d23d98d544a5578-Paper-Conference.pdf)
  attends over labeled examples and queries for protein-property prediction.
- [Metalic](https://arxiv.org/abs/2410.08355) extends support conditioning toward
  learning across tasks; support conditioning and parameter adaptation can
  complement each other.
- [Kermut, NeurIPS 2024](https://arxiv.org/abs/2407.00002) uses Gaussian-process
  regression with mutation, PLM, and structure-derived similarity. It is a
  strong within-assay comparator, not evidence of cross-assay pretraining.

### First experiment

Freeze one mutation encoder. With identical support examples, compare:

1. Ridge or a small MLP fitted on support examples.
2. A similarity/kernel predictor.
3. A small attention module over support mutation representations and labels.
4. LoRA adaptation.

Use GB1 and avGFP as initial diagnostics. To establish general transfer, train
the support module across assays and evaluate held-out protein families.
Compare random support/query partitions with unfamiliar-position partitions.
Never expose query labels to the support context or adaptation objective.

### Goal and decision

Target competitive accuracy with 16–128 labels and lower adaptation latency or
variance. Demonstrate that learned support conditioning improves over simple
similarity-based prediction at the same label budget.

**Risk:** The method may only implement nearest-neighbor interpolation. Audit
query-batch dependence and specify whether unlabeled queries interact, since
this changes the deployment/evaluation protocol.

**Role:** Attractive extension that provides context for unfamiliar assays.

## 5. Combinatorial mutation specialist

**Research question:** Can sparse mutant computation predict combinations that
strong additive and cheap interaction predictors cannot?

### Literature context

- [Mutate Everything, NeurIPS 2023](https://arxiv.org/abs/2310.12979) already uses
  shared WT representations for efficient higher-order stability prediction.
- [ESM-MSR, bioRxiv preprint 2026](https://www.biorxiv.org/content/10.64898/2026.06.04.730231v1)
  uses WT and mutant-context adaptation paths for stability and interaction
  prediction. Its backbone and structural inputs differ from sequence-only ESM-C.
- [CombinGym](https://github.com/sitonglab/CombinGym) provides lower-order to
  higher-order prediction tasks and prediction/design metrics.
- [FLIP2](https://flip.protein.properties/) supplies additional position,
  mutation-depth, fitness, and WT shifts; simple baselines remain important.
- [Protein stability models fail to capture epistatic interactions,
  Protein Science 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11659742/) shows
  why total double-mutant accuracy must be separated from interaction accuracy.

### First experiment

Use released CombinGym 2-vs-rest splits across multiple proteins. Compare:

1. Additive prediction from independent substitution effects.
2. An additive latent model fitted with a nonlinear observation link.
3. A cached WT-feature interaction head.
4. Sparse adapted mutant computation.
5. Independently trained full-sequence adaptation.

Pin benchmark versions and genotype splits. Evaluate Spearman by mutation
depth and released design metrics. Add a stability-specific interaction
analysis only where matched single and double measurements exist, using
predicted double/single effects to construct predicted interaction residuals.
Measured held-out singles may define evaluation targets but are not deployable
model inputs unless the protocol explicitly supplies them.

### Goal and decision

Aim for repeatable gains over the strongest cheap baseline on higher-order
variants, with better candidate selection and a computational advantage.
A +0.03 macro Spearman gain would be encouraging, not a universal requirement.

**Risks:** Total fitness gains do not prove epistasis prediction. A monotone
transformation of a fixed additive scalar cannot improve its Spearman ordering;
its value lies in fitting the latent model or predicting observed magnitudes.
Assay floors and nonlinear readouts can create apparent interactions.

**Role:** Coherent specialist paper; alternative to, rather than a prerequisite
for, the broad mutation-model route. Existing modest Megascale gains over
additivity still need confirmation.

## 6. Biophysical or structure-informed mutation pretraining

**Research question:** Can physical or structural supervision supply information
that experimental mutation training alone does not learn efficiently?

### Literature context

[METL, Nature Methods 2025](https://www.nature.com/articles/s41592-025-02776-2)
uses Rosetta-derived multitask pretraining followed by experimental adaptation.
It motivates a physical-prior direction distinct from cross-assay experimental
training. Protein-local and global pretraining have different prior-data access.
Use the final 2025 article rather than confusing earlier manuscript versions
with a separate result.

### First experiment

Before generating new simulations, augment the mutation head with available
frozen structural or inverse-folding features. Compare sequence-only and
augmented variants with identical splits, label budgets, and tuning access.

If this helps, run a small auxiliary-pretraining experiment using existing
physical labels, then apply the same few-shot transfer protocol as direction 2.
Avoid starting by reproducing a massive simulation pipeline.

### Goal and decision

Improve low-data transfer or stability-sensitive tasks enough to justify extra
inputs, preparation cost, and deployment constraints.

**Risk:** Target-specific structures or simulations can explain gains and change
the problem setting. Include their cost and information access explicitly.
Do not compare synthetic total-score training with METL's multitask pretraining
as if they were identical objectives.

**Role:** Optional branch if sequence-only transfer plateaus.

## 7. Deferred idea: sequence-based adaptation to the sparse architecture

Added 2026-11-02 after discussion. **Park this idea for now.** The user wants
to prioritize the simpler implementation, LoRA, and prediction-head experiments
in TODO.md before taking on evolutionary-data curation or pretraining.

### Motivation and possible formulations

Use natural protein sequence variation to initialize the sparse computation
before experimental mutation-effect training. Homologs can be represented as
substitution sets relative to a reference, allowing many examples to share its
layer states. An initial source could be versions of E. coli proteins across
strains, grouped by homologous/orthologous protein family; unrelated proteins
from the same proteome must not be aligned together.

The objective is not necessarily reference-conditioned: it depends on what
information is retained in the reference stream and how masking is performed.
Keep both possibilities open:

- **Reference-conditioned reconstruction:** for reference x and homolog y,
  predict y_j from the other residues of y plus x. This gives conditional
  pseudolikelihood terms `log p(y_j | y_without_j, x)`. A fixed clean reference
  is legitimate side information and makes cache reuse and batching especially
  straightforward. Activate the target's substitutions plus the masked query
  position, including queries where target and reference agree.
- **Ordinary sequence reconstruction through sparse execution:** construct
  reference/context states only from the permitted masked target information,
  without an extra intact homolog as side information. This can target
  `log p(y_j | y_without_j)` rather than a reference-conditioned objective.
  Hiding only position j in a separate homolog does not automatically remove
  all extra conditioning information. The actual information available to
  both streams determines the objective.

For reconstruction, mask prediction targets before encoding any context that
would otherwise expose the target sequence's answers indirectly. Clean homolog
context in an explicitly conditional task is different from caching an intact
target and claiming masked prediction. Query metadata, alternative-residue
embeddings, and true mutation-count/change flags must not disclose hidden labels.

Sample individually masked query positions for a stochastic pseudolikelihood
objective. Masking several positions simultaneously is a related multi-mask
reconstruction objective, not the identical single-site objective. Sampling
only changed sites also changes the target distribution; include conserved
sites or explicitly define and weight the alternative objective.

### Why it is deferred

The main unresolved difficulty is **phylogenetic confounding**. Natural
substitutions co-occur because of shared ancestry as well as biological
constraints. Near-duplicate strains may make copying easy and inflate sample
size; naturally observed substitutions are not assay-specific fitness labels.
Reweighting or clustering may help, but neither should be assumed to solve
the problem. This could require substantial effort relative to the first
implementation experiments.

Other costs include orthology/alignment quality, indels incompatible with the
current substitution-only representation, and the tradeoff between sparse
close homologs and informative but more divergent sequences. Start with
confidently aligned substitution-only examples if this direction is revived.

### Small experiment if revisited

Compare no additional pretraining, conventional continued MLM on the same
sequence corpus, and sparse sequence reconstruction. Keep conditional and
ordinary-context formulations separate. Start with variant-only LoRA to
preserve reference reuse; shared/base updates require reference recomputation.

Measure conserved-site and changed-site reconstruction separately against
reference-copying and family-frequency controls. Hold out sequence subclusters
and, separately, whole protein families. The decisive outcome is improved
downstream mutation-effect adaptation under identical label budgets and
training protocols, not reconstruction loss alone.

A simpler related initialization idea is frozen-teacher distillation: match
full-mutant ESM-C representations at substituted positions using the sparse
student, with mutation-induced representation differences as diagnostics.
This avoids interpreting natural sequence occurrence as fitness, but has
teacher-target generation costs and inherits teacher biases. It is also an
optional pretraining experiment, not a prerequisite for the current TODO queue.

Relevant context: [MSA Transformer](https://proceedings.mlr.press/v139/rao21a.html),
[PoET](https://arxiv.org/abs/2306.06156), and
[EVE](https://www.nature.com/articles/s41586-021-04043-8) establish evolutionary
family modeling as a useful approach. The proposed contribution would concern
sparse reference-based execution and transferable initialization, not the
general idea of learning constraints from homologous sequences.

## Shared evaluation requirements

1. **Partition before aggregating labels.** Group duplicate genotypes, repeated
   experiments, overlapping constructs, modified WT backgrounds, and homologous
   domains appropriately. Consensus labels must not incorporate held-out assay
   measurements. Inspect identity and alignment coverage; one clustering
   threshold is not a complete family-leakage audit.
2. **Name the transfer claim.** Random within-assay, unseen position, higher
   mutation depth, new assay on a seen protein, and unseen protein family are
   distinct settings. None substitutes for the others.
3. **Separate development from confirmation.** Count support and validation
   labels; fix query sets across budgets; reserve new tasks/families for final
   confirmation. A temporal holdout also needs a homology audit.
4. **Use meaningful aggregation.** Report per-assay, per-protein, and functional
   category results. Group related landscapes when estimating uncertainty.
   Optimizer seeds and support draws are not independent protein families.
5. **Include cheap competitors.** Frozen WT features, ridge/kernel predictors,
   and relevant additive controls should receive adequate validation tuning.
6. **Measure both accuracy and cost.** Include preprocessing, reference/feature
   extraction, training, validation, and tuning. Compare at equal compute as
   well as adequately trained configurations where feasible.
7. **Distinguish data access.** Sequence pretraining, mutation supervision,
   structures, simulations, and target labels are different information sources.
   Published numbers from incompatible protocols are context, not head-to-heads.

## Test diversity before scale

Run an explicit data-composition experiment alongside direction 2:

- More measured variants from the same training proteins.
- More protein families at the same total variant count.
- More phenotype families at the same total variant count.

Use nested subsets and fixed held-out evaluation families. Balance sampling
so the largest assay does not silently determine the objective. Compare equal
compute and sufficiently optimized runs to separate data value from training
duration. Track whether broader data improve or harm each phenotype category.

The hypothesis is that diversity helps transfer; it is not an established
property of this model. If neither mutation pretraining nor broader training
diversity improves held-out transfer, scaling the corpus is not yet justified.

During evaluation, “all mutation data” means all eligible training measurements
after exclusions. A later all-data release checkpoint cannot inherit a held-out
performance claim if it has trained on those evaluation measurements.

## Recommended sequence and success criteria

| Priority | Direction | Decision it supports |
| --- | --- | --- |
| First | Implementation, LoRA, and head checks in TODO.md | Is the platform reliable and competitive? |
| Main research bet | Transferable mutation-trained initialization | Does mutation supervision create reusable knowledge? |
| Next | Assay-conditioned prediction | Can heterogeneous data help without erasing phenotype differences? |
| Conditional extension | Support conditioning | Can a few measurements identify an unfamiliar assay efficiently? |
| Alternative specialist paper | Higher-order prediction | Do interactions add value beyond strong cheap controls? |
| Optional enrichment | Physical/structural priors | Does extra prior information justify its cost? |
| Deferred idea | Sequence-based sparse pretraining | Can natural variation adapt the architecture without ancestry dominating the learned signal? |

The first general-model pilot should be small: train one shared sparse model
on a curated subset, hold out complete protein families, and compare 16/64/256-
label adaptation against the original ESM-C with an identical adaptation recipe.
Add a frozen-feature comparator and the data-diversity control before expanding.

Promising outcomes are repeated transfer improvements, reduced label needs,
useful gains across functional categories, and better models at fixed compute.
Reasons to narrow or stop are gains confined to homologous proteins, stability-
only improvements presented as universal, negative transfer hidden by averages,
or cheap feature models matching the adapted model at lower cost.

The literature supports this research program. It does not support assuming
that dataset size alone will produce a general mutation model.
