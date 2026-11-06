# TODO: measure and reduce pretrained-weight / sparse-architecture mismatch

Created 2026-11-02. This is a staged plan, not evidence that the mismatch is
large or that distillation will improve mutation prediction. Begin with the
measurement experiment; only run pretraining if the diagnostic finds a material,
systematic mismatch.

## Motivation and proposed scheme

The sparse model initializes from pretrained ESM-C weights, but changes the
computation at inference: only mutated positions evolve, and unchanged
reference positions do not receive feedback from the mutations. Full-sequence
ESM-C lets a mutation alter contextual states throughout the protein. Therefore
the pretrained weights may be poorly calibrated for the sparse computation,
even when the sparse forward is implemented correctly.

Candidate scheme: use a frozen, full-sequence ESM-C teacher and the current
sparse ESM-C student on the same WT/mutant examples. Train only the student's
variant LoRA (initially) to match teacher representations at substituted
positions. Then fine-tune on measured mutation effects and compare against the
same model without distillation. This adapts the sparse computation to a
teacher signal without changing the sparse architecture.

Do not conflate two questions:

1. **How different are the pretrained representations under the two forwards?**
2. **Does reducing that difference improve supervised prediction or cost?**

The first is answered by the initial experiment below. A representation gap by
itself is not a defect: sparse and full execution intentionally compute
different functions, and task-specific training may make the sparse model
effective without imitating the teacher.

## Experiment 0: quantify the mismatch before training

### Data and controls

- [ ] Use the original pretrained ESM-C 300M checkpoint in eval mode, with no
  task-trained adapters and the same tokenizer, sequence, residue mapping,
  boundary-token convention, and final normalization in both paths.
- [ ] Select a small, reproducible sample of Megascale WT backgrounds, including
  varied lengths and both natural and designed domains. Include a fixed,
  stratified sample of measured singles and doubles per protein. Save sequence,
  variant, and source-row IDs; hash the manifest. Do not select by measured
  outcomes. A practical first pass is 20–50 proteins and up to 100 variants per
  protein, capped to fit available memory.
- [ ] Add a tiny synthetic validation panel with substitutions at one site,
  separated sites, adjacent sites, and several mutation depths. This tests
  mutation parsing and tensor alignment independently of the data loader.
- [ ] Run teacher full-mutant forward on each complete mutant sequence. Run the
  student using the sparse cache with the same pretrained weights. Verify that
  sparse outputs match the clamped `recompute` oracle to fp32 tolerance before
  comparing either with the teacher. The exact execution path is the full-mutant
  computation; `recompute` is only the sparse-model correctness oracle.
- [ ] Add reference-state checks: WT sequence full forward and cached reference
  outputs must agree. At positions not mutated, the sparse stream intentionally
  uses reference states; compare these with teacher mutant states as a separate
  non-mutant-context diagnostic, not as a student-token error.
- [ ] Keep calculations in fp32 initially. If the final training recipe uses
  bf16, repeat a small subset in that precision to distinguish numerical noise
  from architecture differences.

### Measurements

For each variant and mutated position, compare teacher and sparse student hidden
states at every transformer layer (block outputs and final normalized output):

- [ ] cosine similarity and `1 - cosine`;
- [ ] relative L2 error, normalized by teacher-state norm, plus absolute L2;
- [ ] mean and per-layer distribution, with protein-level summaries;
- [ ] prediction-relevant pooled/token features if applicable;
- [ ] paired mismatch by mutation depth, sequence length, mutation location,
  and whether mutated positions are close in sequence or structure when structure
  data are available.
- [ ] Compare a position's representation under WT, sparse mutant, and full
  mutant forwards. Quantify teacher mutation-induced delta
  `(h_teacher(mut) - h_teacher(WT))` against student delta
  `(h_sparse(mut) - h_sparse(WT))`, in addition to absolute-state matching.
- [ ] Quantify whether mismatch is dominated by baseline representation offset,
  the mutation-induced change, or interaction-dependent change. For doubles,
  compare the teacher's double-mutant delta to the sum of its teacher single
  deltas, and the same for sparse states, only where matched examples exist.
- [ ] Summarize fraction of examples above preregistered error levels and paired
  protein-bootstrap intervals. Avoid treating residues from one protein as
  independent samples.
- [ ] Measure teacher exact-forward and student sparse-forward runtime, peak
  memory, and cost to produce/cache targets. This informs whether distillation
  is practical; it is not itself an accuracy result.

### Decide whether to proceed

- [ ] Inspect per-layer/per-protein results before choosing a target layer or
  loss. Do not declare mismatch "large" from raw L2, which depends on scale.
- [ ] Proceed if mismatch is reproducible across proteins and affects the
  mutation-induced deltas or deeper layers in a way that could plausibly impair
  the supervised task. If the differences are small, early-layer-only, or
  unrelated to downstream errors, defer distillation and continue direct
  supervised training.
- [ ] Freeze the diagnostic report and target construction before using the
  held-out task test set. The diagnostic may use training/validation proteins;
  test proteins are reserved for the final comparison.

## Experiment 1: frozen full-mutant teacher distillation

### Target design

- [ ] Teacher is the original pretrained ESM-C, frozen. Generate its hidden
  states for observed mutant sequences and WT sequences. Cache targets with
  model checksum, data-manifest hash, dtype, layer selection, and token mapping.
- [ ] Student is the sparse architecture, initialized from the identical base
  checkpoint. Begin with variant-only LoRA rank 8/alpha 16 on q/o/gate/up/down
  and no task-trained head unless the objective explicitly includes it. Keep
  the reference stream frozen/reusable for the first pass.
- [ ] Primary target: hidden states at true substituted positions, aligned to
  the same mutant residues and teacher positions. Compare final normalized
  representations first. Add intermediate-layer targets only if Experiment 0
  shows a meaningful depth profile and the single-target run is stable.
- [ ] Use cosine loss as the primary scale-insensitive matching objective, with
  a small relative-L2 diagnostic or secondary loss. Set target normalization
  and any layer weights using train/validation only. Report loss per mutation
  and per protein; long variants must not dominate merely by having more tokens.
- [ ] Include both singles and doubles in the distillation pool if available.
  Sample proteins and mutation depths deliberately; record data exposure. Do
  not construct synthetic combinations unless a separate augmentation ablation
  is planned.
- [ ] Avoid trivial objectives: do not primarily match unchanged positions,
  which the sparse model copies by construction. Monitor the mutation-induced
  delta against the WT representation so a low loss cannot come solely from
  matching generic pretrained states.

### Controlled arms

| Arm | Procedure |
| --- | --- |
| A: no extra pretraining | Original ESM-C weights → supervised mutation-effect training |
| B: sparse teacher distillation | Original ESM-C → sparse LoRA distillation → same supervised training |
| C: continued MLM control (optional) | Original ESM-C → continued MLM on the same unlabeled sequence exposure → same supervised training |

- [ ] Keep supervised data, split, sampler, optimizer tuning budget, prediction
  head, LoRA scope/rank, and total downstream training budget matched between A
  and B. Use the Megascale held-out-protein split fixed in
  `TODO_ESM_MSR_EXPERIMENT.md`, after its provenance audit.
- [ ] Run A and B on the same three seeds. Select checkpoints with validation
  only. Start with one short smoke run per arm to verify decreasing objective,
  gradients, reload, and cache correctness before the replicated comparison.
- [ ] Use validation-based early stopping for distillation with a fixed
  validation set of variants. Separately evaluate teacher matching on held-out
  validation proteins to measure transfer of the representation adaptation.
- [ ] If C is included, match sequence corpus, number of tokens, update budget,
  and optimizer access. It is optional; the first causal question is whether
  sparse-specific distillation helps over no extra pretraining.

### Outcomes

- [ ] Report downstream single/double prediction metrics per protein, direct
  epistasis metrics on matched WT/A/B/AB groups, stabilizing-variant retrieval,
  and validation-selected costs. Follow the metric caveats in the Megascale TODO.
- [ ] Report teacher/student representation mismatch before and after
  distillation on train, validation, and held-out test proteins, by layer and
  mutation depth. Test labels are never used to train the student or select
  targets/hyperparameters.
- [ ] Report accuracy versus downstream fine-tuning steps/data, not only final
  accuracy. A useful initialization should improve sample efficiency or final
  prediction under equal budgets; merely lowering teacher loss is insufficient.
- [ ] Measure total added cost: teacher target generation, storage, distillation
  time, and downstream inference/training. State when cached targets amortize.
- [ ] Treat improved representation matching without downstream improvement as
  a negative result for the proposed motivation. If downstream improvement
  appears only on train-like proteins, investigate family leakage and split
  generalization.

## Risks and interpretation

- Teacher matching transfers the teacher's behavior, including its biases and
  limits. It cannot establish that the teacher is biologically correct.
- Full-mutant teacher targets are expensive; cache them once and include their
  generation/storage in the cost comparison.
- Absolute hidden-state matching can overweight generic sequence information.
  Delta matching is more aligned with mutation effects but can be noisy; compare
  both and keep the primary objective prespecified.
- This is representation distillation, not sequence MLM or direct fitness
  pretraining. Keep those ideas as separate follow-up experiments.
- The sparse and full models implement different functions by design. A
  persistent teacher gap can be harmless if task predictions are competitive.
- No claim of mutation-pretraining benefit is supported until the matched
  downstream experiment improves prediction or reduces labeled-data/compute
  needs on held-out proteins.

## Deliverables

- [ ] Raw mismatch-measurement script and immutable variant manifest.
- [ ] Per-layer/per-protein mismatch report, plots, and runtime/memory record.
- [ ] Written gate decision with justification before training distillation.
- [ ] If gate passes, cached teacher-target generator and exact provenance.
- [ ] Matched A/B downstream runs, raw predictions, and results summary.
