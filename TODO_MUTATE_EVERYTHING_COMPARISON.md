# Comparison with Mutate Everything

Created 2026-11-04. Status: cDNA comparison code implemented; data downloaded
and audited, real-weight smoke passed, first validation-only ESM-2 pilot
launched. The released ESM-2 comparator has now been reproduced locally on all
cDNA2 rows: pooled Spearman 0.4539, macro Spearman 0.5326, nDCG@30 0.3792
(reported 0.38); local and original evaluators agree. The first frozen seed-0
pilot comparison is complete: pooled Spearman 0.5427 vs 0.4539, macro Spearman
0.6388 vs 0.5326, but nDCG@30 0.3600 vs 0.3792. The untuned three-seed
joint/additive sweep is complete. Systematic tuning is now running, with
validation-only selection and separate confirmation seeds. Start with the
[experiment index](experiments/MUTATE_EVERYTHING_EXPERIMENTS.md) and
[live tuning status](experiments/mutate_everything_tuning/STATUS.md). See
[implementation and commands](experiments/MUTATE_EVERYTHING_COMPARISON.md).
Run details and pinned data hashes:
[first pilot](experiments/MUTATE_EVERYTHING_PILOT_LAUNCH.md).
Measured comparator and reproduction evidence:
[baseline](experiments/MUTATE_EVERYTHING_BASELINE.md).
Exact matched-row pilot results:
[comparison](experiments/MUTATE_EVERYTHING_PILOT_COMPARISON.md).
Expanded S669, ProTherm, cDNA1 and reverse tests, plus completed joint/additive
seed repeats: [breadth results](experiments/MUTATE_EVERYTHING_BREADTH_RESULTS.md).

Implementation findings: upstream detection uses a 0.5 kcal/mol stabilization
threshold and sklearn's default nDCG tie averaging. The released ESM comparator
fine-tunes backbone layers: matching pretrained ESM-2 initialization does not
match final backbone weights or adaptation policy. Preserve that qualification
when interpreting the method comparison below.

## Question and scope

How does the sparse variant-cache model compare with Mutate Everything
(Ouyang-Zhang et al., *Predicting a Protein's Stability under a Million
Mutations*, NeurIPS 2023) on stability prediction, in accuracy and in the cost
of scoring many mutants?

**A sequence-only comparison is feasible.** Their released code provides an
ESM-2 backbone variant (`esm2_t33_650M_UR50D`) alongside the AlphaFold one,
with a downloadable checkpoint (`models/mutate_everything_esm.pth`), and they
release their cDNA split files. So this does *not* reduce to "sequence-only vs
structure-aware": the ESM-2 variant is the right comparator, and the AlphaFold
variant is a separate, clearly labelled row showing what structure buys.

**Backbone is matched by porting ESM-2** (§1a), so the headline row is our
method and theirs on the same `esm2_t33_650M_UR50D` weights, with our ESM-C
runs kept as a secondary row. This removes the main confound: an accuracy
difference on matched weights is attributable to the method, not the backbone.

What still differs, and must be stated rather than engineered away:

- **Decoder**: they sum per-mutation representations and decode with a 3-layer
  MLP; we use sparse cross-attention over mutated positions with LoRA.
- **Regime**: they decode every mutation from one pass; we score a supplied
  variant list. §4 keeps these separate.

**Decided 2026-11-04: accuracy is the question.** This method is expected to be
slower per mutant; the claim at stake is whether it sits on the accuracy/cost
Pareto frontier, i.e. whether it is *more accurate* than Mutate Everything at
some cost. So sections 1-3 and 5 are the work, and §4 drops to a single
order-of-magnitude cost number for the frontier axis rather than a full matched
benchmark.

For a Pareto claim, an accuracy win has to survive the obvious confound: our
backbone is not theirs. See §0 and the backbone-sensitivity control in §2.

Sources (fetched 2026-11-04; the paper-derived numbers below came via a page
summary and still need checking against the PDF directly):

- [Paper](https://arxiv.org/abs/2310.12979) /
  [NeurIPS version](https://proceedings.neurips.cc/paper_files/paper/2023/file/f03cb785864596fa5901f1359d23fd81-Paper-Conference.pdf)
- [Code and checkpoints](https://github.com/jozhang97/MutateEverything)
- Prior art for matching an external stack:
  [TODO_ESM_MSR_EXPERIMENT.md](TODO_ESM_MSR_EXPERIMENT.md) §5

## 0. Decisions needed from a human

- [x] **Which comparison**: accuracy (decided 2026-11-04).
- [x] **Backbone policy**: port ESM-2 650M and run our method on their
  backbone, for a matched-backbone comparison (decided 2026-11-04). §1a is the
  port. ESM-C runs stay as a secondary row.
- [ ] **Scope of their benchmark suite.** Their cDNA split is the core. S669
  and ProTherm Multiple are also released and would need new loaders; decide if
  the headline claim needs them.
- [ ] **Whether to re-run their model locally** (§2) or compare against
  published numbers only. Published-only forbids any superiority claim.

## 1a. Port ESM-2 650M as a backbone

Mostly mechanical: ESM-2 uses RoPE like ESM-C, so the cache machinery and
position handling transfer. It is *not* a literal drop-in, and these are the
concrete deltas (checked 2026-11-04 against
`transformers/models/esm/modeling_esm.py`, v4.57.6):

- [ ] **Gated vs plain MLP.** ESM-C uses a SwiGLU-style block
  (`gate`/`up`/`down`); ESM-2 uses `fc1`/`fc2` with ESM's own GELU, which the
  HF source explicitly warns is not `F.gelu` ("yields subtly wrong results").
  This changes the block and the valid `--lora-targets` names, which currently
  accept `q k v o gate up down`. Decide whether targets become
  backbone-dependent or a shared alias set.
- [ ] **Biases.** ESM-2's linears are `bias=True`; confirm ESM-C's are not and
  keep them separate rather than assuming one convention.
- [ ] **Query scaling quirk.** HF scales the query explicitly to match the
  original implementation when rotary is in play; mirror whichever reference we
  test against.
- [ ] **`token_dropout`.** ESM-2 rescales embeddings by the masked-token ratio.
  Probably inert for full-sequence scoring, but it is live for any
  masked-marginal baseline, so handle it explicitly rather than by omission.
- [ ] **Tokenizer and vocab.** ESM-2's alphabet and special tokens differ from
  `backbone/tokenizer.py`'s ESM-C vocab; the port needs its own.
- [ ] **Equivalence test**, per the repo convention that every backbone port is
  tested against the official package. Note the installed `esm` is
  EvolutionaryScale's (ESM-C/ESM3) and has no ESM-2, so decide the reference:
  HF `transformers` (already installed, weights already cached) or add
  `fair-esm`. Test in fp32, as the convention requires.
- [ ] Re-check the cache path's assumptions against the new block: the variant
  cache and LoRA wiring are written against ESM-C's module names.

## 1. Pin their data and protocol

- [ ] Download their released data (`cdna_train.csv`, `cdna2_test.csv`,
  `protherm_multiple.csv`, s669) and the ESM-2 checkpoint. Record file names
  and hashes, as ESM-MSR §5.1 does.
- [ ] **Their split is the comparison's foundation and is released**, so use it
  rather than reproducing it: train on `cdna_train`, test on `cdna2_test`.
  Verify the reported shape against the files: ~116 train proteins / ~213k
  mutations (~117k single, ~97k double), and cDNA2 as ~18 mini-proteins with
  ~22k doubles. Confirm MMseqs2 clustering and the ~36% identity ceiling
  between train and test.
- [ ] **This is a different split from the ESM-MSR work**, so it needs its own
  training runs. Nothing from `esm_msr_hyperopt` carries over.
- [ ] Add a loader for their cDNA tables under `mutant_cache/data/`, with the
  usual exclusion audit, reusing the Tsuboyama ingestion where the underlying
  rows are the same measurements.
- [ ] Audit overlap between their training proteins and our test proteins, both
  directions, and state it.
- [ ] Confirm the ΔΔG sign convention on the joined frame instead of trusting
  it (their stabilizing direction vs ours).

## 2. Reproduce their predictions locally

- [ ] Run their ESM-2 checkpoint on `cdna2_test` and reproduce their published
  numbers. **Gate:** our evaluator must reproduce their reported metrics from
  their own predictions before any cross-method number is quoted. If it cannot,
  label their results "published" and make no superiority claim.
- [ ] Inner-join their predictions with ours and evaluate both on identical
  rows; log unmatched rows rather than dropping them.
- [ ] Keep the AlphaFold-backbone numbers as a separate labelled row. Reported
  S669 singles Spearman is ~0.47 (ESM-2) vs ~0.56 (AlphaFold) — verify both
  from the paper before quoting.
- [ ] **Headline row: our method on ESM-2 650M**, their backbone weights, so
  the comparison isolates the method. Train under the §3 recipe on their split.
- [ ] **Secondary rows: ESM-C 300M and 600M**, which also give a
  backbone-sensitivity reading: how much of our accuracy is backbone rather
  than method. Worth reporting whichever way it comes out.

## 3. Match the metrics

Their metric set differs from ours in three ways that all need code:

- [ ] **Pooled Spearman across all mutations in a dataset** appears to be their
  headline, where ours is macro per protein. `pooled_metrics` already computes
  this; confirm which they report per table and never compare our macro against
  their pooled.
- [ ] **nDCG at K=30** (ours defaults to 96; `ndcg_at_k` takes `k`, so this is
  a parameter change, not new code). Verify their relevance definition and gain
  match ours (floor at 0, linear gain, tie handling) against their code.
- [ ] **Detection Precision at K=30** — the fraction of experimentally
  stabilizing mutations among the top 30 predicted. Our `retrieval_metrics`
  already returns a precision at k; check whether the definitions coincide
  exactly or whether theirs is a distinct quantity, and add it if so.
- [ ] Their doubles use a **learned higher-order aggregation** (summed
  representations into an MLP), not additivity in ΔΔG space. So their doubles
  are a genuine comparator for our `joint` composition, and our `additive`
  composition is the shared null for both. State this explicitly: it is the
  most interesting part of the comparison.
- [ ] Check whether they report a doubles-only breakdown or only pooled
  singles+doubles, and match it.

## 4. Cost, as the Pareto frontier's other axis (deprioritized)

Accuracy is the question, and this method is accepted to be slower per mutant,
so cost needs only enough measurement to place both methods on the frontier —
not the full matched benchmark the earlier draft planned. Their published
figure is ~0.6 s for a 317-residue protein with the ESM-2 backbone, covering
all singles and doubles (verify from the paper).

- [ ] Report our measured throughput on the §2 evaluation runs: mutants per
  second, cold time, warm throughput, peak memory, on one A6000 (GPU 0 or 2).
  Never infer speed from run wall-clock, and account for the fixed per-forward
  launch cost measured earlier.
- [ ] Place both methods on one accuracy-vs-cost plot, stating that the cost
  numbers are same-GPU-class but not a controlled benchmark.
- [ ] Say plainly which regime each method is built for: they decode every
  mutation from one pass, we score a supplied sparse library. A frontier plot
  that ignores this flatters whichever method the framing favours.

Only if a cost claim becomes load-bearing:

- [ ] Add a saturation-scan driver (enumerate all L x 19 singles plus a double
  budget from one cached reference), verify its scores against the normal path
  on a small protein before timing, and run their ESM-2 variant on the same
  enumeration, same GPU, same batching.

## 5. Reporting

- [ ] Use their summary convention for the head-to-head, with our macro
  per-protein numbers alongside and clearly labelled.
- [ ] Paired per-protein differences with protein-bootstrap intervals, seeds
  retained. Three seeds minimum before calling a difference real.
- [ ] State the system differences in the same breath as any number: decoder
  and regime. With the backbone matched, say so explicitly, and keep the ESM-C
  rows labelled as a different backbone. Name their ESM-2 variant as the
  comparator and keep AlphaFold separate.
- [ ] State what was not controlled; current repo results are single-seed or
  validation-only.

## Deliverables

- [ ] Pinned data note: file hashes, split shape checks, overlap audit.
- [ ] A joined prediction frame and matched-evaluator table, or an explicit
  statement that only published numbers were available.
- [ ] An ESM-2 650M backbone in `mutant_cache/backbone/` with its equivalence
  test passing in fp32.
- [ ] Matched-backbone head-to-head on their split, plus ESM-C rows as a
  backbone-sensitivity reading.
- [ ] An accuracy-vs-cost placement for both methods, with the cost caveat.
- [ ] A limitations paragraph covering backbone, decoder and regime.
