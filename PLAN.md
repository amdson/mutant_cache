# mutant_cache: work list

Ordered so the go/no-go experiment comes first. Dataset notes: DATASETS.md.

## Phase 0 — port + validate cache on a protein LM (go/no-go)
1. Hackable ESM-C 300M reimplementation (`mutant_cache/backbone`), numerically matching the official `esm` package. ESM-2 650M fallback.
2. Bidirectional variant cache (`mutant_cache/cache`): variant queries attend to {ref keys at non-variant positions} ∪ {variant keys}; two attentions merged by log-sum-exp; reference kept at batch 1; RoPE per absolute position; QK-norm on both streams.
3. Bruteforce oracle + equivalence tests.
4. Degradation vs mutation count (1..50) on real DMS proteins: hidden-state cosine + masked-marginal LLR agreement, cache vs full forward.
5. Runtime sweep: seq length x n_mut x batch; cache vs full forward.
6. Decision.

## Phase 1 — repo skeleton
7. `mutant_cache/{backbone,cache,data,models,eval,baselines}`, `experiments/` (raw scripts), `datasets/` Makefiles into scratch.
8. One shared harness + thin run.py per experiment, run record.
9. `ReferenceCache` keyed by (target_id, seq hash), persisted to disk.

## Phase 2 — data
10. MaveDB bulk dump loader + index by max mutation count, assay type, reliability; rank-normalize per score set.
11. FLIP GB1 2-vs-rest (green splits only, match downsampling).
12. Tsuboyama Megascale: Dataset1 + single/double/triple lists; epsilon = ddG_double - sum singles; uncertainties as ceiling.
13. Exclusion: ProteinGym DMS_substitutions.csv vs training targets; drop overlap; mmseqs2 30% clustering; protein-level split.
14. Per-assay monotone link (later).
15. CombinGym, full ProteinGym, Domainome 1 (later).

## Phase 3 — model
16. Variant tokens (position, alt residue) cross-attending cached ref states every layer; variant-variant attention = interaction module.
17. Variant-only LoRA; frozen bf16 base, fp32 adapters.
18. Head: pooled variant tokens + ref summary + assay embedding; additive part initialised from ESM-C masked marginals.
19. Loss: within-score-set rank/pairwise or MSE on rank-normalized; ddG MSE for Megascale.
20. Ablation switches: additive, no-cross-attn, random-ref, position-only.

## Phase 4 — baselines
21. One-hot ridge per assay; ridge/MLP on frozen ESM-C embeddings; ESM-C masked-marginal zero-shot.
22. Full-recompute control with the same weights: throughput ratio + accuracy delta by depth.
23. Published numbers (CombinGym, ProteinNPT/Kermut, ESMRank).

## Phase 5 — eval
24. Depth-stratified Spearman (1/2/3/4/5+) beside reliability ceiling.
25. Epsilon correlation on Megascale doubles/triples.
26. Zero-shot rows labelled as context only.

## Phase 6 — first real experiment (the gate)
27. Train Megascale singles -> test doubles vs additive ablation; GB1 2-vs-rest as second check.

## Open questions
- ESM-C license (300M/600M).
- Ref summary: mean-pool vs learned query.
- MGnify stability pretraining if variant branch underfits.
