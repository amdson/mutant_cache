# Protein dataset shortlist: sparse-variant model

Oct 16, 2026 · @Andrew

MaveDB in bulk is the training pool — over 7 million measurements, any target, one 1.4 GB download. Tsuboyama, CombinGym and ProteinGym are the parts worth pulling separately, for multi-mutants and for evaluation.

## Core training pool

Start with the [MaveDB bulk dump](https://zenodo.org/records/14172004): every published CC0 score set as one zip, 1.4 GB at v2 (Nov 2024), with newer versions on the same record. It held over 7 million variant effect measurements across 1,884 datasets as of November 2024, and a September 2026 analysis counts 2,803 score sets. Anything not CC0 comes through the [API](https://github.com/VariantEffect/mavedb-api) instead. Targets are heterogeneous — abundance, binding, fluorescence, growth, enzymatic activity — which is fine: rank-normalize within score set and condition the head on assay identity.

Within that pool, the Tsuboyama Megascale files are worth pulling separately from [Zenodo](https://zenodo.org/records/7844779), because they are the only multi-protein set with paired singles and multi-mutants in one assay. Check for a version newer than v2\_230420.

| Artifact | Contents | Why you need it |
| --- | --- | --- |
| `Processed_K50_dG_datasets.zip` → `Tsuboyama2023_Dataset1_*.csv` | Curated ΔG per variant, 776,298 measurements over 479 domains | Primary training labels |
| `Double_DMS_list.csv` | Index of the 210,118 double mutants, 559 site pairs across 190 domains | The interaction training signal — the whole point |
| `Triple_DMS_list.csv` | Index of triple mutants | Depth-3 holdout without leaving the assay |
| `Single_DMS_list.csv` | Index of the single-mutant scans | Additive baseline and the ε construction |
| `K50_dG_tables.zip` | Per-library K50 and ΔG with uncertainties | Reliability ceiling on ε; needed to report against it |
| `Raw_NGS_count_tables.zip` (1.0 GB) | Per-library read counts | Only if you want your own noise model — skip on first pass |

The reason to care about ΔG is not the units but the link function. Across MaveDB each assay has its own monotone squashing between an additive latent trait and the reported score, so a model trained across score sets will spend capacity learning those links. For a first run — does the architecture train, does the cache give the speedup — rank-normalization absorbs this and it does not matter. It matters only once you claim to predict interactions, at which point ΔG data is the clean case and everything else needs a fitted link.

Two filters worth applying to the dump: most score sets are single mutants, so index by maximum mutation count per score set before deciding what trains what; and reported correlations are capped by assay reproducibility — 2,452 of 2,803 score sets carry enough to estimate it, though only about a tenth are findable by a column-name search, and 30–52% of human deposits with a computable ceiling fall below 0.90.

Optional second pull: the [MGnify Stability dataset](https://www.biorxiv.org/content/10.64898/2026.05.19.726285.full.pdf) (Cho, Tsuboyama, Rocklin, May 2026) has >1.8 million sequences and >200,000 unique protein clusters with measured ΔG for domains under 80 aa. These are diverse natural domains, not mutational scans, so they carry no coupling signal — but they are a large labelled corpus for pretraining the reference encoder if the variant branch underfits.

## Multi-mutant sets: one repo covers most of them

[CombinGym](https://github.com/sitonglab/CombinGym) curates 14 combinatorial variant datasets spanning 9 proteins and over 400,000 data points, with 4 to 16 mutations per landscape, and ships 0/1/2/3-vs-rest splits with a published leaderboard. Pull this instead of chasing the individual papers.

| Source | What it gives you | Note |
| --- | --- | --- |
| CombinGym (repo + [combingym.org](https://www.combingym.org/)) | CR9114 (16 mutations, 3 antigens), CR6261 (11 mutations, 2 antigens), SpCas9/SaCas9, a 2^11 fluorescent-protein set, 9 proteins total | Splits and baselines (ESM-1b/1v, EVmutation, DeepSequence, GVP-MSA, MAVE-NN, CNN, Ridge, BLOSUM62) already defined |
| [FLIP](https://github.com/J-SNACKKB/FLIP) GB1 landscape | 149,361 variants at V39/D40/G41/V54; `2-vs-rest` trains on HD ≤ 2 and tests on HD 3–4 | The canonical depth-extrapolation split; raw data at `data.bioembeddings.com/public/FLIP` |
| FLIP2 (ICML 2026) | Seven further datasets — enzymes, protein-protein interactions, light-sensitive proteins | Optional; its headline finding is that simple models often beat fine-tuned pLMs |
| avGFP, via ProteinGym `GFP_AEQVI_Sarkisyan_2016` | 51,715 random genotypes, mean \~3.7 substitutions, up to \~15 | Closest to your 1% target density; also the depth-5+ bucket |
| Olson GB1, via ProteinGym `SPG1_STRSG_Olson_2014` | \~537k near-complete singles + doubles over 56 residues | Dense coupling matrix for one protein |

Two FLIP quirks that will silently break comparability: the repo marks splits with a green/orange/red semaphore and only green ones should be used for reported numbers, and GB1 non-functional sequences are down-sampled because over 96% of mutants fall below 0.5 fitness. Match both or your numbers are not on the same scale as the leaderboard.

Unlike Tsuboyama, these are fitness assays, so apply a monotone link per assay before computing ε. Without it a sigmoid fit will masquerade as learned epistasis.

## Held-out benchmarks

| Benchmark | Access | What it tests | Leakage against Tsuboyama |
| --- | --- | --- | --- |
| ProteinGym v1.3 substitutions | [GitHub](https://github.com/OATML-Markslab/ProteinGym), [Zenodo](https://zenodo.org/records/14997691), `ProteinGymPy` | 217 assays, \~2.7M variants; Spearman stratified by mutation depth 1/2/3/4/5+ | Overlaps — its curated Tsuboyama subset is 64 proteins and 117,811 sequences |
| Human Domainome 1 | Supplementary tables of [Beltran et al. 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11754108/) | 563,534 variants across 522 domains, singles only | Clean — not in ProteinGym |
| CombinGym leaderboard | combingym.org | Depth extrapolation with published baselines | Clean |

The depth-stratified columns on ProteinGym are the single most legible way to present a result: every published model drops from depth 1 to depth 2+, so a model that holds flat explains itself in one table. To keep that comparison honest, exclude from training every protein that appears in ProteinGym under any assay, then cluster-filter the remainder at 30% identity. Pull `DMS_substitutions.csv` from the repo first and diff it against the Megascale domain list — that diff decides your training pool.

One comparison to avoid: the zero-shot leaderboard. VenusREM at ρ = 0.523 and VenusREM2 at 0.556 see no DMS labels, so placing a supervised number beside them is a category error. Your comparators are the supervised ProteinGym entries, the CombinGym baselines, and your own additive ablation with variant-variant attention masked.

## Baselines

The ablations matter more than the published comparators, because only they isolate what the architecture contributes. Training across MaveDB rather than one assay invites a specific degenerate solution — a per-position tolerance predictor with an assay embedding — and three of the four ablations below exist to detect it.

| Ablation | What it removes | What a tie means |
| --- | --- | --- |
| Additive | Variant-variant attention masked | No epistasis is being learned; the interaction module is dead weight |
| No cross-attention | Variant tokens cannot read cached reference states | ESM-C context contributes nothing; you have a learned lookup over (position, residue) |
| Random reference states | ESM-C activations replaced by random vectors of matching shape | The architecture, not the pretrained features, is doing the work |
| Position-only tokens | Alt-residue identity dropped, position kept | You built a conservation predictor — the main risk of MaveDB-wide training |

| Comparator | Where | Covers |
| --- | --- | --- |
| [ESMRank](https://www.biorxiv.org/content/10.64898/2026.02.26.708185.full.pdf) | \~1,100 MaveDB score sets, >2M variants, strict protein-level partitioning | Closest existing work — same data, same split philosophy; read before building |
| Ridge on one-hot, per assay | FLIP baseline | FLIP2 found simple models often match fine-tuned pLMs |
| Ridge / small MLP on frozen ESM-C embeddings | Your own | The real supervised bar, given TabPFN3 at 0.767 mean Spearman on ESM-C 300M features |
| ESM-C masked marginals | Your own | Also the additive initialization |
| ProteinNPT, Kermut | ProteinGym supervised leaderboard | Within-assay supervised setting |
| CombinGym suite (ESM-1b/1v, EVmutation, DeepSequence, GVP-MSA, MAVE-NN, CNN, Ridge, BLOSUM62) | CombinGym repo | Depth-extrapolation splits |
| ThermoMPNN | Trained on Megascale | Only if reporting on the stability subset |

Zero-shot leaders (VenusREM 0.523, VenusREM2 0.556, ProSST) belong in a context row labelled as using no labels, never as competitors.

Compute baseline: full ESM-C recompute per variant, same trained weights. Report throughput ratio and the accuracy delta as a function of mutation count — the analogue of the 99% cosine similarity at 50 edits from the SSM version.

Report the reliability ceiling beside every correlation. Per-assay reproducibility caps what is achievable, and 30–52% of human MaveDB deposits with a computable ceiling fall below 0.90, so 0.6 against a test-retest of 0.7 reads nothing like 0.6 against 0.95.

## Download order

The first two items are enough to run the experiment that decides whether the model works — train on Megascale singles for a domain, test on its doubles, compare against the additive ablation. Pull the rest only if that clears.

- [ ] MaveDB bulk dump — the training pool; index it by maximum mutation count per score set on arrival
- [ ] FLIP GB1 `2-vs-rest` — smallest download, and the additive baseline on it tells you early whether interaction capacity is worth building
- [ ] Tsuboyama `Processed_K50_dG_datasets.zip` + `K50_dG_tables.zip` — the multi-mutant subset and its uncertainties
- [ ] ProteinGym `DMS_substitutions.csv` reference file — needed before training, to define the exclusion list
- [ ] CombinGym repo — depth-extrapolation evaluation and baselines
- [ ] ProteinGym full substitution benchmark (5.9 GB on Zenodo) — only once you are ready to report stratified numbers
- [ ] Human Domainome 1 supplementary tables — clean held-out set for the additive component

Deferred by the protein-first decision: antibody repertoires (OAS) for pretraining, bacterial pangenomes, anything genome-scale. None of them are needed to test whether the architecture holds.

## Sources

- [MaveDB bulk data dump — Zenodo](https://zenodo.org/records/14172004) and [mavedb-api](https://github.com/VariantEffect/mavedb-api)
- [Tsuboyama et al. 2023, Megascale — Zenodo record](https://zenodo.org/records/7844779)
- [Cho, Tsuboyama, Rocklin 2026, MGnify Stability](https://www.biorxiv.org/content/10.64898/2026.05.19.726285.full.pdf)
- [CombinGym repository](https://github.com/sitonglab/CombinGym) and [preprint](https://www.biorxiv.org/content/10.64898/2026.03.24.714074.full.pdf)
- [FLIP repository](https://github.com/J-SNACKKB/FLIP)
- [ProteinGym repository](https://github.com/OATML-Markslab/ProteinGym) and [Zenodo](https://zenodo.org/records/14997691)
- [Beltran et al. 2025, Human Domainome 1](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11754108/)
- [Measurement reliability bounds functional benchmarks, 2026](https://www.biorxiv.org/content/10.64898/2026.09.14.751496.full.pdf)
