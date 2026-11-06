# Sparse variant cache

Train protein variant-effect models by reusing reference states and recomputing mutated positions. Start with the [research journal](experiments/RESEARCH_JOURNAL.md) for results and the [literature notes](docs/VARIANT_EFFECT_NOTES.md) for related work.

## Everyday experiments

Use the existing environment (`~/anaconda3/envs/mutant_cache/bin/python`). From this repository:

```bash
python -m mutant_cache.train --help
python -m mutant_cache.train \
  --protocol datasets/protocols/mutate_everything_v1.json \
  --recipe experiments/recipes/esm2_650m_shared.json \
  --seed 0 --dry-run
```

Remove `--dry-run` to train after preparing the data. Ordinary training arguments can override a recipe; no experiment registration is required. Use `--help` for supported flags. The CLI and `mutant_cache.training.train` wrap the existing Megascale implementation; the original entry points remain available. Training data and pretrained weights must already be available.

Each new run records its resolved settings, source and data provenance, and outputs under `$MUTANT_CACHE_SCRATCH` (default `~/mutant_cache_scratch`). The existing training queue remains available through `python -m mutant_cache.tuning --help`; its historical plans and checkpoint-template interface remain compatible.

## Repository map

| Location | Purpose |
| --- | --- |
| `mutant_cache/` | Reusable model, cache, training, data and evaluation code |
| `datasets/` | Source records, protocol definitions and Make dependencies |
| `experiments/recipes/` | Explicit model/training settings |
| `experiments/stability_paper/` | Named experiment and comparison definitions |
| `experiments/dev/` | Optional raw exploratory launch scripts |
| `analysis/` | Analysis entry points consuming saved results |
| `tests/` | Correctness oracles and focused checks |
| `archive/` | Historical plans, reports, scripts and small artifacts |
| `$MUTANT_CACHE_SCRATCH` | Raw/processed data, weights, caches, runs and reports |

To set up another machine, see [new server setup](docs/NEW_SERVER_SETUP.md). For parallel runs on all GPUs, use `python -m mutant_cache.queue_runner`. See [datasets](datasets/README.md) for build commands and [experiment index](experiments/INDEX.md) for studies. Make is an optional convenience layer: `make help`. It never launches training as a dependency of preparing data or producing a report.

## Research scope

The first paper compares sparse stability prediction under separate Mutate Everything, ESM-MSR and StabilityArc protocols. The latter is a planned integration, not an implemented dataset or reproduced result. Existing metric implementations remain protocol-specific; the matched-suite CLI currently evaluates Mutate Everything panels only. Combined-corpus supervised pretraining is a separate future track.

Historical run directories and data paths are retained. The reorganization does not move large scratch artifacts, retrain models, or rewrite completed scientific results. Follow [CLAUDE.md](CLAUDE.md) for code and experiment conventions.
