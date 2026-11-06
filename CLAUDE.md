# CLAUDE.md

Sparse variant KV-cache for protein language models: cache one reference
sequence's per-layer states, then recompute only the mutated positions for
thousands of mutants. For current work see [README.md](README.md),
[experiments/INDEX.md](experiments/INDEX.md) and the
[research journal](experiments/RESEARCH_JOURNAL.md); [INTERFACES.md](INTERFACES.md)
holds the contracts between modules. The original work list is archived at
[archive/2026-11-05/PLAN.md](archive/2026-11-05/PLAN.md) (history only).

## Conventions

- **Env:** `~/anaconda3/envs/mutant_cache/bin/python` (cloned from svar + `esm`).
- **Scratch:** raw downloads and caches go in `$MUTANT_CACHE_SCRATCH`
  (default `~/mutant_cache_scratch`), never in the repo.
- **One-off / experiment / scratch scripts: keep the raw version, no processing.**
  Copy experiment scripts into `experiments/` exactly as run — no argparse, no
  refactoring, hardcoded paths are fine.
- **Precision:** experiments run a frozen bf16 base with fp32 heads/adapters
  (`harness --precision` defaults to bf16). Equivalence tests, oracles, and
  sanity notebooks stay fp32 (`load_esmc` default) — bf16 noise would hide bugs.
- **Correctness oracle first.** Every cache path has a bruteforce counterpart
  and a test comparing them; every backbone port has an equivalence test
  against the official `esm` package.
- Origin: the cache method was developed for DNA LMs in ~/svar
  (`CARBON_modules/variant_cache_layers.py`, `DNABERT2_modules/variant_cache_layers.py`,
  `CARBON_modules/variant_lora_layers.py`).
