# Repository reorganization: remaining work

Status: **completed 2026-11-05** (items 1–6 below done; see the journal entry "Repository reorganization"). Remaining known gaps: StabilityArc is planned only; `make -C datasets` does not regenerate the holdout1/2 split parquets, `sign_and_overlap_audit.json` or `largest_external_label_differences.csv`; the ESM-MSR raw archive is a manual fetch; `experiments/mutate_everything_external_driver.py` calls the now-archived `mutate_everything_replicas.py` (raw script, left unchanged).

Original status: The reorganization is intentionally a thin layer over existing training and benchmark code. No model training, data downloads, or changes to active experiment runs are part of this work.

## Started

- Added repository entry points and maps: `README.md`, root `Makefile`, `experiments/INDEX.md`, `analysis/README.md`, and `experiments/dev/README.md`.
- Added proposed experiment records for Mutate Everything, ESM-MSR, and StabilityArc under `experiments/stability_paper/`. StabilityArc is clearly marked planned.
- Copied the already-completed Mutate Everything full-data comparison manifest into its new experiment directory. Its labels and checkpoints still live under scratch.
- The dataset, CLI, and archive pieces are being completed in parallel. Do not assume their files, commands, or links are ready until checked below.
- The E02 round-2 tuning workers and holdout coordinator were active when this handoff was written. Keep `experiments/mutate_everything_accuracy_v2/` and every referenced running-run artifact at its existing path.
- The user explicitly asked to keep `docs/VARIANT_EFFECT_NOTES.md` and `experiments/RESEARCH_JOURNAL.md` outside the archive.

## Finish before calling the reorganization done

1. **Collect the three parallel changes.** Inspect CLI files, dataset rules/protocols/recipes, and archive manifest. Ensure `python -m mutant_cache.train --help`, root `make help`, and all referenced paths agree. The CLI runs one seed per invocation. Do not claim StabilityArc data/import/evaluation is implemented unless it is.
2. **Check dataset workflow without downloading or overwriting data.** Run `make -n -C datasets mutate-everything` and `make -n -C datasets esm-msr`; inspect the generated rules for correct existing input/output paths. StabilityArc must explain which release/protocol artifacts are still needed or fail clearly while marked planned. Preserve raw scripts and the scratch data tree.
3. **Review archive completeness and safety.** Confirm moves are represented by a manifest with hashes; the two protected documents remain in place; live round-2 files and reusable benchmark interfaces remain available; and all moved documentation links are fixed. Keep `AGENTS.md`, `CLAUDE.md`, `pyproject.toml`, code, tests, and operational guidance available. Archiving means moving and preserving, not deleting.
4. **Resolve links, not history.** Update root guidance that points to archived `PLAN.md`/`INTERFACES.md`; make journal links to archived findings navigable. Retain the original scripts/results for historical provenance. Avoid changing historical logs or source snapshots.
5. **Check the handoff and working tree.** Run `git diff --check`; inspect `git status --short` for accidental removals or writes to an active experiment; confirm the sample recipes use supported `megascale` argument names. If doing a tiny CLI smoke test, use `--dry-run` and an existing processed dataset; never start a GPU job by accident.
6. **Record the reorganization in the journal.** Add a brief dated entry distinguishing files actually moved from work that remains planned. Do not edit the journal to claim checks or dataset support that were not completed.

## Known boundaries

- `mutant_cache.matched_suite` is currently specialized to Mutate Everything labels and metrics. It is not yet a general ESM-MSR or StabilityArc evaluator. Keep the corresponding new README honest.
- ESM-MSR training data preparation exists, but a matched external comparison still has unresolved checkpoint provenance, row mapping, metric and reproduction requirements.
- StabilityArc currently has a proposed protocol/experiment record only; import, released holdouts, baseline reproduction, and matched evaluator are future work.
- The new ESM-2 and ESMC recipes are starting points, not selected or benchmark-validated training configurations.
- The combined-corpus supervised pretraining project remains separate from the stability comparison.

## Useful starting points

- Read `experiments/RESEARCH_JOURNAL.md` for research state and live status.
- See `datasets/README.md` for confirmed data targets once the dataset change lands.
- See `experiments/INDEX.md` for experiment states.
- Preserve active run state under `$MUTANT_CACHE_SCRATCH`; do not rebuild the same processed dataset path during this reorganization.
