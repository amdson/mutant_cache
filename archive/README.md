# Archive

Historical project material that is no longer current guidance. Everything here
was moved, not rewritten or deleted; use it for provenance, not as current
configuration. Current entry points are the [repository README](../README.md),
the [experiment index](../experiments/INDEX.md) and the
[research journal](../experiments/RESEARCH_JOURNAL.md).

## Snapshots

| Snapshot | Contents |
| --- | --- |
| [2026-11-05](2026-11-05/README.md) | 144 files (~1.7 MB) from the repository root and `experiments/` |

### 2026-11-05

Paths keep their original repository-relative layout beneath
`archive/2026-11-05/` (for example `experiments/phase0_runtime.py` is now
`archive/2026-11-05/experiments/phase0_runtime.py`).

- **Root plans and handoffs (12):** [PLAN.md](2026-11-05/PLAN.md),
  [TODO.md](2026-11-05/TODO.md), [DATASETS.md](2026-11-05/DATASETS.md),
  [VARIANTS_AND_DATASETS.md](2026-11-05/VARIANTS_AND_DATASETS.md),
  [RESEARCH_DIRECTIONS.md](2026-11-05/RESEARCH_DIRECTIONS.md),
  [PAPER_PLAN.md](2026-11-05/PAPER_PLAN.md),
  [PAPER_FINDINGS.md](2026-11-05/PAPER_FINDINGS.md),
  [HANDOFF.md](2026-11-05/HANDOFF.md) (October 21),
  [HANDOFF_2026-10-24.md](2026-11-05/HANDOFF_2026-10-24.md),
  [TODO_ESM_MSR_EXPERIMENT.md](2026-11-05/TODO_ESM_MSR_EXPERIMENT.md),
  [TODO_ARCHITECTURE_MISMATCH_PRETRAINING.md](2026-11-05/TODO_ARCHITECTURE_MISMATCH_PRETRAINING.md),
  [TODO_MUTATE_EVERYTHING_COMPARISON.md](2026-11-05/TODO_MUTATE_EVERYTHING_COMPARISON.md).
- **Completed experiment reports (8):**
  [implementation pilot E0–E4](2026-11-05/experiments/IMPLEMENTATION_PILOT_RESULTS.md),
  [ThermoMPNN split](2026-11-05/experiments/THERMOMPNN_SPLIT_RESULTS.md),
  [Mutate Everything baseline](2026-11-05/experiments/MUTATE_EVERYTHING_BASELINE.md),
  [pilot launch](2026-11-05/experiments/MUTATE_EVERYTHING_PILOT_LAUNCH.md),
  [pilot comparison](2026-11-05/experiments/MUTATE_EVERYTHING_PILOT_COMPARISON.md),
  [breadth results](2026-11-05/experiments/MUTATE_EVERYTHING_BREADTH_RESULTS.md),
  [comparison](2026-11-05/experiments/MUTATE_EVERYTHING_COMPARISON.md),
  [old experiment index](2026-11-05/experiments/MUTATE_EVERYTHING_EXPERIMENTS.md).
- **Raw experiment scripts and artifacts (124):** the scripts (`.py`, `.sh`),
  CSVs, logs, summaries and manifests behind those reports — phase-0
  degradation/runtime, cache-vs-exact, METL GB1/avGFP, DMS tuning, E0–E4,
  `am0_*`, ESM-MSR screens, and the Mutate Everything pilot/breadth scripts.
  They are kept exactly as run; hardcoded `experiments/...` paths inside them
  refer to their pre-archive locations.

Not archived (still at original paths): model code, tests, data build rules,
`INTERFACES.md`, `AGENTS.md`, `CLAUDE.md`, `docs/VARIANT_EFFECT_NOTES.md`,
`experiments/RESEARCH_JOURNAL.md`, `experiments/PHASE0_RESULTS.md`, the active
Mutate Everything training/holdout scripts, and the live
`experiments/mutate_everything_accuracy_v2/` run plan and status.

## Verifying a snapshot

Each snapshot has a `moves.json` listing `original_path`, `archive_path`,
`bytes` and `sha256` of the file's pre-move content (identical to
`git show 4f74a31:<original_path>`, the last commit before the move). Check it with:

```bash
python - <<'EOF'
import hashlib, json, os
for e in json.load(open('archive/2026-11-05/moves.json')):
    b = open(e['archive_path'], 'rb').read()
    ok = len(b) == e['bytes'] and hashlib.sha256(b).hexdigest() == e['sha256']
    if not ok or os.path.exists(e['original_path']):
        print('CHECK', e['archive_path'])
EOF
```

Run from the repository root. Three archived Markdown files report `CHECK` by
design: their relative link targets were rewritten after the move so they still
resolve (`PAPER_FINDINGS.md`, `TODO_MUTATE_EVERYTHING_COMPARISON.md`,
`experiments/MUTATE_EVERYTHING_EXPERIMENTS.md`). Compare them against
`git show 4f74a31:<original_path>`; the only differences are link targets.
