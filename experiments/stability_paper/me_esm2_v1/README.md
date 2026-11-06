# Mutate Everything / ESM-2

`experiment.json` is a development recipe on the existing internal protein split. Resolve it with:

```bash
python -m mutant_cache.train --experiment experiments/stability_paper/me_esm2_v1/experiment.json --dry-run
```

The CLI runs one seed at a time (`--seed 0`, `--seed 1`, etc.). It does not launch an entire sweep or external evaluation. Changing settings creates a new run; it does not revise historical results.

`comparison.json` is a copy of the completed full-116-protein comparison definition and points to its original frozen checkpoints and canonical labels in scratch. It is deliberately separate from the new development experiment. Validate without inference using:

```bash
python -m mutant_cache.matched_suite --config experiments/stability_paper/me_esm2_v1/comparison.json \
  --output /tmp/unused_report --validate-only
```

`--output` is required by the parser but nothing is written with `--validate-only`.

The [original final-fit protocol](../../mutate_everything_full_comparison/README.md) remains authoritative for those completed fits. To evaluate new fits, create a new comparison definition with explicit checkpoint paths and actual training provenance. Do not substitute internally selected 104-protein fits while retaining the full-116 description.
