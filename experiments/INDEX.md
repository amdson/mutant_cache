# Experiment index

[Research journal](RESEARCH_JOURNAL.md) records findings. JSON definitions specify experiments; scratch run directories hold actual executions. A proposed experiment is not a completed measurement.

| Study | Definition / entry point | Status |
| --- | --- | --- |
| Mutate Everything, ESM-2 | [me_esm2_v1](stability_paper/me_esm2_v1/README.md) | New explicit development recipe; completed full-data comparison linked separately |
| ESM-MSR, ESM-2 | [msr_esm2_v1](stability_paper/msr_esm2_v1/README.md) | Training protocol available; matched external reproduction remains incomplete |
| StabilityArc, ESMC | [arc_esmc_v1](stability_paper/arc_esmc_v1/README.md) | Planned; dataset and evaluator integration required |
| E02 accuracy round 2 | [plan](mutate_everything_accuracy_v2/README.md), [status](mutate_everything_accuracy_v2/STATUS.md) | Existing queue; inspect live status before interpreting |
| Completed full-data ME comparison | [protocol](mutate_everything_full_comparison/README.md) | Existing frozen results under scratch |
| Matched ME evaluator | [usage](mutate_everything_matched/README.md) | Existing reusable scoring/reporting CLI |

## Exploratory work

Run `python -m mutant_cache.train` with a recipe/protocol and CLI overrides. A named experiment is optional. Put raw one-off launch scripts in [dev](dev/README.md); do not refactor scripts that have already produced results. Recipes supply settings, while experiment definitions identify a question, protocol and intended repetitions. A single CLI launch runs one seed; it does not automatically execute a seed/fold grid.

Historical materials are indexed in [archive](../archive/README.md). Do not use old plans as current configuration specifications.
