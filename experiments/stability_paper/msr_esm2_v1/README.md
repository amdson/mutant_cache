# ESM-MSR / ESM-2

This definition makes an E02-style development run explicit using the existing ESM-MSR dataset importer and training implementation. It is not a claim of reproducing ESM-MSR's architecture, training recipe or final external results.

```bash
python -m mutant_cache.train --experiment experiments/stability_paper/msr_esm2_v1/experiment.json --dry-run
```

One launch runs one seed. Final comparison requires frozen fits, matched upstream rows and protein grouping, verified metric definitions and the appropriate released comparator checkpoint. No runnable `comparison.json` is supplied until that matching is completed. The existing Mutate Everything matched-suite evaluator must not be used as if it implemented ESM-MSR's nDCG@96 and grouping rules.
