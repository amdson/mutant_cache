# Analysis

Analysis consumes saved predictions and reports; it must not silently train models or choose checkpoints using test results.

For current matched Mutate Everything reports use `python -m mutant_cache.matched_suite --help`. It writes per-seed/per-protein metrics, coverage checks, bootstrap results and hashes. Paper-specific plotting and table scripts can be added here as their inputs are frozen. Existing historical analysis scripts are retained in the archive or their operational experiment location.

Exploratory notebooks belong in `analysis/notebooks/` for new work; original notebooks remain in `notebooks/` to preserve references. No new paper-level aggregate is defined yet: metrics from different protocols should not be averaged together by default.
