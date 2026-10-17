"""Eval layer: depth-stratified Spearman, epsilon, and the results table."""
from mutant_cache.eval.depth import (
    ALL, DEPTH_BUCKETS, depth_bucket, depth_spearman, depth_summary,
)
from mutant_cache.eval.epsilon import (
    attenuation_ceiling, build_epsilon_table, epsilon_correlation,
)
from mutant_cache.eval.table import ResultRow, ResultsTable, write_results_table

__all__ = [
    "depth_spearman", "depth_bucket", "depth_summary", "DEPTH_BUCKETS", "ALL",
    "build_epsilon_table", "epsilon_correlation", "attenuation_ceiling",
    "ResultsTable", "ResultRow", "write_results_table",
]
