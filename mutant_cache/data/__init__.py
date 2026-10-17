"""Canonical data tables for mutant_cache. See INTERFACES.md section 2.

    from mutant_cache.data import load_variants, parse_mutant
    parse_mutant("A23G:K45R")   # -> [(23, "A", "G"), (45, "K", "R")]
    load_variants("tsuboyama")  # -> polars DataFrame with the canonical columns
"""

from .mutations import (
    AA20,
    ConflictingSubstitutions,
    UnsupportedHGVS,
    WTCheck,
    diff_sequences,
    format_mutant,
    mutate,
    n_mut,
    parse_hgvs_pro,
    parse_mutant,
    suggest_offsets,
    verify_wt,
)
from .paths import (
    DATASETS,
    exclusion_dir,
    processed_dir,
    raw_dir,
    scratch,
    splits_dir,
)
from .schema import (
    SCORE_SETS_SCHEMA,
    TARGETS_SCHEMA,
    VARIANTS_SCHEMA,
    SchemaError,
    load_score_sets,
    load_split,
    load_targets,
    load_variants,
    rank_normalize,
)

__all__ = [
    "AA20",
    "ConflictingSubstitutions",
    "DATASETS",
    "SCORE_SETS_SCHEMA",
    "SchemaError",
    "TARGETS_SCHEMA",
    "UnsupportedHGVS",
    "VARIANTS_SCHEMA",
    "WTCheck",
    "diff_sequences",
    "exclusion_dir",
    "format_mutant",
    "load_score_sets",
    "load_split",
    "load_targets",
    "load_variants",
    "mutate",
    "n_mut",
    "parse_hgvs_pro",
    "parse_mutant",
    "processed_dir",
    "rank_normalize",
    "raw_dir",
    "scratch",
    "splits_dir",
    "suggest_offsets",
    "verify_wt",
]
