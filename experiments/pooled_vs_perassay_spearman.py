"""Pooled vs per-assay Spearman on a harness run's predictions.csv.

ThermoMPNN accumulates ONE torchmetrics SpearmanCorrCoef over every mutation in
every test protein and computes it once (analysis/thermompnn_benchmarking.py:
get_metrics() + metric.update() per mutation). So their published 0.73 is a
POOLED, variant-level correlation on ddG in kcal/mol.

Our eval/depth.py deliberately does the opposite: the `ALL` row is the *mean of
per-score-set* Spearmans, on the grounds that pooling "would be dominated by
between-assay scale".

Both are defensible; they are not the same number. This prints both so a
comparison against ThermoMPNN is made on their convention, and so the size of
the convention gap is visible.

Usage: python pooled_vs_perassay_spearman.py <run_dir>
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

run_dir = sys.argv[1]
df = pd.read_csv(f"{run_dir}/predictions.csv")

print(f"{run_dir}")
print(f"rows {len(df)}  score sets {df.score_set_id.nunique()}  "
      f"targets {df.target_id.nunique()}")

for depth_label, sub in (("singles", df[df.n_mut == 1]),
                         ("doubles", df[df.n_mut == 2]),
                         ("all muts", df[df.n_mut >= 1])):
    if not len(sub):
        continue
    pooled = spearmanr(sub["pred"], sub["score"]).statistic
    per = (sub.groupby("score_set_id")
              .apply(lambda g: spearmanr(g["pred"], g["score"]).statistic
                     if len(g) > 2 and g["score"].nunique() > 1 else np.nan,
                     include_groups=False)
              .dropna())
    print(f"\n{depth_label}: n={len(sub)}  sets={sub.score_set_id.nunique()}")
    print(f"  pooled   (ThermoMPNN convention) : {pooled:.4f}")
    print(f"  per-assay mean (our ALL row)     : {per.mean():.4f}  "
          f"(median {per.median():.4f}, n_sets={len(per)})")
    print(f"  convention gap                   : {pooled - per.mean():+.4f}")
