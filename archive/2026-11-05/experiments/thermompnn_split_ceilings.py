"""Reliability ceilings on the ThermoMPNN-split Tsuboyama val/test sets.

HANDOFF.md 6.3: `reliability` is null for all 412 Tsuboyama score sets, but
Tsuboyama ships per-variant standard errors, so the attenuation ceiling is
computable. This matters right now for a specific reason: the mse_dg training
curve plateaus in Spearman around step 3000-4000 while the loss keeps falling.
Two readings --

  (a) objective misalignment: MSE keeps improving magnitude without improving
      rank, so a rank loss should go higher; or
  (b) saturation: we are at the noise ceiling and no loss can do better.

(b) is the boring explanation and has to be excluded first. If rho_max is near
0.70, there is nothing to find.

ceiling = sqrt(1 - E[se^2] / Var(score)), per eval/epsilon.py.
"""
import os

import numpy as np
import polars as pl

from mutant_cache.eval.epsilon import attenuation_ceiling

D = os.path.expanduser("~/mutant_cache_scratch/processed")
v = pl.read_parquet(f"{D}/tsuboyama/variants.parquet")

for split_name in ("thermompnn_mega_strict",):
    s = pl.read_parquet(f"{D}/splits/{split_name}.parquet")
    j = v.join(s, on="target_id", how="inner").filter(pl.col("n_mut") == 1)

    print(f"\n=== {split_name} ===")
    for part in ("val", "test"):
        sub = j.filter(pl.col("split") == part)
        if not sub.height:
            continue
        pdf = sub.to_pandas()

        pooled = attenuation_ceiling(pdf["score"].to_numpy(),
                                     pdf["score_se"].to_numpy())
        per = []
        for ssid, g in pdf.groupby("score_set_id"):
            c = attenuation_ceiling(g["score"].to_numpy(),
                                    g["score_se"].to_numpy())
            if np.isfinite(c):
                per.append(c)
        per = np.array(per)

        print(f"\n{part}: n={len(pdf)}  sets={pdf.score_set_id.nunique()}")
        print(f"  pooled ceiling              : {pooled:.4f}")
        print(f"  per-assay ceiling mean      : {per.mean():.4f} "
              f"(median {np.median(per):.4f}, min {per.min():.4f}, "
              f"max {per.max():.4f}, n={len(per)})")
        print(f"  mean se                     : {pdf['score_se'].mean():.4f}")
        print(f"  Var(score)                  : {pdf['score'].var():.4f}")
