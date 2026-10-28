"""
build_accuracy_splits.py -- within-assay random splits (70/10/20 per score set)
for the cache-vs-exact accuracy test. Written to scratch splits/.

    ~/anaconda3/envs/mutant_cache/bin/python experiments/build_accuracy_splits.py
"""
import numpy as np
import polars as pl

D = "/home/andrew/mutant_cache_scratch/processed/"
TSUB = ["tsuboyama_3DKM.pdb", "tsuboyama_2MXD.pdb", "tsuboyama_2KGT.pdb", "tsuboyama_4G3O.pdb"]


def split(v, seed=0):
    # one label per unique (score_set_id, mutant): replicate rows must not straddle splits
    v = v.select("score_set_id", "mutant").unique(maintain_order=True)
    rng = np.random.default_rng(seed)
    u = rng.random(v.height)
    lab = np.where(u < 0.7, "train", np.where(u < 0.8, "val", "test"))
    return v.with_columns(split=pl.Series(lab))


v = pl.read_parquet(D + "tsuboyama/variants.parquet").filter(pl.col("score_set_id").is_in(TSUB))
s = split(v)
s.write_parquet(D + "splits/acc_tsuboyama4_random.parquet")
print("tsuboyama", s.group_by("split").len().sort("split").rows())

v = pl.read_parquet(D + "mavedb/variants.parquet").filter(
    pl.col("score_set_id") == "urn:mavedb:00000003-b-1")
s = split(v)
s.write_parquet(D + "splits/acc_brca1_random.parquet")
print("brca1", s.group_by("split").len().sort("split").rows())
