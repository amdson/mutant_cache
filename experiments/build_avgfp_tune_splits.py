# build_avgfp_tune_splits.py -- avGFP analogue of build_dms_tune_splits.py (GB1), for the
# TODO.md 2026-11-01 queue (E0). Same train/val as METL's resampled reps; "test" is a
# fixed 5k DEVELOPMENT holdout drawn from genotypes outside METL's supertest AND outside
# the train/val rows of every replicate/budget in the queue (ds320, ds1280; reps 0-2).
# Not filtered by outcome. Seed 0. The harness stores it as split "test"; report it as DEV.
import numpy as np
import polars as pl
from mutant_cache.data.paths import splits_dir

SIZES, REPS, NDEV, SEED = (320, 1280), (0, 1, 2), 5000, 0
SSID = "metl_avgfp_dms"
v = pl.read_parquet("/home/andrew/mutant_cache_scratch/processed/metl_avgfp_dms/variants.parquet",
                    columns=["mutant"])
tabs = {(n, r): pl.read_parquet(splits_dir() / f"{SSID}_ds{n}_r{r}.parquet")
        for n in SIZES for r in REPS}
used = set()
for t in tabs.values():
    used |= set(t["mutant"].to_list())          # train + val + supertest of every rep
pool = np.array(sorted(set(v["mutant"].to_list()) - used), dtype=object)
ndev = min(NDEV, len(pool))
dev = np.random.default_rng(SEED).choice(pool, ndev, replace=False).tolist()
for (n, r), t in tabs.items():
    tv = t.filter(pl.col("split") != "test")
    out = pl.concat([tv, pl.DataFrame({"score_set_id": SSID, "mutant": dev, "split": "test"})])
    assert out["mutant"].n_unique() == len(out)
    out.write_parquet(splits_dir() / f"{SSID}_tune_ds{n}_r{r}.parquet")
print(f"pool {len(pool):,}, dev {len(dev):,}, wrote {len(tabs)} tune splits")
