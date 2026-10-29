# build_dms_tune_splits.py -- tuning splits for the GB1 DMS settings sweep.
# Same train/val as METL's resampled reps, but "test" is a fixed 5k DEV set drawn
# from rows outside METL's supertest and outside every train/val used here, so
# configs are compared without touching the real test set.
import numpy as np
import polars as pl
from mutant_cache.data.paths import splits_dir

SIZES, REPS, NDEV = (80, 320, 1280), (0, 1, 2), 5000
SSID = "metl_gb1_dms"
v = pl.read_parquet("/home/andrew/mutant_cache_scratch/processed/metl_gb1_dms/variants.parquet",
                    columns=["mutant"])
tabs = {(n, r): pl.read_parquet(splits_dir() / f"{SSID}_ds{n}_r{r}.parquet")
        for n in SIZES for r in REPS}
used = set()
for t in tabs.values():
    used |= set(t["mutant"].to_list())          # train + val + test of every rep
pool = np.array(sorted(set(v["mutant"].to_list()) - used), dtype=object)
dev = np.random.default_rng(0).choice(pool, NDEV, replace=False).tolist()
for (n, r), t in tabs.items():
    tv = t.filter(pl.col("split") != "test")
    out = pl.concat([tv, pl.DataFrame({"score_set_id": SSID, "mutant": dev, "split": "test"})])
    assert out["mutant"].n_unique() == len(out)
    out.write_parquet(splits_dir() / f"{SSID}_tune_ds{n}_r{r}.parquet")
print(f"pool {len(pool):,}, dev {len(dev):,}, wrote {len(tabs)} tune splits")
