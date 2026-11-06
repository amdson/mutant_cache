"""Build a protein-level split table from ThermoMPNN's released Megascale split.

ThermoMPNN (PNAS 2024) ships dataset_splits/mega_splits.pkl: domain-level
train/val/test lists (239/31/28) clustered with mmseqs2 at 25% identity, plus
5 CV folds. Their names are exactly our Tsuboyama target_ids (298/298 match).

Our targets.parquet has 479 rows because Tsuboyama measured some domains in a
mutated background (`1GB4.pdb_F53D`). Those inherit their base domain's split,
or we leak the same domain across train and test. Targets whose base domain is
not in ThermoMPNN's 298 are dropped -- they are domains ThermoMPNN excluded,
and keeping them would make the training pool incomparable.

Downloaded from:
  https://raw.githubusercontent.com/Kuhlman-Lab/ThermoMPNN/main/dataset_splits/mega_splits.pkl
"""
import pickle
import re
from collections import Counter

import polars as pl

PKL = "/home/andrew/mutant_cache_scratch/raw/thermompnn/mega_splits.pkl"
TARGETS = "/home/andrew/mutant_cache_scratch/processed/tsuboyama/targets.parquet"
OUT = "/home/andrew/mutant_cache_scratch/processed/splits/thermompnn_mega.parquet"

sp = pickle.load(open(PKL, "rb"))
label = {}
for k in ("train", "val", "test"):
    names = sp[k] if isinstance(sp[k], list) else sp[k].tolist()
    for n in names:
        label[n] = k
print(f"ThermoMPNN domains: {Counter(label.values())}  total {len(label)}")


def base_domain(target_id: str) -> str:
    """`1GB4.pdb_F53D` -> `1GB4.pdb`; leaves plain names untouched."""
    m = re.match(r"^(.*?\.pdb)(_.+)?$", target_id)
    return m.group(1) if m else target_id


targets = pl.read_parquet(TARGETS)
rows = []
dropped = []
for tid in targets["target_id"].to_list():
    b = base_domain(tid)
    if b in label:
        rows.append({"target_id": tid, "split": label[b]})
    else:
        dropped.append(tid)

out = pl.DataFrame(rows)
print(f"targets mapped: {out.height}  dropped (not in ThermoMPNN's 298): {len(dropped)}")
print(out.group_by("split").len().sort("split"))

# ThermoMPNN's Dataset Curation drops modified-wild-type backgrounds entirely
# ("272,712 mutations across 298 proteins" after that drop). We inherit their
# split for those targets above, which is right for leakage but leaves 10 extra
# train and 2 extra test domains relative to their curation. The strict table
# removes them, and then test is 28 domains / 28,172 singles -- the exact count
# their own loader scores (mega_test.csv minus its 140 wild-type rows).
is_mod_wt = pl.col("target_id").str.contains(r"^.*?\.pdb_.+$")
strict = out.filter(~is_mod_wt)
print("\nstrict (ThermoMPNN curation: modified-WT backgrounds dropped)")
print(strict.group_by("split").len().sort("split"))
strict.write_parquet(OUT.replace(".parquet", "_strict.parquet"))
print(f"wrote {OUT.replace('.parquet', '_strict.parquet')}")

# leak check: no base domain may appear in two splits
chk = out.with_columns(
    pl.col("target_id").map_elements(base_domain, return_dtype=pl.String).alias("base")
)
bad = (chk.group_by("base").agg(pl.col("split").n_unique().alias("n"))
          .filter(pl.col("n") > 1))
assert bad.height == 0, f"base domain in multiple splits:\n{bad}"
print("leak check: no base domain spans splits")

out.write_parquet(OUT)
print(f"wrote {OUT}")
