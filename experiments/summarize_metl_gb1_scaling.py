"""
summarize_metl_gb1_scaling.py -- table for metl_gb1_scaling.sh.

    ~/anaconda3/envs/mutant_cache/bin/python experiments/summarize_metl_gb1_scaling.py

Test Spearman (METL supertest subsample) by train size and mutation depth,
median over seeds as the METL paper reports, plus the exact-backend control
delta where it was run.
"""
import glob
import json
import os
import re

import pandas as pd

OUT = "/home/andrew/mutant_cache_scratch/runs/metl_gb1_scaling"

rows = []
for f in sorted(glob.glob(f"{OUT}/*/results.json")):
    tag = os.path.basename(os.path.dirname(f))
    m = re.match(r"ds(\d+)_s(\d+)$", tag)
    if not m:
        continue
    n, seed = int(m.group(1)), int(m.group(2))
    r = json.load(open(f))
    mean = r["metrics"]["test"]["mean"]
    row = dict(ds=n, train=int(round(n * 0.9)), seed=seed)
    for k, v in mean.items():
        if k.startswith("spearman_depth_"):
            row[k.replace("spearman_depth_", "rho_")] = v
        elif k.startswith("control_spearman_depth_"):
            row["exact_" + k.replace("control_spearman_depth_", "rho_")] = v
    rows.append(row)

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no results yet")
df = df.sort_values(["ds", "seed"])
depths = [c for c in df.columns if c.startswith("rho_")]

med = df.groupby(["ds", "train"])[depths].median().reset_index()
cnt = df.groupby("ds").size().rename("seeds").reset_index()
med = med.merge(cnt, on="ds")
print("median test Spearman over seeds (METL supertest subsample, n=50,000)\n")
print(med.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if "exact_rho_all" in df.columns and df["exact_rho_all"].notna().any():
    d = df.dropna(subset=["exact_rho_all"]).copy()
    d["delta_all"] = d["exact_rho_all"] - d["rho_all"]
    print("\nsame weights scored through the exact full forward (seed 0):\n")
    print(d[["ds", "train", "rho_all", "exact_rho_all", "delta_all"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

df.to_csv("/home/andrew/mutant_cache/experiments/metl_gb1_scaling.csv", index=False)
print("\nwrote experiments/metl_gb1_scaling.csv")
