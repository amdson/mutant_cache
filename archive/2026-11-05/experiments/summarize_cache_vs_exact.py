"""
summarize_cache_vs_exact.py -- table for cache_vs_exact_accuracy.sh.

    ~/anaconda3/envs/mutant_cache/bin/python experiments/summarize_cache_vs_exact.py

For each dataset x depth: Spearman (mean ± sd over seeds) of
  cache-trained scored by cache | cache-trained scored by exact |
  exact-trained scored by exact | exact-trained scored by cache
plus the two paired deltas that answer the questions:
  same weights:  exact - cache scoring of the cache-trained model
  train-through: exact-trained(exact) - cache-trained(cache)
"""
import glob
import json
import os

import numpy as np
import pandas as pd

OUT = "/home/andrew/mutant_cache_scratch/runs/cache_vs_exact"

rows = []
for f in sorted(glob.glob(f"{OUT}/*/results.json")):
    tag = os.path.basename(os.path.dirname(f))
    ds, tb, seed = tag.split("_")
    tb = tb.replace("train-", "")
    m = json.load(open(f))["metrics"]["test"]["mean"]
    other = "exact" if tb == "cache" else "cache"
    for k, v in m.items():
        if not k.startswith(("spearman_depth_", "control_spearman_depth_")):
            continue
        scored = other if k.startswith("control_") else tb
        depth = k.split("depth_")[1]
        rows.append(dict(dataset=ds, train=tb, scored=scored, seed=int(seed[1:]),
                         depth=depth, rho=v))
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no results yet")

cols = [("cache", "cache"), ("cache", "exact"), ("exact", "exact"), ("exact", "cache")]
out = []
for (ds, depth), g in df.groupby(["dataset", "depth"]):
    r = {"dataset": ds, "depth": depth}
    piv = g.pivot_table(index="seed", columns=["train", "scored"], values="rho")
    for c in cols:
        if c in piv:
            x = piv[c].dropna()
            r[f"{c[0]}-trained / {c[1]}"] = f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f} (n={len(x)})"
    if ("cache", "exact") in piv and ("cache", "cache") in piv:
        d = (piv[("cache", "exact")] - piv[("cache", "cache")]).dropna()
        r["same weights: exact - cache"] = f"{d.mean():+.3f} ± {d.std(ddof=1) if len(d) > 1 else 0:.3f}"
    if ("exact", "exact") in piv and ("cache", "cache") in piv:
        d = (piv[("exact", "exact")] - piv[("cache", "cache")]).dropna()
        r["train-through: exact - cache"] = f"{d.mean():+.3f} ± {d.std(ddof=1) if len(d) > 1 else 0:.3f}"
    out.append(r)
t = pd.DataFrame(out)
order = {"1": 0, "2": 1, "3": 2, "4": 3, "5+": 4, "all": 9}
t = t.sort_values(["dataset", "depth"], key=lambda s: s.map(order) if s.name == "depth" else s)
print(t.to_string(index=False))
t.to_csv("/home/andrew/mutant_cache/experiments/cache_vs_exact_accuracy.csv", index=False)
