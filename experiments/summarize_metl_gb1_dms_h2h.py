# summarize_metl_gb1_dms_h2h.py -- median-over-replicates test Spearman per train
# size from runs/metl_gb1_dms_h2h/*/results.json (model + baselines).
import json, glob, re
import pandas as pd

OUT = "/home/andrew/mutant_cache_scratch/runs/metl_gb1_dms_h2h"

# METL paper (Gelman et al. 2025 Nat Methods) Fig. 2 GB1 panel, median test
# Spearman over replicates. No tabulated source data exists; these were read
# from the vector paths of the main-text PDF (page 4) calibrated to the axis
# ticks, ~±0.001. Cite as "read from Fig. 2", not as published numbers.
# Flat lines: Rosetta 0.336, EVE 0.306, RaSP 0.535.
METL = pd.DataFrame({
    "n":       [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480],
    "metl_l":  [.438, .574, .668, .767, .817, .858, .888, .921, .946, .957, .965, .970],
    "metl_g":  [.316, .457, .543, .667, .736, .828, .878, .914, .938, .950, .961, .967],
    "esm2_ft": [.267, .368, .468, .636, .729, .836, .894, .925, .947, .959, .966, .972],
    "linear":  [.217, .315, .467, .576, .702, .789, .844, .883, .921, .942, .953, .959],
    "pnpt":    [.127, .174, .217, .367, .583, .752, .853, .904, .938, .953, .964, .969],
}).set_index("n")
rows = []
for f in glob.glob(f"{OUT}/ds*_r*/results.json"):
    m = re.search(r"ds(\d+)_r(\d+)", f)
    r = json.load(open(f))
    row = {"n": int(m[1]), "rep": int(m[2]),
           "ours": r["metrics"]["test"]["mean"]["spearman_depth_all"],
           "best_step": r["extra"]["best"].get("step")}
    for k, v in r["extra"]["baselines"].items():
        row[k] = v.get("spearman_depth_all")
    rows.append(row)
df = pd.DataFrame(rows).sort_values(["n", "rep"])
df.to_csv("experiments/metl_gb1_dms_h2h_runs.csv", index=False)
cols = [c for c in df.columns if c not in ("n", "rep", "best_step")]
g = df.groupby("n")
summ = g[cols].median().add_prefix("median_")
summ.insert(0, "reps", g.size())
summ = summ.join(METL, how="left")
summ["ours_minus_metl_l"] = summ["median_ours"] - summ["metl_l"]
summ.to_csv("experiments/metl_gb1_dms_h2h.csv")
print(summ.round(3).to_string())
