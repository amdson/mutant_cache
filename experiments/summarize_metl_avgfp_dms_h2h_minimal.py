# summarize_metl_avgfp_dms_h2h_minimal.py -- as summarize_metl_gb1_dms_h2h_minimal.py, for avGFP.
# Per-run test Spearman from runs/metl_avgfp_dms_h2h_minimal/*/results.json (model + baselines).
import json, glob, re
import pandas as pd

OUT = "/home/andrew/mutant_cache_scratch/runs/metl_avgfp_dms_h2h_minimal"

# METL paper (Gelman et al. 2025 Nat Methods) Fig. 2 GFP panel, median test
# Spearman over replicates, read from the vector paths of the main-text PDF
# (page 4) calibrated to the axis ticks, same method as the GB1 panel (which it
# reproduces to 0.001). ~±0.001. Cite as "read from Fig. 2", not as published numbers.
# Flat lines: Rosetta 0.664, EVE 0.655, RaSP 0.681.
METL = pd.DataFrame({
    "n":       [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480],
    "metl_l":  [.637, .683, .711, .722, .750, .773, .793, .824, .843, .860, .869, .876],
    "metl_g":  [.375, .470, .557, .615, .668, .720, .639, .787, .805, .823, .858, .865],
    "esm2_ft": [.190, .295, .378, .501, .559, .655, .740, .806, .838, .856, .861, .871],
    "linear":  [.095, .148, .217, .297, .414, .542, .662, .733, .789, .813, .838, .854],
    "lin_eve": [.639, .651, .652, .657, .667, .694, .721, .753, .793, .815, .838, .854],
    "pnpt":    [.242, .394, .477, .515, .595, .641, .708, .760, .814, .845, .857, .865],
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
df.to_csv("experiments/metl_avgfp_dms_h2h_minimal_runs.csv", index=False)
cols = [c for c in df.columns if c not in ("n", "rep", "best_step")]
g = df.groupby("n")
summ = g[cols].median().add_prefix("median_")
summ.insert(0, "reps", g.size())
summ = summ.join(METL, how="left")
summ["ours_minus_metl_l"] = summ["median_ours"] - summ["metl_l"]
summ.to_csv("experiments/metl_avgfp_dms_h2h_minimal.csv")
print(summ.round(3).to_string())
