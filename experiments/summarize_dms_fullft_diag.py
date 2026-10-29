# summarize_dms_fullft_diag.py -- dev Spearman per knob (reps 0-1) + best val loss.
import json, glob, re
import pandas as pd

OUT = "/home/andrew/mutant_cache_scratch/runs/dms_fullft_diag"
rows = []
for f in glob.glob(f"{OUT}/*_r*/results.json"):
    m = re.search(r"/([\w.-]+)_r(\d+)/", f)
    r = json.load(open(f))
    rows.append({"cfg": m[1], "rep": int(m[2]),
                 "dev_rho": r["metrics"]["test"]["mean"]["spearman_depth_all"],
                 "best_step": r["extra"]["best"].get("step")})
df = pd.DataFrame(rows)
df.to_csv("experiments/dms_fullft_diag_runs.csv", index=False)
t = df.pivot_table(index="cfg", columns="rep", values="dev_rho")
t["mean"] = t.mean(axis=1)
t["best_steps"] = df.groupby("cfg")["best_step"].apply(lambda s: list(s))
print(t.sort_values("mean", ascending=False).round(3).to_string())
print("\nreference: ESM-2 35M FT ds1280 = 0.925 (METL test set; dev is same distribution)")
