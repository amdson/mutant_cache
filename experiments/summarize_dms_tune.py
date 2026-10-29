# summarize_dms_tune.py -- median DEV Spearman over reps, config x train size,
# from runs/dms_tune/*/results.json. ESM-2 35M FT row is METL Fig. 2 (TEST set,
# same distribution, different rows -- indicative only).
import json, glob, re
import pandas as pd

OUT = "/home/andrew/mutant_cache_scratch/runs/dms_tune"
rows = []
for f in glob.glob(f"{OUT}/*_ds*_r*/results.json"):
    m = re.search(r"/(\w+?)_ds(\d+)_r(\d+)/", f)
    r = json.load(open(f))
    rows.append({"cfg": m[1], "n": int(m[2]), "rep": int(m[3]),
                 "dev_rho": r["metrics"]["test"]["mean"]["spearman_depth_all"]})
df = pd.DataFrame(rows)
df.to_csv("experiments/dms_tune_runs.csv", index=False)
t = df.pivot_table(index="cfg", columns="n", values="dev_rho", aggfunc="median")
reps = df.pivot_table(index="cfg", columns="n", values="dev_rho", aggfunc="size")
t.loc["esm2_35m_ft (test, Fig.2)"] = {80: .636, 320: .836, 1280: .925}
t.to_csv("experiments/dms_tune.csv")
print(t.round(3).to_string()); print("\nreps:\n" + reps.to_string())
