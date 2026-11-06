# summarize_e3_heads.py -- TODO.md E3: paired head-minus-mean (H0) differences on avGFP ds1280
# DEVELOPMENT holdout, V_1e-3 reference, replicates 0-2. H0 = pilot_e2 V_1e-3 runs (same code
# diff hash, same protocol). Reads experiments/pilot_runs.csv (run summarize_pilot.py first).
# Also writes the validation curves to experiments/e3_val_curves.csv.
import json
import pandas as pd

R = "/home/andrew/mutant_cache_scratch/runs"
REF = "V_1e-3"
df = pd.read_csv("experiments/pilot_runs.csv")
df = df[(df.task == "avgfp") & (df.budget == 1280)]
h0 = df[(df.stage == "e2") & (df.cfg == REF)].copy(); h0["head"] = "mean"
hx = df[(df.stage == "e3") & df.cfg.str.startswith(REF + "_")].copy()
runs = pd.concat([h0, hx])
cols = ["dev_rho", "dev_rho_d1", "dev_rho_d2", "dev_rho_d3", "dev_rho_d4", "dev_rho_d5+"]
base = h0.set_index("rep")[cols]

with pd.option_context("display.width", 250, "display.max_columns", 40):
    print("== per run ==")
    print(runs[["head", "rep"] + cols + ["best_step", "best_val_loss", "val_rho_at_best", "head_params",
                                          "trainable_params", "cold_total_s", "train_variants_per_s",
                                          "peak_alloc_gb", "peak_reserved_gb"]]
          .sort_values(["head", "rep"]).round(4).to_string(index=False))
    print("\n== paired head - mean, per replicate ==")
    out = []
    for h, g in hx.groupby("head"):
        d = g.set_index("rep")[cols] - base
        d["head"] = h
        out.append(d.reset_index())
    pdiff = pd.concat(out)
    print(pdiff.round(4).to_string(index=False))
    print("\n== paired head - mean, mean over replicates (wins/3 overall) ==")
    agg = pdiff.groupby("head")[cols].mean()
    agg["wins_overall"] = pdiff.groupby("head")["dev_rho"].apply(lambda s: int((s > 0).sum()))
    print(agg.round(4).to_string())
    print("\n== means per head ==")
    print(runs.groupby("head")[cols + ["best_step", "best_val_loss", "head_params", "cold_total_s",
                                      "peak_alloc_gb"]].mean().round(4).to_string())

curves = []
for _, r in runs.iterrows():
    stage = "pilot_e2" if r.stage == "e2" else "pilot_e3"
    tag = f"avgfp_ds1280_{r.cfg}_r{r.rep}"
    for h in json.load(open(f"{R}/{stage}/{tag}/results.json"))["history"]:
        if h.get("tag") == "init":
            continue
        curves.append({"head": r["head"], "rep": r.rep, "step": h["step"], "train_loss": h.get("train_loss"),
                       "val_loss": h.get("val_loss"), "val_rho": h.get("spearman_depth_all")})
cv = pd.DataFrame(curves)
cv.to_csv("experiments/e3_val_curves.csv", index=False)
print("\n== val loss / val rho by step (mean over reps) ==")
with pd.option_context("display.width", 250):
    print(cv.groupby(["step", "head"])[["val_loss", "val_rho", "train_loss"]].mean().unstack("head")
          .round(3).to_string())
