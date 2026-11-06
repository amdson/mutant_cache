# summarize_e4_transfer.py -- TODO.md E4: H0 (mean) vs chosen head, V_1e-3, on avGFP-320,
# GB1-320, GB1-1280, replicates 0-2; paired on the same genotype splits. DEVELOPMENT holdouts.
# Reads experiments/pilot_runs.csv (run summarize_pilot.py first). GB1 dev is ~all doubles, so
# only d2 is defined there.
import pandas as pd

HEAD = "sum_mean_count"
df = pd.read_csv("experiments/pilot_runs.csv")
df = df[df.stage == "e4"]
cols = ["dev_rho", "dev_rho_d1", "dev_rho_d2", "dev_rho_d3", "dev_rho_d4", "dev_rho_d5+"]
with pd.option_context("display.width", 250, "display.max_columns", 40):
    print(df[["task", "budget", "head", "rep", "n_train", "n_val", "n_dev"] + cols +
             ["best_step", "best_val_loss", "head_params", "cold_total_s", "peak_alloc_gb"]]
          .sort_values(["task", "budget", "head", "rep"]).round(4).to_string(index=False))
    for (t, n), g in df.groupby(["task", "budget"]):
        a = g[g["head"] == HEAD].set_index("rep")[cols]
        b = g[g["head"] == "mean"].set_index("rep")[cols]
        d = (a - b).dropna(how="all")
        print(f"\n== {t} ds{n}: {HEAD} - mean (paired) ==")
        print(d.round(4).to_string())
        print("mean", d.mean().round(4).to_dict(), "wins", int((d.dev_rho > 0).sum()), "/", len(d))
