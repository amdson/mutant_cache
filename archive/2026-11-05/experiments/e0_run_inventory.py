# e0_run_inventory.py -- TODO.md 2026-11-01, E0. One row per (experiment, config, task,
# budget) over every completed results.json under runs/ that the pilot could reuse,
# with replicate count, median AND mean, and the protocol fields that decide whether a
# result is reusable (split family, precision, loss, lrs, LoRA, epochs, selection).
# Metric label: TEST(supertest) for METL h2h runs, DEV for *_tune_* splits.
import glob, json, re
import numpy as np
import pandas as pd

R = "/home/andrew/mutant_cache_scratch/runs"
EXPS = ["metl_avgfp_dms_h2h_minimal", "metl_gb1_dms_h2h_minimal", "metl_gb1_dms_h2h",
        "dms_tune", "dms_fullft_diag"]
rows = []
for e in EXPS:
    for f in glob.glob(f"{R}/{e}/*/results.json"):
        run = f.split("/")[-2]
        m = re.match(r"(?:(\w+?)_)?ds(\d+)_r(\d+)$", run)
        if not m:
            continue
        r = json.load(open(f))
        try:
            hp = json.load(open(f.replace("results.json", "run.json")))["hyperparams"]
        except Exception:
            hp = {}
        split = hp.get("split") or ""
        metric = list(r["metrics"])[0]
        rows.append({"experiment": e, "cfg": m[1] or "-", "dataset": hp.get("dataset"),
                     "n": int(m[2]), "rep": int(m[3]),
                     "label": "DEV" if "_tune_" in split else
                              ("TEST(supertest)" if metric == "test" else metric),
                     "rho": r["metrics"][metric]["mean"].get("spearman_depth_all"),
                     "best_step": r["extra"].get("best", {}).get("step"),
                     "precision": hp.get("precision"), "loss": hp.get("loss"),
                     "target_col": hp.get("target_col"), "lr": hp.get("lr"),
                     "train_backbone": hp.get("train_backbone"),
                     "backbone_lr": hp.get("backbone_lr") if hp.get("train_backbone") else None,
                     "lora_r": hp.get("lora_r"), "shared_lora_r": hp.get("shared_lora_r"),
                     "lora_lr": hp.get("lora_lr"), "minimal": hp.get("minimal"),
                     "epochs": hp.get("epochs"), "batch_size": hp.get("batch_size"),
                     "select_on_val": hp.get("select_on_val")})
df = pd.DataFrame(rows).sort_values(["experiment", "cfg", "n", "rep"])
keys = ["experiment", "cfg", "dataset", "label", "n"]
proto = ["precision", "loss", "target_col", "lr", "train_backbone", "backbone_lr", "lora_r",
         "shared_lora_r", "lora_lr", "minimal", "batch_size", "select_on_val"]
g = df.groupby(keys, dropna=False)
inv = g["rho"].agg(reps="size", median="median", mean="mean", sd="std", min="min", max="max")
inv["best_step_median"] = g["best_step"].median()
inv["epochs"] = g["epochs"].agg(lambda s: "/".join(map(str, sorted(set(s)))))
for c in proto:   # one value per group, or "MIXED" if replicates disagree
    inv[c] = g[c].agg(lambda s: s.iloc[0] if s.fillna("NA").astype(str).nunique() == 1 else "MIXED")
inv = inv.reset_index()
inv.to_csv("experiments/e0_run_inventory.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 40):
    print(inv[inv["n"].isin([320, 1280])].round(4).to_string(index=False))
