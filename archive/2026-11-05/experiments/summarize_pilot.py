# summarize_pilot.py -- TODO.md 2026-11-01 implementation pilot (E2-E4). One row per completed
# run under runs/pilot_e*/ -> experiments/pilot_runs.csv. All Spearman here is on the
# DEVELOPMENT holdout (stored as split "test" in *_tune_* manifests), never a supertest.
# Depth bins with n < 20 or constant labels are reported as NaN (undefined).
import glob, json, re
import numpy as np
import pandas as pd

R = "/home/andrew/mutant_cache_scratch/runs"
MIN_N = 20
rows = []
for f in sorted(glob.glob(f"{R}/pilot_e*/*/results.json")):
    d = f.rsplit("/", 1)[0]
    stage, tag = d.split("/")[-2], d.split("/")[-1]
    m = re.match(r"(avgfp|gb1)_ds(\d+)_(.+)_r(\d+)$", tag)
    r = json.load(open(f)); e = r["extra"]
    dt = pd.read_csv(f"{d}/depth_spearman.csv")
    dt = dt[dt.score_set_id == "ALL"].set_index("depth")
    hist = [h for h in r["history"] if h.get("tag") != "init"]
    best_step = e["best"].get("step")
    at_best = next((h for h in hist if h["step"] == best_step), {})
    vs = [(h["step"], h.get("spearman_depth_all", np.nan)) for h in hist]
    vmax = max(vs, key=lambda x: -np.inf if x[1] != x[1] else x[1]) if vs else (None, np.nan)
    row = {"stage": stage.replace("pilot_", ""), "task": m[1], "budget": int(m[2]), "cfg": m[3],
           "rep": int(m[4]), "head": e.get("head"),
           "dev_rho": float(dt.loc["all", "spearman"]),
           "n_train": e["protocol"]["n_train_unique"], "n_val": e["protocol"]["n_val_unique"],
           "n_dev": e["protocol"]["n_eval_unique"],
           "best_step": best_step, "best_val_loss": e["best"].get("val_loss"),
           "val_rho_at_best": at_best.get("spearman_depth_all"),
           "train_loss_at_best": at_best.get("train_loss"),
           "val_rho_max": vmax[1], "val_rho_max_step": vmax[0],
           "final_train_loss": hist[-1]["train_loss"] if hist else np.nan,
           "final_val_loss": hist[-1].get("val_loss") if hist else np.nan,
           "steps": e["timing"].get("steps"), "cold_total_s": e["timing"].get("cold_total_s"),
           "train_s": e["timing"].get("train_s"), "train_eval_s": e["timing"].get("train_eval_s"),
           "final_eval_s": e["timing"].get("final_eval_s"),
           "train_variants_per_s": e["timing"].get("steady_train_variants_per_s"),
           "peak_alloc_gb": e["memory"].get("peak_allocated_gb"),
           "peak_reserved_gb": e["memory"].get("peak_reserved_gb"),
           "ref_cache_mb": e["memory"].get("ref_cache_mb"),
           "head_params": e.get("active_head_params"),
           "trainable_params": sum(g["n_params"] for g in e["timing"].get("optimizer_groups", [])),
           "split_sha1": e["protocol"]["split_sha1"], "git_sha": json.load(open(f"{d}/run.json"))["git_sha"]}
    for b in ("1", "2", "3", "4", "5+"):
        ok = b in dt.index and dt.loc[b, "n"] >= MIN_N
        row[f"dev_rho_d{b}"] = float(dt.loc[b, "spearman"]) if ok else np.nan
        row[f"n_d{b}"] = int(dt.loc[b, "n"]) if b in dt.index else 0
    # update diagnostics at the selected step
    for k, v in at_best.items():
        if k.startswith(("rel_update_from_init_", "grad_norm_")):
            row[k] = v
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv("experiments/pilot_runs.csv", index=False)
show = ["stage", "task", "budget", "cfg", "rep", "dev_rho", "dev_rho_d1", "dev_rho_d2", "dev_rho_d3",
        "dev_rho_d4", "dev_rho_d5+", "best_step", "best_val_loss", "val_rho_at_best", "val_rho_max",
        "val_rho_max_step", "train_loss_at_best", "cold_total_s", "peak_alloc_gb", "trainable_params"]
with pd.option_context("display.width", 250, "display.max_columns", 40):
    print(df[show].round(4).to_string(index=False))
