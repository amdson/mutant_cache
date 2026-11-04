# summarize_esm_msr.py -- TODO_ESM_MSR_EXPERIMENT.md §3-§4 summaries.
#   python experiments/summarize_esm_msr.py select   screen (seed 0): per-arm config by the fixed
#                                                    rule (lowest selected macro val double RMSE,
#                                                    then singles RMSE); prints SEL_S / SEL_A
#   python experiments/summarize_esm_msr.py val      every val run, all seeds
#   python experiments/summarize_esm_msr.py final    test: per seed and paired per-protein
#                                                    S vs A, S vs S-add, protein bootstrap
# Writes experiments/esm_msr_<mode>.csv (+ esm_msr_final_paired.csv). Macro over proteins is the
# primary summary; the measured-singles sum is a privileged diagnostic, not a baseline.
import glob, json, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/andrew/mutant_cache")
from mutant_cache.eval import megascale as M

R = "/home/andrew/mutant_cache_scratch/runs"
mode = sys.argv[1] if len(sys.argv) > 1 else "select"
TAG = re.compile(r"(joint|additive)_alr([0-9e.-]+)_hlr([0-9e.-]+)_s(\d+)$")


def runs(stage):
    rows = []
    for f in sorted(glob.glob(f"{R}/{stage}/*/results.json")):
        d = f.rsplit("/", 1)[0]
        m = TAG.match(d.rsplit("/", 1)[1])
        if not m:
            continue
        r = json.load(open(f)); e = r["extra"]
        s = pd.DataFrame(e["val_summary"])
        row = {"dir": d, "arm": {"joint": "S", "additive": "A"}[m[1]], "adapter_lr": m[2],
               "head_lr": m[3], "seed": int(m[4]),
               "sel_exposure": e["selection"].get("exposure"),
               "sel_step": e["selection"].get("step"),
               "steps": e["timing"]["total_steps"], "cold_total_s": e["timing"]["cold_total_s"],
               "train_rows_per_s": e["timing"]["steady_train_rows_per_s"],
               "peak_alloc_gb": e["memory"].get("peak_allocated_gb"),
               "peak_reserved_gb": e["memory"].get("peak_reserved_gb"),
               "ref_cache_mb": e["memory"].get("ref_cache_mb"),
               "active_params": e["active_params"], "exposure_by_n_mut": e["exposure_by_n_mut"]}
        for dep in ("single", "double", "epistasis"):
            for met in M.METRICS:
                row[f"val_{dep}_{met}"] = M.macro(s, dep, met)
        rr = pd.DataFrame(e.get("val_retrieval", []))
        if len(rr):
            row["val_ndcg96_all"] = float(rr.set_index("subset").loc["all", "ndcg_macro"])
        h = pd.DataFrame(r["history"])
        row["still_improving_at_end"] = bool(len(h) > 2 and h["val_double_rmse"].iloc[-1]
                                             < h["val_double_rmse"].iloc[-2])
        rows.append(row)
    return pd.DataFrame(rows)


show = ["arm", "adapter_lr", "head_lr", "seed", "sel_exposure", "val_double_rmse",
        "val_single_rmse", "val_double_spearman", "val_single_spearman", "val_epistasis_rmse",
        "val_epistasis_spearman", "val_ndcg96_all", "still_improving_at_end", "cold_total_s",
        "peak_alloc_gb"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

if mode in ("select", "val"):
    df = runs(__import__("os").environ.get("SCREEN", "esm_msr_screen"))  # SCREEN=esm_msr_screen_es for the early-stopping screen
    if mode == "select":
        df = df[df.seed == 0]
    df.to_csv(f"experiments/esm_msr_{mode}.csv", index=False)
    print(df[[c for c in show if c in df]].sort_values(["arm", "seed", "val_double_rmse"])
          .round(4).to_string(index=False))
    if mode == "select":
        for arm, g in df.groupby("arm"):
            g = g.assign(k1=g.val_double_rmse.fillna(np.inf), k2=g.val_single_rmse.fillna(np.inf))
            b = g.sort_values(["k1", "k2"]).iloc[0]
            print(f"SEL_{arm}=\"{b.adapter_lr} {b.head_lr}\"   (val double RMSE {b.val_double_rmse:.4f},"
                  f" single {b.val_single_rmse:.4f}; {len(g)} configs)")
        if df.still_improving_at_end.any():
            print("NOTE: some runs still improving at the final exposure:",
                  df.loc[df.still_improving_at_end, ["arm", "adapter_lr", "head_lr"]]
                  .to_dict("records"), "-> TODO §3: consider the 20-exposure extension for both arms")

if mode == "final":
    rows, pp = [], {}
    for f in sorted(glob.glob(f"{R}/esm_msr_final/*/predictions_test.csv")):
        d = f.rsplit("/", 1)[0]
        m = TAG.match(d.rsplit("/", 1)[1])
        arm, seed = {"joint": "S", "additive": "A"}[m[1]], int(m[4])
        pt = pd.read_csv(f)
        preds = {arm: "pred"} | ({"S-add": "pred_add"} if arm == "S" else {})
        for name, col in preds.items():
            ev = M.evaluate_frame(pt.assign(pred=pt[col]), "pred")
            pp[(name, seed)] = ev["per_protein"]
            row = {"model": name, "seed": seed}
            for dep in ("single", "double", "epistasis"):
                for met in M.METRICS:
                    row[f"test_{dep}_{met}"] = M.macro(ev["summary"], dep, met)
                    row[f"test_{dep}_{met}_median"] = float(
                        ev["summary"].query("depth == @dep and metric == @met")["median"].iloc[0]) \
                        if len(ev["summary"].query("depth == @dep and metric == @met")) else np.nan
            for _, r in ev["pooled"].iterrows():
                row[f"pooled_{r.depth}_spearman"] = r.spearman
            rs = ev["retrieval_summary"].set_index("subset")
            for sub in rs.index:
                row[f"ndcg96_{sub}"] = rs.loc[sub, "ndcg_macro"]
                row[f"precision96_{sub}"] = rs.loc[sub, "precision_macro"]
                row[f"ndcg96_{sub}_n_proteins"] = rs.loc[sub, "n_proteins"]
            row["n_epistasis_groups"] = len(ev["epistasis_table"])
            rows.append(row)
        if arm == "S" and seed == 0:
            # privileged diagnostic: sum of MEASURED held-out singles (uses test labels)
            tab = M.epistasis_table(pt, "pred")
            meas = tab.assign(pred_meas=tab["ddg"] - tab["eps_obs"])
            ms = [{"target_id": p, **M._block(g["ddg"].to_numpy(), g["pred_meas"].to_numpy(), M.MIN_N)}
                  for p, g in meas.groupby("target_id")]
            ms = pd.DataFrame(ms)
            print("PRIVILEGED measured-singles sum on complete doubles (not deployable, not a "
                  f"ceiling): macro RMSE {ms.rmse.mean():.4f}, Spearman {ms.spearman.mean():.4f}, "
                  f"{len(meas)} doubles / {len(ms)} proteins")
    df = pd.DataFrame(rows).sort_values(["model", "seed"])
    df.to_csv("experiments/esm_msr_final.csv", index=False)
    print(df.round(4).to_string(index=False))
    paired = []
    for a, b in (("S", "A"), ("S", "S-add")):
        for seed in sorted({s for (_, s) in pp}):
            if (a, seed) not in pp or (b, seed) not in pp:
                continue
            for dep, met in (("double", "rmse"), ("double", "spearman"), ("single", "rmse"),
                             ("single", "spearman"), ("epistasis", "rmse"), ("double", "mae")):
                r = M.paired_protein_bootstrap(pp[(a, seed)], pp[(b, seed)], met, dep, seed=seed)
                paired.append({"a": a, "b": b, "seed": seed, "depth": dep, "metric": met, **r})
    pr = pd.DataFrame(paired)
    pr.to_csv("experiments/esm_msr_final_paired.csv", index=False)
    print(pr.round(4).to_string(index=False))
