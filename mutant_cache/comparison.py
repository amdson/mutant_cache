"""Evaluate matched Mutate Everything predictions and repeated cache fits.

python -m mutant_cache.comparison --theirs their.csv --ours seed0.csv seed1.csv seed2.csv --out results
An optional expected JSON is {"double": {"ndcg30": ...}}. It must contain only
verified, cited numbers for the same dataset. A reproduction gate is evaluated
on their complete predictions before joining our predictions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mutant_cache.eval.mutate_everything import (
    canonical_predictions, evaluate, join_predictions, paired_seed_bootstrap, reproduction_gate)


def benchmark_inference(model, batches, frame, *, repeats=3):
    """Synchronised scoring, including reference construction in the cold pass.

    Model loading and batch construction are excluded. Warm passes reuse the
    reference cache. Peak allocated memory includes resident model weights.
    """
    import torch
    from mutant_cache.megascale import score
    if repeats < 1 or not len(frame):
        raise ValueError("benchmark requires rows and positive repeats")
    device = next(model.parameters()).device
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    model.eval()
    model.clear_reference_cache()
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        cold = score(model, batches, frame)
    sync()
    cold_s = time.perf_counter()-t0
    warm = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = score(model, batches, frame)
        sync()
        warm.append(time.perf_counter()-t0)
        if not np.allclose(cold, pred, rtol=1e-5, atol=1e-6):
            raise ValueError("cold and warm predictions disagree")
    return dict(rows=len(frame), cold_s=cold_s, warm_s=warm,
        warm_mutants_per_s=len(frame)/float(np.median(warm)),
        warm_seconds_per_mutant=float(np.median(warm))/len(frame),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        regime="supplied sparse library", includes_model_loading=False)


def compare(theirs, ours, out, *, expected=None, expected_source=None, atol=0.005):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    theirs = canonical_predictions(theirs)
    ev = evaluate(theirs)
    for name, table in ev.items():
        table.to_csv(out / f"theirs_full_{name}.csv", index=False)
    gate = reproduction_gate(ev["table"], expected or {}, atol=atol)
    gate["source"] = expected_source
    gate["passed"] = gate["passed"] and bool(expected_source)
    summaries, paired, all_unmatched, seeds = [], [], [], []
    for seed, frame in ours:
        if seed in seeds:
            raise ValueError(f"duplicate seed {seed}")
        seeds.append(seed)
        joined, unmatched = join_predictions(frame, theirs)
        if joined.empty:
            raise ValueError(f"seed {seed}: no matched rows")
        joined.to_csv(out / f"joined_seed{seed}.csv", index=False)
        all_unmatched.append(unmatched.assign(seed=seed))
        evals = {m: evaluate(joined, f"pred_{m}") for m in ("ours", "theirs")}
        for method, res in evals.items():
            summaries.append(res["table"].assign(seed=seed, method=method))
            for name in ("per_protein", "macro", "detection"):
                res[name].to_csv(out / f"{method}_{name}_seed{seed}.csv", index=False)
        for dep in ("all", "single", "double", "multi"):
            a, b = [r["per_protein"].query("depth == @dep").set_index("target_id")
                    for r in evals.values()]
            for metric in ("spearman", "pearson", "mae", "rmse"):
                delta = (a[metric]-b[metric]).dropna()
                paired.extend(dict(seed=seed, target_id=t, depth=dep, metric=metric,
                                   difference=float(v)) for t, v in delta.items())
            a, b = [r["detection"].query("depth == @dep").set_index("target_id")
                    for r in evals.values()]
            for metric in ("ndcg", "precision"):
                delta = (a[metric]-b[metric]).dropna()
                paired.extend(dict(seed=seed, target_id=t, depth=dep, metric=metric,
                                   difference=float(v)) for t, v in delta.items())
    unmatched = pd.concat(all_unmatched, ignore_index=True)
    unmatched.to_csv(out / "unmatched.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(out / "matched_metrics.csv", index=False)
    differences = pd.DataFrame(paired)
    differences.to_csv(out / "paired_differences.csv", index=False)
    intervals = []
    if len(differences):
        for (depth, metric), g in differences.groupby(["depth", "metric"]):
            # A cell must have all supplied seeds, not just all observed seeds.
            complete = len(set(g.seed)) == len(seeds) and g.groupby("target_id").seed.nunique().eq(len(seeds)).all()
            if complete:
                intervals.append(dict(depth=depth, metric=metric, **paired_seed_bootstrap(g)))
    pd.DataFrame(intervals).to_csv(out / "protein_bootstrap.csv", index=False)
    status = dict(reproduction=gate, seeds=seeds, unmatched_rows=len(unmatched),
        superiority_claim_eligible=gate["passed"] and len(seeds) >= 3 and unmatched.empty,
        interpretation="Eligibility is not evidence of a win; inspect paired protein intervals. "
                       "Backbone weights, decoder and scoring regime must be stated.")
    (out / "status.json").write_text(json.dumps(status, indent=2))
    (out / "limitations.md").write_text(
        "This report evaluates supplied predictions on identical joined rows. "
        f"Reproduction gate: {gate['passed']}; seeds: {seeds}; unmatched rows: {len(unmatched)}. "
        "A gate pass does not establish superiority; inspect the paired protein intervals.\n\n"
        "Label the comparator as Mutate Everything ESM-2 650M. Matching ESM-2 "
        "pretrained initialization does not match trained weights or adaptation policy. "
        "Our sparse attention with LoRA and their learned mutation-representation "
        "aggregation differ in decoder and training protocol. Our joint composition "
        "is compared with their higher-order model; additive composition is a null. "
        "Keep ESM-C and AlphaFold rows separately labelled.\n\n"
        "They decode a dense landscape per reference; we score a supplied sparse library. "
        "Timing sources and hardware must accompany any cost plot; these are not "
        "controlled same-GPU benchmarks. Test results, sequence-identity ceiling, "
        "and cross-benchmark leakage require their own audits.\n")
    return status


def plot_frontier(rows, path):
    """Rows need label, seconds_per_mutant, accuracy, regime and cost_source.

    Sources and regimes appear in the figure; published and measured timings
    must not be presented as a controlled same-hardware benchmark.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows.to_dict("records"):
        ax.scatter(r["seconds_per_mutant"], r["accuracy"])
        ax.annotate(f'{r["label"]}\n{r["regime"]}\n{r["cost_source"]}',
                    (r["seconds_per_mutant"], r["accuracy"]), xytext=(5, 5), textcoords="offset points")
    ax.set(xscale="log", xlabel="Seconds per scored mutant (regime dependent)",
           ylabel="Accuracy (same metric and matched rows)",
           title="Accuracy vs cost — timings are not a controlled benchmark")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--theirs", type=Path, required=True)
    p.add_argument("--ours", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected", type=Path)
    p.add_argument("--expected-source")
    p.add_argument("--atol", type=float, default=0.005)
    p.add_argument("--cost-csv", type=Path)
    args = p.parse_args()
    fits = []
    for path in args.ours:
        df = pd.read_csv(path)
        if "seed" not in df or df.seed.nunique() != 1:
            raise ValueError(f"{path}: needs exactly one seed value")
        fits.append((int(df.seed.iloc[0]), df))
    print(compare(pd.read_csv(args.theirs), fits, args.out,
        expected=json.loads(args.expected.read_text()) if args.expected else None,
        expected_source=args.expected_source, atol=args.atol))
    if args.cost_csv:
        plot_frontier(pd.read_csv(args.cost_csv), args.out / "accuracy_cost.png")


if __name__ == "__main__":
    main()
