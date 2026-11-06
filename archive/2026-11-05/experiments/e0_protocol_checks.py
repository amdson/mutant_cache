# e0_protocol_checks.py -- TODO.md 2026-11-01, E0. Validates the four task/budget
# manifests (avGFP + GB1, ds320 + ds1280, reps 0-2) used by the implementation pilot
# and writes experiments/e0_manifests.csv + experiments/e0_distributions.csv.
# Asserts: unique genotypes within each split table; no train/val/dev overlap within a
# replicate; released train/val counts (asserted from the manifests, not assumed);
# one shared dev holdout per task, disjoint from the METL supertest and from the
# train/val of every replicate/budget in the queue; every mutant reconstructs against
# the reference (ref residue matches the WT sequence at its 1-based position).
import hashlib
import numpy as np
import polars as pl
from mutant_cache.data.paths import splits_dir
from mutant_cache.data.mutations import parse_mutant

S = "/home/andrew/mutant_cache_scratch/processed"
TASKS = {"metl_avgfp_dms": "metl_avgfp_dms", "metl_gb1_dms": "metl_gb1_dms"}
SIZES, REPS = (320, 1280), (0, 1, 2)
EXPECT = {320: (256, 64), 1280: (1024, 256)}
sha = lambda p: hashlib.sha1(open(p, "rb").read()).hexdigest()[:16]

rows, dist = [], []
for task in TASKS:
    v = pl.read_parquet(f"{S}/{task}/variants.parquet")
    wt = pl.read_parquet(f"{S}/{task}/targets.parquet")["sequence"][0]
    score = dict(zip(v["mutant"].to_list(), v["score"].to_list()))
    nmut = dict(zip(v["mutant"].to_list(), v["n_mut"].to_list()))
    # reconstruction: every mutant's ref residue is the WT residue at that position
    bad = [m for m in v["mutant"].to_list() for p, r, a in parse_mutant(m)
           if wt[p - 1] != r or a == r]
    assert not bad, f"{task}: {len(bad)} mutants do not reconstruct, e.g. {bad[:3]}"
    supertest = set(pl.read_parquet(splits_dir() / f"{task}_ds320_r0.parquet")
                    .filter(pl.col("split") == "test")["mutant"].to_list())
    devs, trainval = [], set()
    for n in SIZES:
        for r in REPS:
            f = splits_dir() / f"{task}_tune_ds{n}_r{r}.parquet"
            t = pl.read_parquet(f)
            assert t["mutant"].n_unique() == len(t), f"{f.name}: duplicate genotypes"
            part = {s: set(t.filter(pl.col("split") == s)["mutant"].to_list())
                    for s in ("train", "val", "test")}
            assert not (part["train"] & part["val"]) and not (part["train"] & part["test"]) \
                and not (part["val"] & part["test"]), f"{f.name}: partition overlap"
            assert (len(part["train"]), len(part["val"])) == EXPECT[n], \
                f"{f.name}: train/val {len(part['train'])}/{len(part['val'])} != {EXPECT[n]}"
            # train/val identical to METL's released replicate
            orig = pl.read_parquet(splits_dir() / f"{task}_ds{n}_r{r}.parquet")
            for s in ("train", "val"):
                assert part[s] == set(orig.filter(pl.col("split") == s)["mutant"].to_list())
            assert not (part["test"] & supertest), f"{f.name}: dev overlaps supertest"
            assert all(m in score for p in part.values() for m in p)
            devs.append(frozenset(part["test"]))
            trainval |= part["train"] | part["val"]
            rows.append({"task": task, "budget": n, "replicate": r, "file": f.name,
                         "sha1": sha(f), "n_train": len(part["train"]), "n_val": len(part["val"]),
                         "n_dev": len(part["test"])})
            for s, ms in part.items():
                ms = sorted(ms)
                d = np.array([nmut[m] for m in ms]); y = np.array([score[m] for m in ms])
                dist.append({"task": task, "budget": n, "replicate": r,
                             "partition": {"test": "dev"}.get(s, s), "n": len(ms),
                             **{f"depth_{k}": int((d == k).sum()) for k in range(1, 5)},
                             "depth_5plus": int((d >= 5).sum()),
                             "score_mean": float(y.mean()), "score_sd": float(y.std()),
                             "score_q10": float(np.quantile(y, .1)),
                             "score_q50": float(np.quantile(y, .5)),
                             "score_q90": float(np.quantile(y, .9))})
    assert len(set(devs)) == 1, f"{task}: dev holdout differs across manifests"
    assert not (devs[0] & trainval), f"{task}: dev overlaps some queue train/val"
    print(f"{task}: OK  dev {len(devs[0]):,}, union train/val {len(trainval):,}, "
          f"supertest {len(supertest):,}, all {len(v):,} mutants reconstruct")
pl.DataFrame(rows).write_csv("experiments/e0_manifests.csv")
pl.DataFrame(dist).write_csv("experiments/e0_distributions.csv")
print(pl.DataFrame(rows))
