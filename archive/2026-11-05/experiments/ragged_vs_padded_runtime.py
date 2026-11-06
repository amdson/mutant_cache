"""
ragged_vs_padded_runtime.py -- padded shared-position batches (what
build_variant_batches + forward_variants do today) vs ragged per-variant
positions (forward_variants_ragged), on real assays.

Run as:
    cd ~/mutant_cache
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
      ~/anaconda3/envs/mutant_cache/bin/python experiments/ragged_vs_padded_runtime.py

Datasets:
  BRCA1 (MaveDB urn:mavedb:00000003-b-1): 15966 variants, singles/doubles at mixed sites
  GB1 (FLIP): 4-site combinatorial, ~all variants share the same sites (padded's best case)
Same batch composition for padded and ragged at a given batch size (ragged rows are
exactly padded[var_mask]); ragged is additionally run with big batches, since it
no longer pays for the union.
eval = no_grad forward over every batch; train = fwd+bwd with a variant LoRA (r=8,
fp32 adapters on a bf16 base) over the first TRAIN_VARIANTS variants' batches.
"""
import time

import pandas as pd
import polars as pl
import torch

from mutant_cache.backbone import load_esmc
from mutant_cache.cache.lora import LoRAConfig, attach_variant_lora, detach_variant_lora
from mutant_cache.cache.variant_cache import (forward_variants, forward_variants_ragged,
                                              ragged_from_padded)
from mutant_cache.models.variant_model import build_variant_batches

D = "/home/andrew/mutant_cache_scratch/processed/"
OUT = "/home/andrew/mutant_cache/experiments/ragged_vs_padded_runtime.csv"
TRAIN_VARIANTS = 4096
GB1_SUBSAMPLE = 20000

m = load_esmc("esmc_300m", device="cuda", dtype=torch.bfloat16).freeze_base()


def load(name):
    if name == "BRCA1":
        v = pl.read_parquet(D + "mavedb/variants.parquet").filter(
            pl.col("score_set_id") == "urn:mavedb:00000003-b-1")
        t = pl.read_parquet(D + "mavedb/targets.parquet")
    else:
        v = pl.read_parquet(D + "flip_gb1/variants.parquet")
        v = v.filter(pl.col("score_set_id") == v["score_set_id"][0]).sample(
            GB1_SUBSAMPLE, seed=0)
        t = pl.read_parquet(D + "flip_gb1/targets.parquet")
    v = v.filter(pl.col("n_mut") > 0)
    t = t.filter(pl.col("target_id").is_in(v["target_id"].unique()))
    return v.to_pandas(), dict(zip(t["target_id"], t["sequence"]))


def batches_for(v, seqs, bs):
    bb, _ = build_variant_batches(v, seqs, dict(m.aa_to_id), batch_size=bs,
                                  tokenizer=m.tokenizer, shuffle_seed=0)
    return bb


def run(kind, bb, ref, train, limit=None):
    calls = []
    n_var = 0
    for b in bb:
        if limit is not None and n_var >= limit:
            break
        n_var += b.variant_ids.shape[0]
        pos, vid, vm = b.positions.cuda(), b.variant_ids.cuda(), b.var_mask.cuda()
        rv = ragged_from_padded(pos, vid, vm) if kind == "ragged" else None
        calls.append((pos, vid, vm, rv))

    def one(c):
        pos, vid, vm, rv = c
        if kind == "ragged":
            out = forward_variants_ragged(m, ref, rv)
        else:
            out = forward_variants(m, ref, pos, vid, var_mask=vm)
        if train:
            out.logits.float().logsumexp(-1).sum().backward()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for c in calls[:3]:
            one(c)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        for c in calls:
            one(c)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    rows_padded = sum(int(c[1].numel()) for c in calls)
    rows_real = sum(int(c[2].sum()) for c in calls)
    return dict(variants=n_var, calls=len(calls), rows_computed=rows_real if kind == "ragged"
                else rows_padded, rows_real=rows_real, seconds=dt,
                variants_per_s=n_var / dt, peak_gib=peak)


rows = []
for name in ["BRCA1", "GB1"]:
    v, seqs = load(name)
    seq = next(iter(seqs.values()))
    ref = m.forward_reference(m.tokenizer.encode(seq))
    print(f"\n== {name}: {len(v)} variants, L={len(seq)}, "
          f"n_mut {v['n_mut'].value_counts().sort_index().to_dict()}", flush=True)
    configs = [("padded", 32), ("ragged", 32), ("padded", 256), ("ragged", 256),
               ("ragged", 1024)]
    cache = {}
    for train in (False, True):
        if train:
            attach_variant_lora(m, LoRAConfig(r=8))
        for kind, bs in configs:
            if bs not in cache:
                cache[bs] = batches_for(v, seqs, bs)
            try:
                r = run(kind, cache[bs], ref, train, TRAIN_VARIANTS if train else None)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                r = dict(variants=float("nan"), calls=float("nan"), rows_computed=float("nan"),
                         rows_real=float("nan"), seconds=float("nan"),
                         variants_per_s=float("nan"), peak_gib=float("nan"))
            r.update(dataset=name, mode="train" if train else "eval", kind=kind, batch_size=bs)
            rows.append(r)
            print(f"  {r['mode']:5} {kind:6} bs={bs:<5} calls={r['calls']:<5} "
                  f"rows {r['rows_computed']:>8} (real {r['rows_real']:>7})  "
                  f"{r['seconds']:7.2f}s  {r['variants_per_s']:8.0f} var/s  "
                  f"peak {r['peak_gib']:.2f} GiB", flush=True)
        if train:
            detach_variant_lora(m)

pd.DataFrame(rows).to_csv(OUT, index=False)
print("wrote", OUT)
