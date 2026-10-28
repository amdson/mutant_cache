"""
naive_vs_cache_runtime.py -- full forward of every mutant sequence (naive) on the
same assays as ragged_vs_padded_runtime.py, for the speedup vs naive.

Run as:
    cd ~/mutant_cache
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
      ~/anaconda3/envs/mutant_cache/bin/python experiments/naive_vs_cache_runtime.py

eval  = no_grad full forward of every mutant (fused SDPA), batch sizes swept, best kept
train = fwd+bwd, shared LoRA r=8 (fp32 adapters, bf16 base) so the full stream has
        something to train, over the first TRAIN_VARIANTS variants (same as the cache run)
Loss is logsumexp of the logits at the mutated positions only, like the cache run.
"""
import time

import pandas as pd
import polars as pl
import torch

from mutant_cache.backbone import load_esmc
from mutant_cache.cache.lora import LoRAConfig, attach_shared_lora, detach_shared_lora
from mutant_cache.data.mutations import parse_mutant

D = "/home/andrew/mutant_cache_scratch/processed/"
OUT = "/home/andrew/mutant_cache/experiments/naive_vs_cache_runtime.csv"
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
    seq = t.filter(pl.col("target_id") == v["target_id"][0])["sequence"][0]
    return v, seq


def mutant_tensors(v, seq):
    ref = m.tokenizer.encode(seq).long()
    ids = ref[None].repeat(len(v), 1)
    tok_pos = []
    for i, mut in enumerate(v["mutant"].to_list()):
        ps = []
        for p, wt, alt in parse_mutant(mut):
            ids[i, p] = m.aa_to_id[alt]           # residue p (1-based) -> token p
            ps.append(p)
        tok_pos.append(ps)
    return ids, tok_pos


def run(ids, tok_pos, bs, train, limit=None):
    n = len(ids) if limit is None else min(limit, len(ids))
    chunks = []
    for s in range(0, n, bs):
        rows = [r for r, ps in enumerate(tok_pos[s:s + bs]) for _ in ps]
        cols = [p for ps in tok_pos[s:s + bs] for p in ps]
        chunks.append((ids[s:s + bs].cuda(), torch.tensor(rows).cuda(), torch.tensor(cols).cuda()))

    def one(c):
        x, r, p = c
        out = m(x)
        if train:
            out.logits[r, p].float().logsumexp(-1).sum().backward()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for c in chunks[:2]:
            one(c)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        for c in chunks:
            one(c)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return dict(variants=n, seconds=dt, variants_per_s=n / dt,
                peak_gib=(torch.cuda.max_memory_allocated() - base) / 2**30)


rows = []
for name in ["BRCA1", "GB1"]:
    v, seq = load(name)
    ids, tok_pos = mutant_tensors(v, seq)
    print(f"\n== {name}: {len(v)} variants, L={len(seq)}", flush=True)
    for train in (False, True):
        if train:
            attach_shared_lora(m, LoRAConfig(r=8))
        for bs in ([32, 128, 512] if not train else [8, 32, 64]):
            try:
                r = run(ids, tok_pos, bs, train, TRAIN_VARIANTS if train else None)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                r = dict(variants=float("nan"), seconds=float("nan"),
                         variants_per_s=float("nan"), peak_gib=float("nan"))
            r.update(dataset=name, mode="train" if train else "eval", batch_size=bs)
            rows.append(r)
            print(f"  {r['mode']:5} naive bs={bs:<4} {r['seconds']:8.2f}s "
                  f"{r['variants_per_s']:8.0f} var/s  peak {r['peak_gib']:.2f} GiB", flush=True)
        if train:
            detach_shared_lora(m)

pd.DataFrame(rows).to_csv(OUT, index=False)
print("wrote", OUT)
