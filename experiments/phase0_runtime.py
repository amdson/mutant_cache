"""
phase0_runtime.py  --  PLAN.md Phase 0 item 5 (go/no-go).

Sparse variant cache vs a full forward of every mutant: throughput, speedup,
peak memory, break-even mutation count, and the LoRA fine-tuning path.

Run as:
    cd ~/mutant_cache
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
      ~/anaconda3/envs/mutant_cache/bin/python experiments/phase0_runtime.py

Writes experiments/phase0_runtime_grid.csv, _nsweep.csv, _lora.csv.

Accounting
----------
The realistic workload is one reference and thousands of mutants, so two
numbers are reported for the cache:
  amortized  N / (t_reference + t_variants)   -- includes the one-off ref pass
  per-call   N / t_variants                   -- ref already cached on disk
The full-forward baseline is the same N mutant sequences run in batches
(t_full / N). Reference forward time is measured once per length.

bf16 is the headline dtype (frozen bf16 base + fp32 adapters is the training
plan); the grid is repeated in fp32 for completeness.
"""
import time

import numpy as np
import pandas as pd
import torch

from mutant_cache.backbone import load_esmc
from mutant_cache.cache.lora import LoRAConfig, attach_variant_lora, detach_variant_lora

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

OUT_GRID = "/home/andrew/mutant_cache/experiments/phase0_runtime_grid.csv"
OUT_NSWEEP = "/home/andrew/mutant_cache/experiments/phase0_runtime_nsweep.csv"
OUT_LORA = "/home/andrew/mutant_cache/experiments/phase0_runtime_lora.csv"

LENGTHS = [64, 128, 256, 512, 1024]
KS = [1, 2, 5, 10, 20, 50, 100, 200, 400]
N_GRID = 256
N_SWEEP = [1, 4, 16, 64, 256, 1024]
REPS = 3
AA = "ACDEFGHIKLMNPQRSTVWY"
MAX_VAR_PAIRS = 16384      # chunk the variant batch so N*K stays bounded
MAX_FULL_TOKENS = 32768    # chunk the full-forward batch the same way

rng = np.random.default_rng(0)


def rand_seq(L):
    return "".join(rng.choice(list(AA), size=L))


def timeit(fn, reps=REPS, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps, torch.cuda.max_memory_allocated() / 2 ** 20


def full_forward_all(model, mut_ids, N):
    """N mutant sequences through the ordinary model, batched."""
    L = mut_ids.shape[1]
    B = max(1, min(N, MAX_FULL_TOKENS // L))
    with torch.no_grad():
        for s in range(0, N, B):
            model(mut_ids[s:s + B])


def variants_all(model, ref, pos, vids, N, grad=False):
    K = pos.shape[0]
    B = max(1, min(N, MAX_VAR_PAIRS // K))
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for s in range(0, N, B):
            out = model.forward_variants(ref, pos, vids[s:s + B])
            if grad:
                out.hidden.float().pow(2).mean().backward()


def make_case(model, L, K, N):
    """Reference ids, variant positions/ids, and the N full mutant sequences."""
    seq = rand_seq(L)
    ids = model.tokenizer.encode(seq).cuda()
    pos = torch.as_tensor(np.sort(rng.choice(L, size=K, replace=False)) + 1,
                          device="cuda").long()
    aa_ids = torch.tensor([model.aa_to_id[a] for a in AA], device="cuda")
    vids = aa_ids[torch.randint(0, 20, (N, K), device="cuda")]
    mut_ids = ids[None].expand(N, -1).contiguous()
    mut_ids[:, pos] = vids
    return ids, pos, vids, mut_ids


# --------------------------------------------------------------------------- #
# 1. grid: length x n_mut, fixed N
# --------------------------------------------------------------------------- #
rows = []
for dtype, dname in [(torch.bfloat16, "bf16"), (torch.float32, "fp32")]:
    model = load_esmc("esmc_300m", device="cuda", dtype=dtype)
    for L in LENGTHS:
        ids, _, _, _ = make_case(model, L, 1, 1)
        t_ref, m_ref = timeit(lambda: model.forward_reference(ids))
        _, _, _, mut_ids = make_case(model, L, 1, N_GRID)
        t_full, m_full = timeit(lambda: full_forward_all(model, mut_ids, N_GRID))
        ref = model.forward_reference(ids)
        for K in KS:
            if K > L:
                continue
            ids2, pos, vids, mut_ids2 = make_case(model, L, K, N_GRID)
            ref2 = model.forward_reference(ids2)
            t_var, m_var = timeit(lambda: variants_all(model, ref2, pos, vids, N_GRID))
            rows.append(dict(
                dtype=dname, L=L, n_mut=K, density=K / L, N=N_GRID,
                t_ref_s=t_ref, t_full_s=t_full, t_var_s=t_var,
                full_mut_per_s=N_GRID / t_full,
                cache_amort_mut_per_s=N_GRID / (t_ref + t_var),
                cache_percall_mut_per_s=N_GRID / t_var,
                speedup_amort=t_full / (t_ref + t_var),
                speedup_percall=t_full / t_var,
                mem_full_mib=m_full, mem_var_mib=m_var, mem_ref_mib=m_ref,
                ref_state_mib=ref2.nbytes() / 2 ** 20,
            ))
            print(rows[-1]["dtype"], L, K, f"sp_amort={rows[-1]['speedup_amort']:.2f}",
                  f"sp_call={rows[-1]['speedup_percall']:.2f}", flush=True)
            del ref2, mut_ids2
            torch.cuda.empty_cache()
        del ref
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()

grid = pd.DataFrame(rows)
grid.to_csv(OUT_GRID, index=False)

# --------------------------------------------------------------------------- #
# 2. N sweep (how fast the reference pass amortizes), L=512, K=5
# --------------------------------------------------------------------------- #
rows = []
model = load_esmc("esmc_300m", device="cuda", dtype=torch.bfloat16)
for L, K in [(512, 1), (512, 5), (512, 20)]:
    for N in N_SWEEP:
        ids, pos, vids, mut_ids = make_case(model, L, K, N)
        t_ref, _ = timeit(lambda: model.forward_reference(ids))
        ref = model.forward_reference(ids)
        t_full, m_full = timeit(lambda: full_forward_all(model, mut_ids, N))
        t_var, m_var = timeit(lambda: variants_all(model, ref, pos, vids, N))
        rows.append(dict(dtype="bf16", L=L, n_mut=K, N=N, t_ref_s=t_ref,
                         t_full_s=t_full, t_var_s=t_var,
                         full_mut_per_s=N / t_full,
                         cache_amort_mut_per_s=N / (t_ref + t_var),
                         cache_percall_mut_per_s=N / t_var,
                         speedup_amort=t_full / (t_ref + t_var),
                         speedup_percall=t_full / t_var,
                         mem_full_mib=m_full, mem_var_mib=m_var))
        print("nsweep", L, K, N, f"{rows[-1]['speedup_amort']:.2f}", flush=True)
        del ref, mut_ids
        torch.cuda.empty_cache()
nsweep = pd.DataFrame(rows)
nsweep.to_csv(OUT_NSWEEP, index=False)

# --------------------------------------------------------------------------- #
# 3. LoRA (the fine-tuning path): forward, and forward+backward
# --------------------------------------------------------------------------- #
rows = []
attach_variant_lora(model, LoRAConfig(r=16, alpha=32.0))
for L in [128, 512, 1024]:
    for K in [1, 5, 20, 50]:
        N = 128
        ids, pos, vids, mut_ids = make_case(model, L, K, N)
        ref = model.forward_reference(ids)
        t_ref, _ = timeit(lambda: model.forward_reference(ids))
        t_var, m_var = timeit(lambda: variants_all(model, ref, pos, vids, N))
        t_bwd, m_bwd = timeit(lambda: variants_all(model, ref, pos, vids, N, grad=True))
        t_full, m_full = timeit(lambda: full_forward_all(model, mut_ids, N))
        rows.append(dict(dtype="bf16", L=L, n_mut=K, N=N, lora_r=16,
                         t_ref_s=t_ref, t_var_fwd_s=t_var, t_var_fwdbwd_s=t_bwd,
                         t_full_fwd_s=t_full,
                         train_mut_per_s=N / t_bwd,
                         speedup_fwd_percall=t_full / t_var,
                         speedup_train_vs_fullfwd=t_full / t_bwd,
                         mem_var_fwd_mib=m_var, mem_var_fwdbwd_mib=m_bwd,
                         mem_full_fwd_mib=m_full))
        print("lora", L, K, f"fwd={t_var:.3f} fwd+bwd={t_bwd:.3f} full={t_full:.3f}",
              flush=True)
        del ref, mut_ids
        torch.cuda.empty_cache()
detach_variant_lora(model)
lora = pd.DataFrame(rows)
lora.to_csv(OUT_LORA, index=False)

# --------------------------------------------------------------------------- #
pd.set_option("display.width", 250)
print("\n=== grid (bf16) speedup_amort, rows=L cols=n_mut ===")
g = grid[grid.dtype == "bf16"]
print(g.pivot(index="L", columns="n_mut", values="speedup_amort").to_string(float_format=lambda x: f"{x:.2f}"))
print("\n=== grid (fp32) speedup_amort ===")
g32 = grid[grid.dtype == "fp32"]
print(g32.pivot(index="L", columns="n_mut", values="speedup_amort").to_string(float_format=lambda x: f"{x:.2f}"))
print("\n=== break-even n_mut (speedup_amort = 1), bf16 ===")
for L, sub in g.groupby("L"):
    sub = sub.sort_values("n_mut")
    s = sub.speedup_amort.values
    k = sub.n_mut.values
    if s.min() > 1:
        print(L, f">{k.max()} (never breaks even up to K={k.max()}, density {k.max()/L:.2f})")
    else:
        i = np.argmax(s < 1)
        kb = np.interp(0.0, [np.log(s[i - 1]), np.log(s[i])], [k[i - 1], k[i]])
        print(L, f"K~{kb:.0f}  density~{kb / L:.3f}")
print("\n=== n sweep ===")
print(nsweep.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== lora ===")
print(lora.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
