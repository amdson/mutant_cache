"""
ragged_batch_memory.py -- peak GPU memory of one training step (fwd+bwd) through
VariantEffectModel on the ragged cache vs T (real substitutions per batch) and L.
Head + variant LoRA r=8, esmc_300m bf16 base, rank-free loss (sum of preds^2).

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 ~/anaconda3/envs/mutant_cache/bin/python experiments/ragged_batch_memory.py
"""
import numpy as np
import pandas as pd
import torch

from mutant_cache.backbone import load_esmc
from mutant_cache.cache.lora import LoRAConfig, attach_variant_lora
from mutant_cache.models import VariantEffectModel, VariantModelConfig, build_variant_batches

AA = "ACDEFGHIKLMNPQRSTVWY"
m = load_esmc("esmc_300m", device="cuda", dtype=torch.bfloat16).freeze_base()
attach_variant_lora(m, LoRAConfig(r=8))
model = VariantEffectModel(m, VariantModelConfig(n_assays=1)).cuda().train()
rng = np.random.default_rng(0)
rows = []
for L in (64, 300, 1000):
    seq = "".join(rng.choice(list(AA), L))
    ref = None
    for n_var in (256, 1024, 4096, 16384, 65536):
        for k in (1, 4):
            muts = []
            for _ in range(n_var):
                ps = sorted(rng.choice(np.arange(1, L + 1), k, replace=False))
                muts.append(":".join(f"{seq[p-1]}{p}{rng.choice([a for a in AA if a != seq[p-1]])}" for p in ps))
            df = pd.DataFrame({"score_set_id": "S", "target_id": "T", "mutant": muts,
                               "n_mut": k, "score": 0.0, "score_rank": 0.5})
            b = build_variant_batches(df, {"T": seq}, dict(m.aa_to_id), tokenizer=m.tokenizer,
                                      batch_size=n_var)[0][0].to("cuda")
            T = int(b.var_mask.sum())
            try:
                model(b).pred.square().sum().backward()          # warm (compile, ref cache)
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                model(b).pred.square().sum().backward()
                torch.cuda.synchronize()
                peak = (torch.cuda.max_memory_allocated() - base) / 2**30
            except torch.OutOfMemoryError:
                peak = float("nan")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            rows.append(dict(L=L, variants=n_var, n_mut=k, T=T, union_K=b.k, peak_gib=peak,
                             kib_per_row=peak * 2**20 / T))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("/home/andrew/mutant_cache/experiments/ragged_batch_memory.csv", index=False)
