# Overfit sanity check without the harness: can the variant-cache model memorize
# 10 Tsuboyama variants (5 singles + 5 doubles, one domain)? Plain loop, MSE on
# raw dG written inline. Two configs: with the ESM-C masked-marginal prior, and
# without it (everything learned through the cache path).
# Data: $MUTANT_CACHE_SCRATCH/processed/overfit10 (experiments/overfit10_build.py).
import pandas as pd
import torch
from scipy.stats import spearmanr

from mutant_cache.backbone import load_esmc
from mutant_cache.baselines.zero_shot import build_llr_matrices
from mutant_cache.models import VariantEffectModel, VariantModelConfig, build_variant_batches

D = "/home/andrew/mutant_cache_scratch/processed/overfit10/"
v = pd.read_parquet(D + "variants.parquet")
t = pd.read_parquet(D + "targets.parquet")
seqs = dict(zip(t.target_id, t.sequence))

torch.manual_seed(0)
bb = load_esmc("esmc_300m", device="cuda", dtype=torch.bfloat16).freeze_base()
llr = build_llr_matrices(bb, seqs)

results = {}
for name, use_prior in [("with_prior", True), ("no_prior", False)]:
    torch.manual_seed(0)
    batches, _ = build_variant_batches(v, seqs, dict(bb.aa_to_id), tokenizer=bb.tokenizer,
                                       batch_size=len(v), target_col="score",
                                       marginals=llr if use_prior else None)
    assert len(batches) == 1
    b = batches[0].to("cuda")
    cfg = VariantModelConfig(d_model=bb.d_model, n_assays=1, vocab_size=bb.vocab_size,
                             dropout=0.0, use_marginals=use_prior)
    model = VariantEffectModel(bb, cfg).cuda()
    opt = torch.optim.AdamW(model.head_parameters(), lr=3e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)   # constant lr oscillates at the end

    model.train()
    for step in range(300):
        pred = model(b).pred.float()
        loss = ((pred - b.target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 299:
            print(f"{name} step {step:3d}  mse {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        pred = model(b).pred.float().cpu()
    out = v.iloc[b.row_index.cpu().numpy()][["mutant", "n_mut", "score"]].copy()
    out["pred"] = pred.numpy()
    out["err"] = out.pred - out.score
    mse = (out.err ** 2).mean()
    rho = spearmanr(out.pred, out.score).statistic
    print(out.sort_values("score").round(4).to_string(index=False))
    print(f"{name}: eval mse {mse:.2e}  max|err| {out.err.abs().max():.4f}  spearman {rho:.4f}\n")
    results[name] = (mse, rho)

for name, (mse, rho) in results.items():
    print(("PASS " if mse < 1e-3 and rho > 0.999 else "FAIL ") + name, f"mse={mse:.2e} rho={rho:.4f}")
