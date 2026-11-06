"""
phase0_degradation.py  --  PLAN.md Phase 0 item 4 (go/no-go).

How badly does the sparse variant cache diverge from a real full forward of the
mutant sequence, as a function of mutation count / density / sequence length?

Run as:
    cd ~/mutant_cache
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
      ~/anaconda3/envs/mutant_cache/bin/python experiments/phase0_degradation.py

Writes experiments/phase0_degradation.csv (one row per sampled mutant) and
experiments/phase0_degradation_summary.csv.

What is compared, per sampled mutant with K substitutions at token positions P:

  exact      model(mutant_ids)                       -- the real thing
  cache      forward_variants(ref_wt, P, alt_ids)    -- the approximation
                                                        (= reference state at
                                                         every position not in P)

  cos_mut    cosine of the final hidden state at the mutated positions
  cos_unmut  cosine at the UNmutated positions, i.e. ref.final vs the exact
             mutant's hidden there -- the part the cache gets wrong by
             construction (reference positions never see the variants)
  cos_pool   cosine of the mean-pooled residue embedding (cache's pooled vector
             is built from ref at unmutated + cache at mutated positions)

  LLR        masked-marginal log-odds in MUTANT context: for each mutated
             position j, mask j (leaving the other K-1 substitutions in place),
             read log p(alt_j) - log p(wt_j); the variant score is the sum.
             For K=1 this is exactly the standard ESM-C zero-shot masked
             marginal.  Cache version: the same K positions are the variant
             slots and slot j carries <mask> instead of alt_j.

fp32 is the headline comparison; the same mutants are also run with a bf16 copy
of the model so the bf16 cost can be separated from the approximation error.
"""
import itertools
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from mutant_cache.backbone import load_esmc

torch.backends.cuda.matmul.allow_tf32 = False   # keep fp32 honest
torch.backends.cudnn.allow_tf32 = False

OUT_ROWS = "/home/andrew/mutant_cache/experiments/phase0_degradation.csv"
OUT_SUMMARY = "/home/andrew/mutant_cache/experiments/phase0_degradation_summary.csv"

# ---------------------------------------------------------------------------
# Reference proteins: real DMS targets, sequences taken verbatim from
# ProteinGym DMS_substitutions.csv (target_seq) / the FLIP GB1 construct.
# ---------------------------------------------------------------------------
PG = pd.read_csv("/home/andrew/mutant_cache_scratch/raw/proteingym/DMS_substitutions.csv")


def pg_seq(dms_id):
    return PG[PG.DMS_id == dms_id].iloc[0].target_seq


PROTEINS = {
    # 56 aa: the GB1 domain used by Olson 2014 / Wu 2016 / FLIP (first 56 aa of
    # the FLIP 5LDE_1 fusion construct).
    "GB1": "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    # 76 aa: ubiquitin, first 76 aa of the RL40A_YEAST ubiquitin-ribosomal
    # fusion used by Roscoe 2013/2014 and Mavor 2016.
    "ubiquitin": pg_seq("RL40A_YEAST_Roscoe_2013")[:76],
    # 238 aa: avGFP, Sarkisyan 2016.
    "avGFP": pg_seq("GFP_AEQVI_Sarkisyan_2016"),
    # 403 aa: human PTEN, Matreyek 2021.
    "PTEN": pg_seq("PTEN_HUMAN_Matreyek_2021"),
}

N_MUTS = [1, 2, 3, 5, 10, 20, 50]
N_SAMPLES = 64
SEED = 0
MASK_BATCH = 32          # sequences per exact masked forward
AA = "ACDEFGHIKLMNPQRSTVWY"

print({k: len(v) for k, v in PROTEINS.items()})

m32 = load_esmc("esmc_300m", device="cuda", dtype=torch.float32)
mbf = load_esmc("esmc_300m", device="cuda", dtype=torch.bfloat16)
MASK = m32.tokenizer.mask_token_id
AA_IDS = {a: m32.aa_to_id[a] for a in AA}


def exact_hidden(model, ids):
    with torch.no_grad():
        return model(ids[None]).hidden[0].float()


def exact_masked_llr(model, mut_ids, pos, alt_ids, wt_ids):
    """Masked-marginal LLR in mutant context, summed over the K positions."""
    K = len(pos)
    seqs = mut_ids[None].expand(K, -1).clone()
    seqs[torch.arange(K), pos] = MASK
    tot = 0.0
    per = []
    with torch.no_grad():
        for s in range(0, K, MASK_BATCH):
            lg = model(seqs[s:s + MASK_BATCH]).logits.float()
            idx = torch.arange(lg.shape[0])
            lp = F.log_softmax(lg[idx, pos[s:s + MASK_BATCH]], dim=-1)
            d = lp[idx, alt_ids[s:s + MASK_BATCH]] - lp[idx, wt_ids[s:s + MASK_BATCH]]
            per.append(d)
            tot += float(d.sum())
    return tot, torch.cat(per)


def cache_masked_llr(model, ref, pos, alt_ids, wt_ids):
    """Same thing through the cache: K variant rows, row j masks slot j."""
    K = len(pos)
    vids = alt_ids[None].expand(K, -1).clone()
    vids[torch.arange(K), torch.arange(K)] = MASK
    with torch.no_grad():
        lg = model.forward_variants(ref, pos, vids).logits.float()
    idx = torch.arange(K)
    lp = F.log_softmax(lg[idx, idx], dim=-1)
    d = lp[idx, alt_ids] - lp[idx, wt_ids]
    return float(d.sum()), d


rows = []
t0 = time.time()
for pname, seq in PROTEINS.items():
    L = len(seq)
    ids32 = m32.tokenizer.encode(seq).cuda()
    T = ids32.shape[0]                       # L + 2
    ref32 = m32.forward_reference(ids32)
    refbf = mbf.forward_reference(ids32)
    refh32 = ref32.final[0].float()
    refhbf = refbf.final[0].float()
    rng = np.random.default_rng(SEED)

    for K in N_MUTS:
        if K > L:
            continue
        for s in range(N_SAMPLES):
            res = np.sort(rng.choice(L, size=K, replace=False)) + 1      # 1-based residue
            pos = torch.as_tensor(res, device="cuda").long()             # token index
            wt = [seq[r - 1] for r in res]
            alt = [rng.choice([a for a in AA if a != w]) for w in wt]
            wt_ids = torch.tensor([AA_IDS[a] for a in wt], device="cuda")
            alt_ids = torch.tensor([AA_IDS[a] for a in alt], device="cuda")

            mut = list(seq)
            for r, a in zip(res, alt):
                mut[r - 1] = a
            mut_ids = m32.tokenizer.encode("".join(mut)).cuda()

            # ---- hidden states ------------------------------------------- #
            h32 = exact_hidden(m32, mut_ids)                 # (T, d) exact fp32
            hbf = exact_hidden(mbf, mut_ids)                 # (T, d) exact bf16
            with torch.no_grad():
                c32 = m32.forward_variants(ref32, pos, alt_ids[None],
                                           output_logits=False).hidden[0].float()
                cbf = mbf.forward_variants(refbf, pos, alt_ids[None],
                                           output_logits=False).hidden[0].float()

            unmut = torch.ones(T, dtype=torch.bool)
            unmut[0] = unmut[T - 1] = False                  # drop BOS/EOS
            unmut[pos.cpu()] = False

            cos_mut = float(F.cosine_similarity(c32, h32[pos], dim=-1).mean())
            cos_mut_min = float(F.cosine_similarity(c32, h32[pos], dim=-1).min())
            cos_unmut = float(F.cosine_similarity(refh32[unmut], h32[unmut], dim=-1).mean())
            cos_unmut_min = float(F.cosine_similarity(refh32[unmut], h32[unmut], dim=-1).min())
            cos_bos = float(F.cosine_similarity(refh32[0], h32[0], dim=0))

            # pooled: the cache's reconstruction of the whole sequence
            full32 = refh32.clone()
            full32[pos] = c32
            resid = slice(1, T - 1)
            cos_pool = float(F.cosine_similarity(full32[resid].mean(0),
                                                 h32[resid].mean(0), dim=0))

            cos_mut_bf = float(F.cosine_similarity(cbf, h32[pos], dim=-1).mean())
            cos_mut_num = float(F.cosine_similarity(hbf[pos], h32[pos], dim=-1).mean())

            # ---- masked-marginal LLR -------------------------------------- #
            e32, _ = exact_masked_llr(m32, mut_ids, pos, alt_ids, wt_ids)
            k32, _ = cache_masked_llr(m32, ref32, pos, alt_ids, wt_ids)
            ebf, _ = exact_masked_llr(mbf, mut_ids, pos, alt_ids, wt_ids)
            kbf, _ = cache_masked_llr(mbf, refbf, pos, alt_ids, wt_ids)

            rows.append(dict(
                protein=pname, L=L, n_mut=K, density=K / L, sample=s,
                mutant=":".join(f"{w}{r}{a}" for w, r, a in zip(wt, res, alt)),
                cos_mut=cos_mut, cos_mut_min=cos_mut_min,
                cos_unmut=cos_unmut, cos_unmut_min=cos_unmut_min,
                cos_bos=cos_bos, cos_pool=cos_pool,
                cos_mut_bf16=cos_mut_bf, cos_mut_bf16_numerics=cos_mut_num,
                llr_exact=e32, llr_cache=k32,
                llr_exact_bf16=ebf, llr_cache_bf16=kbf,
            ))
        print(f"{pname} L={L} K={K} done  ({time.time() - t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_ROWS, index=False)

# ---------------------------------------------------------------------------
# Summary: per (protein, n_mut)
# ---------------------------------------------------------------------------
summ = []
for (p, K), g in df.groupby(["protein", "n_mut"], sort=False):
    gap = (g.llr_cache - g.llr_exact)
    summ.append(dict(
        protein=p, L=int(g.L.iloc[0]), n_mut=K, density=float(g.density.iloc[0]),
        cos_mut=g.cos_mut.mean(), cos_mut_p5=g.cos_mut.quantile(0.05),
        cos_unmut=g.cos_unmut.mean(), cos_unmut_min=g.cos_unmut_min.mean(),
        cos_pool=g.cos_pool.mean(), cos_bos=g.cos_bos.mean(),
        cos_mut_bf16=g.cos_mut_bf16.mean(),
        cos_mut_bf16_numerics=g.cos_mut_bf16_numerics.mean(),
        llr_gap_mean=gap.mean(), llr_gap_abs=gap.abs().mean(),
        llr_gap_abs_per_mut=gap.abs().mean() / K,
        llr_exact_sd=g.llr_exact.std(),
        spearman=spearmanr(g.llr_cache, g.llr_exact).statistic,
        spearman_bf16=spearmanr(g.llr_cache_bf16, g.llr_exact).statistic,
        pearson=float(np.corrcoef(g.llr_cache, g.llr_exact)[0, 1]),
    ))
sdf = pd.DataFrame(summ)
sdf.to_csv(OUT_SUMMARY, index=False)
pd.set_option("display.width", 250)
print(sdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# pooled-across-proteins spearman at each depth, and a density regression
print("\npooled spearman by n_mut:")
for K, g in df.groupby("n_mut"):
    print(K, f"{spearmanr(g.llr_cache, g.llr_exact).statistic:.4f}")
print("\ncos_mut ~ density (log-log slope):")
x = np.log(sdf.density.values)
y = np.log(1 - sdf.cos_mut.values + 1e-9)
print(np.polyfit(x, y, 1))
print("\ncos_mut ~ L at n_mut=1:")
print(sdf[sdf.n_mut == 1][["protein", "L", "cos_mut", "cos_unmut", "spearman"]].to_string(index=False))
print(json.dumps(dict(elapsed_s=time.time() - t0)))
