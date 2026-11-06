"""Does clustering the mutated positions change the cache error?

phase0_degradation.py drew mutated positions uniformly at random over the
sequence. Real combinatorial libraries do not: FLIP GB1 varies V39/D40/G41/V54,
three of which are adjacent, and CombinGym's landscapes are similar. If the
cache's error is driven by local density rather than global density, the
uniform-random numbers are optimistic and item 11 of PLAN.md inherits a problem.

Compares, at matched K, mutated positions drawn (a) uniformly at random,
(b) as one contiguous block, against the full exact forward of the mutant. Then
the real FLIP GB1 site set at its real depths.

fp32, ESM-C 300M, GPU 0.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from mutant_cache.backbone import load_esmc

GB1 = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
PG = pd.read_csv("/home/andrew/mutant_cache_scratch/raw/proteingym/DMS_substitutions.csv")
AVGFP = PG[PG.DMS_id == "GFP_AEQVI_Sarkisyan_2016"].iloc[0].target_seq
PTEN = PG[PG.DMS_id == "PTEN_HUMAN_Matreyek_2021"].iloc[0].target_seq

PROTEINS = {"GB1": GB1, "avGFP": AVGFP, "PTEN": PTEN}
AA = "ACDEFGHIKLMNPQRSTVWY"
N_SAMPLES = 64
SEED = 0

m = load_esmc("esmc_300m", device="cuda", dtype=torch.float32)
MASK = m.tokenizer.mask_token_id
AA_IDS = {a: m.aa_to_id[a] for a in AA}


def exact_masked_llr(mut_ids, pos, alt_ids, wt_ids):
    K = len(pos)
    seqs = mut_ids[None].expand(K, -1).clone()
    seqs[torch.arange(K), pos] = MASK
    out = []
    with torch.no_grad():
        for s in range(0, K, 32):
            lg = m(seqs[s:s + 32]).logits.float()
            idx = torch.arange(lg.shape[0])
            lp = F.log_softmax(lg[idx, pos[s:s + 32]], dim=-1)
            out.append(lp[idx, alt_ids[s:s + 32]] - lp[idx, wt_ids[s:s + 32]])
    return float(torch.cat(out).sum())


def cache_masked_llr(ref, pos, alt_ids, wt_ids):
    K = len(pos)
    vids = alt_ids[None].expand(K, -1).clone()
    vids[torch.arange(K), torch.arange(K)] = MASK
    with torch.no_grad():
        lg = m.forward_variants(ref, pos, vids).logits.float()
    idx = torch.arange(K)
    lp = F.log_softmax(lg[idx, idx], dim=-1)
    return float((lp[idx, alt_ids] - lp[idx, wt_ids]).sum())


def run(seq, pos_sampler, K, rng, label, protein):
    ids = m.tokenizer.encode(seq).cuda()
    L = len(seq)
    with torch.no_grad():
        ref = m.forward_reference(ids)
        href = m(ids[None]).hidden[0].float()
    ex, ca, cos_mut, cos_unmut = [], [], [], []
    for _ in range(N_SAMPLES):
        p = pos_sampler(rng, L, K)
        wt = [seq[i] for i in p]
        alt = [rng.choice([a for a in AA if a != w]) for w in wt]
        tpos = torch.tensor([i + 1 for i in p]).cuda()          # +1 for BOS
        alt_ids = torch.tensor([AA_IDS[a] for a in alt]).cuda()
        wt_ids = torch.tensor([AA_IDS[w] for w in wt]).cuda()
        mut_ids = ids.clone()
        mut_ids[tpos] = alt_ids
        with torch.no_grad():
            hex_ = m(mut_ids[None]).hidden[0].float()
            hca = m.forward_variants(ref, tpos, alt_ids[None]).hidden[0].float()
        cos_mut.append(float(F.cosine_similarity(hca, hex_[tpos], dim=-1).mean()))
        unm = torch.tensor([i for i in range(len(ids)) if i not in set(tpos.tolist())]).cuda()
        cos_unmut.append(float(F.cosine_similarity(href[unm], hex_[unm], dim=-1).mean()))
        ex.append(exact_masked_llr(mut_ids, tpos, alt_ids, wt_ids))
        ca.append(cache_masked_llr(ref, tpos, alt_ids, wt_ids))
    ex, ca = np.array(ex), np.array(ca)
    return dict(protein=protein, L=L, K=K, layout=label,
                cos_mut=np.mean(cos_mut), cos_unmut=np.mean(cos_unmut),
                llr_gap_abs=np.abs(ca - ex).mean(), llr_exact_sd=ex.std(),
                spearman=spearmanr(ca, ex).statistic)


def uniform(rng, L, K):
    return sorted(rng.choice(L, K, replace=False).tolist())


def block(rng, L, K):
    s = int(rng.integers(0, L - K + 1))
    return list(range(s, s + K))


rows = []
for protein, seq in PROTEINS.items():
    for K in [2, 3, 5, 10]:
        for label, sampler in [("uniform", uniform), ("block", block)]:
            rng = np.random.default_rng(SEED)
            rows.append(run(seq, sampler, K, rng, label, protein))
            print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv("/home/andrew/mutant_cache/experiments/phase0_clustered.csv", index=False)

print("\n=== cos_mut / spearman, uniform vs block ===")
piv = df.pivot_table(index=["protein", "K"], columns="layout",
                     values=["cos_mut", "spearman"])
print(piv.to_string())

# The real FLIP GB1 site set: V39 D40 G41 V54 (1-based on the 56-mer), all
# four varied combinatorially -> depth 1..4 at fixed, clustered positions.
print("\n=== FLIP GB1 real sites V39/D40/G41/V54 ===")
sites = [38, 39, 40, 53]                                    # 0-based
ids = m.tokenizer.encode(GB1).cuda()
with torch.no_grad():
    ref = m.forward_reference(ids)
rng = np.random.default_rng(SEED)
grows = []
for K in [1, 2, 3, 4]:
    ex, ca = [], []
    for _ in range(64):
        p = sorted(rng.choice(sites, K, replace=False).tolist())
        wt = [GB1[i] for i in p]
        alt = [rng.choice([a for a in AA if a != w]) for w in wt]
        tpos = torch.tensor([i + 1 for i in p]).cuda()
        alt_ids = torch.tensor([AA_IDS[a] for a in alt]).cuda()
        wt_ids = torch.tensor([AA_IDS[w] for w in wt]).cuda()
        mut_ids = ids.clone()
        mut_ids[tpos] = alt_ids
        ex.append(exact_masked_llr(mut_ids, tpos, alt_ids, wt_ids))
        ca.append(cache_masked_llr(ref, tpos, alt_ids, wt_ids))
    ex, ca = np.array(ex), np.array(ca)
    grows.append(dict(K=K, spearman=spearmanr(ca, ex).statistic,
                      llr_gap_abs=np.abs(ca - ex).mean(), llr_exact_sd=ex.std()))
    print(grows[-1], flush=True)
pd.DataFrame(grows).to_csv(
    "/home/andrew/mutant_cache/experiments/phase0_clustered_flipgb1.csv", index=False)
