"""Architecture-mismatch Experiment 0: pretrained ESM-C 300M, full-mutant
teacher vs sparse-cache student, no adapters. Raw script, run as:

    CUDA_VISIBLE_DEVICES=0 python experiments/am0_measure.py fp32
    CUDA_VISIBLE_DEVICES=0 python experiments/am0_measure.py bf16   # first 8 proteins

Layer index 0..29 = block outputs (pre final norm), 30 = final normalized.
Teacher = model(mutant_ids) (stock path, parity-tested vs official esm).
Student = forward_variants_ragged on the cached WT reference; checked against
forward_variants_bruteforce (clamped oracle) before any teacher comparison.
"""
import hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mutant_cache.backbone import load_esmc
from mutant_cache.cache.reference_cache import base_fingerprint
from mutant_cache.cache.variant_cache import ragged_from_lists
from mutant_cache.data.mutations import parse_mutant, mutate

PREC = sys.argv[1] if len(sys.argv) > 1 else 'fp32'
DTYPE = {'fp32': torch.float32, 'bf16': torch.bfloat16}[PREC]
MAN = '/home/andrew/mutant_cache/experiments/am0_manifest.csv'
SEQS = '/home/andrew/mutant_cache_scratch/processed/esm_msr_mega/targets.parquet'
OUT = Path(f'/home/andrew/mutant_cache_scratch/am0/{PREC}')
OUT.mkdir(parents=True, exist_ok=True)
DEV = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
BANDS = [(1, 1), (2, 3), (4, 8), (9, 10**6)]   # |j - nearest mutated site|
TIMING_N = 2048

man = pd.read_csv(MAN)
man_sha = hashlib.sha256(open(MAN, 'rb').read()).hexdigest()
seqs = dict(pd.read_parquet(SEQS)[['target_id', 'sequence']].values)
proteins = list(dict.fromkeys(man.target_id))
TIMED_ID = proteins[0]
if PREC == 'bf16':
    proteins = proteins[:8]

model = load_esmc('esmc_300m', device=DEV, dtype=DTYPE).freeze_base()
tok = model.tokenizer
aa_id = model.aa_to_id
AAS = sorted(aa_id)
checks = {'prec': PREC, 'manifest_sha256': man_sha, 'base_fingerprint': base_fingerprint(model),
          'proteins': {}}


def stack_full(out):            # ESMCOutput -> (31, B, Ltok, d)
    return torch.stack(out.layer_inputs[1:] + [out.hidden_prenorm, out.hidden]).float()


def stack_var(out):             # VariantOutput -> (31, T, d)
    return torch.stack(out.layer_hidden + [out.hidden]).float()


def cmp(a, b):                  # student a vs target b, (..., d)
    diff = (a - b).norm(dim=-1)
    nb = b.norm(dim=-1)
    return F.cosine_similarity(a, b, dim=-1), diff / nb.clamp_min(1e-12), diff, nb


def timed(fn, reps=3):
    fn(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3, torch.cuda.max_memory_allocated() / 2**20


@torch.no_grad()
def measure(tid, seq, mutants, source_rows, tag):
    L = len(seq)
    ids_wt = tok.encode(seq).to(DEV)
    assert ids_wt.shape[0] == L + 2
    ref = model.forward_reference(ids_wt)
    wt = model(ids_wt)
    ck = {'L': L, 'n_variants': len(mutants)}
    # reference-state check: cached reference == full WT forward
    ck['ref_vs_wt_full_maxabs'] = max(
        [(a[0] - b[0]).abs().max().item() for a, b in zip(wt.layer_inputs, ref.layer_inputs)]
        + [(wt.hidden[0] - ref.final[0]).abs().max().item(),
           (wt.hidden_prenorm[0] - ref.final_prenorm[0]).abs().max().item()])
    W = torch.stack(ref.layer_inputs[1:] + [ref.final_prenorm, ref.final])[:, 0].float()  # (31,Ltok,d)

    subs = [parse_mutant(m) for m in mutants]
    var_lists = []
    for m, sb in zip(mutants, subs):
        for p, w, a in sb:
            assert tok.id_to_aa[int(ref.ids[p])] == w, (tid, m)
        var_lists.append([(p, aa_id[a]) for p, w, a in sb])
    # teacher inputs from the mutated string; must equal splicing into ref.ids
    mut_ids = torch.stack([tok.encode(mutate(seq, m)) for m in mutants]).to(DEV)
    spliced = ref.ids[None].repeat(len(mutants), 1)
    for n, vl in enumerate(var_lists):
        for p, t in vl:
            spliced[n, p] = t
    assert torch.equal(mut_ids, spliced), tid

    teach = model(mut_ids)
    T_full = stack_full(teach)                                    # (31,N,Ltok,d)
    rv = ragged_from_lists(var_lists).to(DEV)
    rv.validate(L + 2)
    so = model.forward_variants_ragged(ref, rv, output_layers=True)
    S = stack_var(so)                                             # (31,Trows,d)
    seg = rv.seg()
    pos = rv.positions

    # correctness oracle: ragged & padded-efficient vs clamped bruteforce
    groups = defaultdict(list)
    for n, vl in enumerate(var_lists):
        groups[tuple(p for p, _ in vl)].append(n)
    offs = rv.offsets.tolist()
    worst = defaultdict(float)
    for gpos, ns in groups.items():
        gp = torch.tensor(gpos, device=DEV)
        vid = torch.tensor([[t for _, t in var_lists[n]] for n in ns], device=DEV)
        bf = model.forward_variants_bruteforce(ref, gp, vid, output_layers=True)
        ef = model.forward_variants(ref, gp, vid, output_layers=True)
        Bf, Ef = stack_var(bf), stack_var(ef)
        rows = torch.tensor([r for n in ns for r in range(offs[n], offs[n + 1])], device=DEV)
        Rg = S[:, rows].reshape(Bf.shape)
        for name, X in (('ragged', Rg), ('padded', Ef)):
            worst[f'{name}_hidden_maxabs'] = max(worst[f'{name}_hidden_maxabs'],
                                                 (X[30] - Bf[30]).abs().max().item())
            rel = ((X[:30] - Bf[:30]).abs().amax(dim=(1, 2, 3))
                   / Bf[:30].abs().amax(dim=(1, 2, 3)).clamp_min(1)).max().item()
            worst[f'{name}_layers_rel'] = max(worst[f'{name}_layers_rel'], rel)
            worst[f'{name}_min_cos'] = min(worst.get(f'{name}_min_cos', 1.0),
                                           F.cosine_similarity(X, Bf, dim=-1).min().item())
    ck.update(worst)
    if PREC == 'fp32':
        assert ck['ref_vs_wt_full_maxabs'] < 1e-4, ck
        for name in ('ragged', 'padded'):
            assert ck[f'{name}_hidden_maxabs'] < 1e-4 and ck[f'{name}_layers_rel'] < 1e-4, ck  # was 1e-5: fp32 oracle itself is 1.0e-5 from fp64 on 2MKX_N29S (sparse 2.9e-6)

    # --- mutated-position metrics, every layer ---
    T = T_full[:, seg, pos]                                       # (31,Trows,d)
    Wp = W[:, pos]
    cos, rel, ab, nT = cmp(S, T)
    dT, dS = T - Wp, S - Wp
    cos_d, rel_d, _, n_dT = cmp(dS, dT)
    nm = torch.tensor([len(v) for v in var_lists], device=DEV)
    site_idx = torch.arange(len(pos), device=DEV) - rv.offsets[seg]
    other = []                                                    # sequence gap to nearest other site
    for vl in var_lists:
        ps = [p for p, _ in vl]
        other += [min([abs(p - q) for q in ps if q != p], default=-1) for p in ps]
    base = pd.DataFrame({
        'target_id': tid, 'tag': tag,
        'mutant': [mutants[i] for i in seg.tolist()],
        'source_row': [source_rows[i] for i in seg.tolist()],
        'n_mut': nm[seg].tolist(), 'site': pos.tolist(), 'site_idx': site_idx.tolist(),
        'rel_site': (pos.float() / L).tolist(), 'gap_other': other, 'L': L})
    recs = []
    for l in range(31):
        d = base.copy()
        d['layer'] = l
        d['cos'] = cos[l].tolist(); d['rel_l2'] = rel[l].tolist(); d['abs_l2'] = ab[l].tolist()
        d['norm_T'] = nT[l].tolist()
        d['cos_delta'] = cos_d[l].tolist(); d['rel_delta'] = rel_d[l].tolist()
        d['ratio_delta'] = (dS[l].norm(dim=-1) / n_dT[l].clamp_min(1e-12)).tolist()
        d['rel_TW'] = (n_dT[l] / Wp[l].norm(dim=-1)).tolist()
        recs.append(d)
    rows_df = pd.concat(recs, ignore_index=True)

    # --- final-layer prediction-relevant features ---
    lpS = so.logits.float().log_softmax(-1)
    lpT = teach.logits.float().log_softmax(-1)[seg, pos]
    lpW = wt.logits.float().log_softmax(-1)[0, pos]
    alt = rv.ids
    wt_tok = ref.ids[pos]
    fin = base.copy()
    fin['kl_T_S'] = (lpT.exp() * (lpT - lpS)).sum(-1).tolist()
    fin['llr_S'] = (lpS.gather(1, alt[:, None])[:, 0] - lpS.gather(1, wt_tok[:, None])[:, 0]).tolist()
    fin['llr_T'] = (lpT.gather(1, alt[:, None])[:, 0] - lpT.gather(1, wt_tok[:, None])[:, 0]).tolist()
    fin['llr_W'] = (lpW.gather(1, alt[:, None])[:, 0] - lpW.gather(1, wt_tok[:, None])[:, 0]).tolist()
    # pooled per variant: mean over mutated sites, and mean over all residues
    N = len(mutants)
    cnt = torch.zeros(N, device=DEV).index_add_(0, seg, torch.ones_like(seg, dtype=torch.float))
    pS = torch.zeros(N, S.shape[-1], device=DEV).index_add_(0, seg, S[30]) / cnt[:, None]
    pT = torch.zeros(N, S.shape[-1], device=DEV).index_add_(0, seg, T[30]) / cnt[:, None]
    pW = torch.zeros(N, S.shape[-1], device=DEV).index_add_(0, seg, Wp[30]) / cnt[:, None]
    seqS = W[30][None].repeat(N, 1, 1)
    seqS[seg, pos] = S[30]
    sS, sT, sW = seqS[:, 1:L + 1].mean(1), T_full[30][:, 1:L + 1].mean(1), W[30][1:L + 1].mean(0)[None]
    var = pd.DataFrame({
        'target_id': tid, 'tag': tag, 'mutant': mutants, 'source_row': source_rows,
        'n_mut': nm.tolist(), 'L': L,
        'cos_sitepool': F.cosine_similarity(pS, pT, dim=-1).tolist(),
        'rel_sitepool': cmp(pS, pT)[1].tolist(),
        'cos_sitepool_delta': F.cosine_similarity(pS - pW, pT - pW, dim=-1).tolist(),
        'cos_seqpool': F.cosine_similarity(sS, sT, dim=-1).tolist(),
        'cos_seqpool_delta': F.cosine_similarity(sS - sW, sT - sW, dim=-1).tolist(),
        'ratio_seqpool_delta': ((sS - sW).norm(dim=-1) / (sT - sW).norm(dim=-1)).tolist()})

    # --- non-mutated positions: teacher context change the sparse stream drops ---
    mutmask = torch.zeros(N, L + 2, dtype=torch.bool, device=DEV)
    mutmask[seg, pos] = True
    j = torch.arange(L + 2, device=DEV)
    dist = torch.full((N, L + 2), 10**6, device=DEV)
    for n, vl in enumerate(var_lists):
        for p, _ in vl:
            dist[n] = torch.minimum(dist[n], (j - p).abs())
    resid = (j >= 1) & (j <= L)
    c_rel = (T_full - W[:, None]).norm(dim=-1) / W[:, None].norm(dim=-1)   # (31,N,Ltok)
    c_cos = F.cosine_similarity(T_full, W[:, None].expand_as(T_full), dim=-1)
    ctx = []
    for k in (1, 2):
        for lo, hi in BANDS:
            msk = (~mutmask) & resid[None] & (dist >= lo) & (dist <= hi) & (nm[:, None] == k)
            c = int(msk.sum())
            if c == 0:
                continue
            ctx.append(pd.DataFrame({
                'target_id': tid, 'tag': tag, 'n_mut': k, 'band': f'{lo}-{hi}', 'layer': range(31),
                'n_pos': c, 'rel_TW_mean': (c_rel * msk).sum((1, 2)).div(c).tolist(),
                'rel_TW_max': c_rel.masked_fill(~msk, 0).amax((1, 2)).tolist(),
                'cos_TW_mean': (c_cos * msk).sum((1, 2)).div(c).tolist()}))
    ctx_df = pd.concat(ctx, ignore_index=True) if ctx else pd.DataFrame()

    # --- doubles vs their singles (only where both singles were measured) ---
    idx = {m: n for n, m in enumerate(mutants)}
    rows_of = lambda n: list(range(offs[n], offs[n + 1]))  # noqa: E731
    dbl = []
    for n, m in enumerate(mutants):
        parts = m.split(':')
        if len(parts) != 2 or not all(x in idx for x in parts):
            continue
        for si, (here, there) in enumerate((parts, parts[::-1])):
            a, b = idx[here], idx[there]
            r_ab = rows_of(n)[si]
            r_a = rows_of(a)[0]
            p = int(pos[r_ab])
            Wa = W[:, p]
            TAB, TA, TB = T_full[:, n, p], T_full[:, a, p], T_full[:, b, p]
            SAB, SA = S[:, r_ab], S[:, r_a]
            NT = TAB - TA - TB + Wa                 # teacher non-additivity at site p
            NS = SAB - SA                           # sparse: single B leaves site p at W
            dbl.append(pd.DataFrame({
                'target_id': tid, 'tag': tag, 'mutant': m, 'site': p, 'gap': abs(p - int(pos[rows_of(b)[0]])),
                'layer': range(31),
                'nonadd_T': (NT.norm(dim=-1) / (TAB - Wa).norm(dim=-1)).tolist(),
                'nonadd_S': (NS.norm(dim=-1) / (SAB - Wa).norm(dim=-1)).tolist(),
                'cos_nonadd': F.cosine_similarity(NS, NT, dim=-1).tolist(),
                'rel_nonadd': ((NS - NT).norm(dim=-1) / NT.norm(dim=-1)).tolist(),
                'cross_T': ((TB - Wa).norm(dim=-1) / (TA - Wa).norm(dim=-1)).tolist()}))
    dbl_df = pd.concat(dbl, ignore_index=True) if dbl else pd.DataFrame()

    # --- runtime / memory at the manifest N and tiled to TIMING_N (first protein only) ---
    if tid != TIMED_ID or tag != 'manifest':
        return ck, rows_df, fin, var, ctx_df, dbl_df
    reps = (TIMING_N + N - 1) // N
    big_ids = mut_ids.repeat(reps, 1)[:TIMING_N]
    big_rv = ragged_from_lists((var_lists * reps)[:TIMING_N]).to(DEV)
    tim = {}
    for nm_, ids_, rv_ in (('N', mut_ids, rv), ('N2048', big_ids, big_rv)):
        tim[f'teacher_ms_{nm_}'], tim[f'teacher_mb_{nm_}'] = timed(lambda: model(ids_))
        tim[f'sparse_ms_{nm_}'], tim[f'sparse_mb_{nm_}'] = timed(
            lambda: model.forward_variants_ragged(ref, rv_, output_logits=False))
    tim['ref_ms'], tim['ref_mb'] = timed(lambda: model.forward_reference(ids_wt))
    ck.update(tim)
    return ck, rows_df, fin, var, ctx_df, dbl_df


@torch.no_grad()
def synthetic_panel(tid, seq):
    """Fixed substitutions: one site, adjacent, separated, depths 3/5/10.
    Also checks forward_variants_exact == model(mutant_ids) at the sites."""
    L = len(seq)
    g = np.random.default_rng(1)
    mid = L // 2
    sets = {'single': [mid], 'adjacent': [mid, mid + 1], 'separated': [L // 4, L // 4 + L // 2]}
    for k in (3, 5, 10):
        sets[f'depth{k}'] = sorted(g.choice(np.arange(1, L + 1), k, replace=False).tolist())
    muts = []
    for name, ps in sets.items():
        subs = []
        for p in ps:
            w = seq[p - 1]
            a = AAS[(AAS.index(w) + 1 + p) % 20]
            a = a if a != w else AAS[(AAS.index(w) + 2) % 20]
            subs.append(f'{w}{p}{a}')
        muts.append(':'.join(subs))
    ids_wt = tok.encode(seq).to(DEV)
    ref = model.forward_reference(ids_wt)
    exact_gap = 0.0
    for m in muts:
        sb = parse_mutant(m)
        gp = torch.tensor([p for p, _, _ in sb], device=DEV)
        vid = torch.tensor([[aa_id[a] for _, _, a in sb]], device=DEV)
        ex = model.forward_variants_exact(ref, gp, vid)
        full = model(tok.encode(mutate(seq, m)).to(DEV)[None])
        exact_gap = max(exact_gap, (ex.hidden[0] - full.hidden[0, gp]).abs().max().item())
    out = measure(tid, seq, muts, [-1] * len(muts), 'synthetic')
    out[0]['exact_vs_full_maxabs'] = exact_gap
    out[0]['panel'] = dict(zip(sets, muts))
    if PREC == 'fp32':
        assert exact_gap < 1e-4, exact_gap
    return out


all_parts = defaultdict(list)
t_start = time.time()
syn_ids = [man[~man.designed].target_id.iloc[0], man[man.designed].target_id.iloc[0]]
for tid in syn_ids:
    ck, *parts = synthetic_panel(tid, seqs[tid])
    checks['proteins'][f'synthetic:{tid}'] = ck
    for k, p in zip(('rows', 'final', 'variant', 'ctx', 'dbl'), parts):
        all_parts[k].append(p)
    print('synthetic', tid, json.dumps({k: v for k, v in ck.items() if k != 'panel'}), flush=True)

for i, tid in enumerate(proteins):
    sub = man[man.target_id == tid]
    ck, *parts = measure(tid, seqs[tid], sub.mutant.tolist(), sub.source_row.tolist(), 'manifest')
    checks['proteins'][tid] = ck
    for k, p in zip(('rows', 'final', 'variant', 'ctx', 'dbl'), parts):
        all_parts[k].append(p)
    print(i, tid, f"L={ck['L']} N={ck['n_variants']} oracle={ck['ragged_hidden_maxabs']:.1e}",
          {k: round(v, 1) for k, v in ck.items() if k.endswith(('_ms_N2048', '_mb_N2048'))}, flush=True)

for k, ps in all_parts.items():
    ps = [p for p in ps if len(p)]
    pd.concat(ps, ignore_index=True).to_parquet(OUT / f'{k}.parquet')
checks['wall_s'] = time.time() - t_start
json.dump(checks, open(OUT / 'checks.json', 'w'), indent=1)
print('done', OUT, f"{checks['wall_s']:.0f}s")
