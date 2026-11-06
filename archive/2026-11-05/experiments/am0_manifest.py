"""Architecture-mismatch Experiment 0: immutable variant manifest.

TODO_ARCHITECTURE_MISMATCH_PRETRAINING.md, "Data and controls". Train+val
proteins of the ESM-MSR hyperopt split only (test reserved). Selection never
reads score columns. Stratified by natural/designed x length tertile.
Per protein: up to 30 random doubles, their constituent singles where they
were measured, then random singles up to 60.
"""
import hashlib
import numpy as np
import pandas as pd

D = '/home/andrew/mutant_cache_scratch/processed/esm_msr_mega/'
OUT = '/home/andrew/mutant_cache/experiments/am0_manifest.csv'
SEED = 0
N_NAT, N_DES = 24, 16
N_DBL, N_SGL = 30, 60

rng = np.random.default_rng(SEED)
v = pd.read_parquet(D + 'variants.parquet')[['target_id', 'mutant', 'n_mut']]
v['source_row'] = np.arange(len(v))
t = pd.read_parquet(D + 'targets.parquet')[['target_id', 'sequence']]
s = pd.read_parquet(D + 'splits/esm_msr_hyperopt.parquet')[['target_id', 'split']]
t = t.merge(s, on='target_id')
t = t[t.split.isin(['train', 'val'])].sort_values('target_id').reset_index(drop=True)
t['L'] = t.sequence.str.len()
t['designed'] = ~t.target_id.str.match(r'^[0-9][A-Z0-9]{3}')

picked = []
for des, n in ((False, N_NAT), (True, N_DES)):
    g = t[t.designed == des].copy()
    g['tert'] = pd.qcut(g.L.rank(method='first'), 3, labels=False)
    per = [n // 3 + (i < n % 3) for i in range(3)]
    for k in range(3):
        gk = g[g.tert == k]
        picked.append(gk.iloc[rng.choice(len(gk), per[k], replace=False)])
prot = pd.concat(picked).sort_values('target_id')

rows = []
for tid in prot.target_id:
    vt = v[v.target_id == tid]
    dbl = vt[vt.n_mut == 2]
    sgl = vt[vt.n_mut == 1]
    dsel = dbl.iloc[np.sort(rng.choice(len(dbl), min(N_DBL, len(dbl)), replace=False))]
    need = {x for m in dsel.mutant for x in m.split(':')}
    s_need = sgl[sgl.mutant.isin(need)]
    rest = sgl[~sgl.mutant.isin(need)]
    n_fill = max(0, N_SGL - len(s_need))
    s_fill = rest.iloc[np.sort(rng.choice(len(rest), min(n_fill, len(rest)), replace=False))]
    rows.append(pd.concat([s_need, s_fill, dsel]))
man = pd.concat(rows).merge(prot[['target_id', 'split', 'L', 'designed']], on='target_id')
man = man.sort_values(['target_id', 'n_mut', 'source_row']).reset_index(drop=True)
man.to_csv(OUT, index=False)
h = hashlib.sha256(open(OUT, 'rb').read()).hexdigest()
open(OUT + '.sha256', 'w').write(h + '  am0_manifest.csv\n')
print(man.groupby(['designed', 'split']).target_id.nunique())
print(man.n_mut.value_counts(), len(man), h)
