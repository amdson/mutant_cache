"""Summarize am0_measure output. Raw script: python experiments/am0_summarize.py fp32

Error levels below were fixed 2026-11-02 after the oracle checks passed and
BEFORE any teacher-vs-student number was inspected:
  final-layer cos(S, T) < 0.99 / 0.95 / 0.90
  final-layer rel_delta  > 0.25 / 0.50 / 1.00
  site-pooled delta cos  < 0.90 / 0.75 / 0.50
Intervals: 95% percentile bootstrap over proteins (2000 reps, seed 0);
statistics are means of per-protein means.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PREC = sys.argv[1] if len(sys.argv) > 1 else 'fp32'
D = f'/home/andrew/mutant_cache_scratch/am0/{PREC}/'
OUT = f'/home/andrew/mutant_cache/experiments/am0_{PREC}_'
V = '/home/andrew/mutant_cache_scratch/processed/esm_msr_mega/variants.parquet'
rng = np.random.default_rng(0)

rows = pd.read_parquet(D + 'rows.parquet')
fin = pd.read_parquet(D + 'final.parquet')
var = pd.read_parquet(D + 'variant.parquet')
ctx = pd.read_parquet(D + 'ctx.parquet')
dbl = pd.read_parquet(D + 'dbl.parquet')
man = pd.read_csv('/home/andrew/mutant_cache/experiments/am0_manifest.csv')
des = dict(zip(man.target_id, man.designed))
for x in (rows, fin, var, ctx, dbl):
    x['designed'] = x.target_id.map(des)
syn = rows[rows.tag == 'synthetic']
rows, fin, var = (x[x.tag == 'manifest'] for x in (rows, fin, var))
ctx, dbl = ctx[ctx.tag == 'manifest'], dbl[dbl.tag == 'manifest']


def boot(df, col, by=()):
    """mean of per-protein means, with protein bootstrap CI."""
    out = []
    for key, g in (df.groupby(list(by)) if by else [((), df)]):
        pm = g.groupby('target_id')[col].mean().values
        bs = rng.choice(pm, (2000, len(pm))).mean(1)
        out.append(dict(zip(by, key if isinstance(key, tuple) else (key,)),
                        stat=col, n_prot=len(pm), mean=pm.mean(),
                        lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                        prot_min=pm.min(), prot_max=pm.max()))
    return pd.DataFrame(out)


lines = []
P = lambda *a: lines.append(' '.join(str(x) for x in a))  # noqa: E731
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 200)

# per-layer profile
cols = ['cos', 'rel_l2', 'cos_delta', 'rel_delta', 'ratio_delta', 'rel_TW']
prof = pd.concat([boot(rows, c, ('n_mut', 'layer')) for c in cols])
prof.to_csv(OUT + 'layer_profile.csv', index=False)
wide = prof.pivot_table(index=['n_mut', 'layer'], columns='stat', values='mean')[cols]
P('== per-layer, mutated sites, mean of protein means (layer 30 = final norm) ==')
P(wide.round(4).to_string())

f30 = rows[rows.layer == 30]
P('\n== final layer, with protein-bootstrap 95% CI ==')
P(pd.concat([boot(f30, c, ('n_mut',)) for c in cols]).round(4).to_string(index=False))

P('\n== preregistered levels: fraction of mutated sites (final layer), mean over proteins ==')
for c, op, levels in (('cos', '<', (0.99, 0.95, 0.90)), ('rel_delta', '>', (0.25, 0.5, 1.0))):
    for lv in levels:
        f30 = f30.assign(_x=(f30[c] < lv) if op == '<' else (f30[c] > lv))
        b = boot(f30, '_x', ('n_mut',))
        for _, r in b.iterrows():
            P(f'  {c} {op} {lv}: n_mut={r.n_mut} frac={r["mean"]:.3f} [{r.lo:.3f},{r.hi:.3f}]')
for lv in (0.9, 0.75, 0.5):
    v2 = var.assign(_x=var.cos_sitepool_delta < lv)
    for _, r in boot(v2, '_x', ('n_mut',)).iterrows():
        P(f'  sitepool cos_delta < {lv}: n_mut={r.n_mut} frac={r["mean"]:.3f} [{r.lo:.3f},{r.hi:.3f}]')

P('\n== strata (final layer, singles+doubles) ==')
f30 = rows[rows.layer == 30].copy()
f30['loc'] = pd.cut(f30.rel_site, [0, 0.1, 0.9, 1.01], labels=['Nterm10%', 'middle', 'Cterm10%'])
f30['Lbin'] = pd.cut(f30.L, [0, 45, 60, 100])
f30['gap'] = pd.cut(f30.gap_other.where(f30.n_mut == 2), [0, 1, 4, 1000], labels=['adj', '2-4', '5+'])
for by in (['designed'], ['loc'], ['Lbin'], ['gap']):
    P(pd.concat([boot(f30.dropna(subset=by), c, by) for c in ('cos', 'cos_delta', 'rel_delta')])
      .round(4).to_string(index=False))

P('\n== variant-level pooled features (final layer) ==')
P(pd.concat([boot(var, c, ('n_mut',)) for c in
             ('cos_sitepool', 'rel_sitepool', 'cos_sitepool_delta', 'cos_seqpool',
              'cos_seqpool_delta', 'ratio_seqpool_delta')]).round(4).to_string(index=False))

P('\n== logits at mutated sites ==')
P(boot(fin, 'kl_T_S', ('n_mut',)).round(4).to_string(index=False))
# zero-shot LLR agreement and relation to measured score (train/val only)
sc = pd.read_parquet(V)[['score']]
sc['source_row'] = np.arange(len(sc))
fv = fin.groupby(['target_id', 'mutant', 'source_row', 'n_mut'])[['llr_S', 'llr_T', 'llr_W']].sum().reset_index()
fv = fv.merge(sc, on='source_row')
pp = []
for (tid, k), g in fv.groupby(['target_id', 'n_mut']):
    if len(g) < 10:
        continue
    pp.append(dict(target_id=tid, n_mut=k,
                   rho_S_T=spearmanr(g.llr_S, g.llr_T)[0],
                   rho_S_score=spearmanr(g.llr_S, g.score)[0],
                   rho_T_score=spearmanr(g.llr_T, g.score)[0],
                   rho_W_score=spearmanr(g.llr_W, g.score)[0]))
pp = pd.DataFrame(pp)
pp.to_csv(OUT + 'zeroshot_per_protein.csv', index=False)
P('zero-shot sum-LLR Spearman, per protein (S sparse, T teacher, W unmasked WT logits):')
for c in ('rho_S_T', 'rho_S_score', 'rho_T_score', 'rho_W_score'):
    P(boot(pp, c, ('n_mut',)).round(3).to_string(index=False))
pp['gap'] = pp.rho_T_score - pp.rho_S_score
P(boot(pp, 'gap', ('n_mut',)).round(3).to_string(index=False))

P('\n== non-mutated positions: teacher context change ||T-W||/||W|| (sparse keeps W) ==')
ctxm = ctx[ctx.layer.isin([0, 5, 10, 15, 20, 25, 29, 30])]
w = ctxm.groupby(['n_mut', 'band', 'layer']).apply(
    lambda g: np.average(g.rel_TW_mean, weights=g.n_pos), include_groups=False).unstack('layer')
P(w.round(4).to_string())
P('compare: mutated-site teacher change rel_TW, final layer:',
  rows[rows.layer == 30].groupby('n_mut').rel_TW.mean().round(4).to_dict())

if len(dbl):
    P('\n== doubles with both singles measured: non-additivity at each site ==')
    d = dbl.copy()
    d['gapb'] = pd.cut(d.gap, [0, 1, 4, 1000], labels=['adj', '2-4', '5+'])
    P('n doubles', d.mutant.nunique(), 'proteins', d.target_id.nunique())
    for c in ('nonadd_T', 'nonadd_S', 'cos_nonadd', 'rel_nonadd', 'cross_T'):
        b = boot(d[d.layer.isin([0, 10, 20, 29, 30])], c, ('layer',))
        P(b.round(4).to_string(index=False))
    P(pd.concat([boot(d[d.layer == 30], c, ('gapb',)) for c in
                 ('nonadd_T', 'nonadd_S', 'cos_nonadd', 'cross_T')]).round(4).to_string(index=False))

P('\n== synthetic panel (final layer) ==')
s30 = syn[syn.layer == 30].copy()
s30['panel'] = s30.groupby(['target_id', 'mutant']).ngroup()
P(s30.groupby(['target_id', 'n_mut', 'mutant'])[['cos', 'cos_delta', 'rel_delta']].mean().round(4).to_string())

open(OUT + 'summary.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
