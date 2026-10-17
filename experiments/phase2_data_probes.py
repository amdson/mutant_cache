"""Phase 2 ad-hoc probes, exactly as run while building the data layer.

Four questions that each changed the loaders. Run with
/home/andrew/anaconda3/envs/mutant_cache/bin/python.
"""

import re
from collections import Counter

import polars as pl

# ---------------------------------------------------------------------------
# 1. Which Tsuboyama file is the curated one?
#    DATASETS.md says Dataset1. Dataset1 has no aa_seq / mut_type / WT_name.
# ---------------------------------------------------------------------------
D = '/home/andrew/mutant_cache_scratch/interim/tsuboyama/Processed_K50_dG_datasets'
print(open(f'{D}/Tsuboyama2023_Dataset1_20230416.csv').readline().strip().split(','))
print(open(f'{D}/Tsuboyama2023_Dataset2_Dataset3_20230416.csv').readline().strip().split(','))

# ---------------------------------------------------------------------------
# 2. Is Tsuboyama's mut_type <wt><pos><alt> or <alt><pos><wt>?
#    Answer: <wt><pos><alt>, 100% of a 20k sample. I had guessed wrong.
# ---------------------------------------------------------------------------
P = f'{D}/Tsuboyama2023_Dataset2_Dataset3_20230416.csv'
d = pl.scan_csv(P, infer_schema_length=100000).select(['WT_name', 'aa_seq', 'mut_type']).collect()
wt = d.filter(pl.col('mut_type') == 'wt').group_by('WT_name').agg(pl.col('aa_seq').first()).to_dict(as_series=False)
seqs = dict(zip(wt['WT_name'], wt['aa_seq']))
SUB = re.compile(r'^([A-Z])(\d+)([A-Z])$')
rows = d.filter(pl.col('mut_type').str.contains(r'^[A-Z]\d+[A-Z]$')).sample(20000, seed=0).to_dicts()
a = b = n = 0
for r in rows:
    s = seqs.get(r['WT_name'])
    if not s:
        continue
    m = SUB.fullmatch(r['mut_type'])
    p = int(m.group(2))
    if p > len(s):
        continue
    n += 1
    a += s[p - 1] == m.group(1)
    b += s[p - 1] == m.group(3)
print(f'n={n}  first-letter-is-WT: {a} ({100*a/n:.2f}%)   third-letter-is-WT: {b} ({100*b/n:.2f}%)')

# ---------------------------------------------------------------------------
# 3. MaveDB WT mismatch: spread out, or concentrated?
#    Answer: 20 of 2,455 targets; one of them (PSD95 PDZ3) is 1.24M of the
#    1.29M bad substitutions and a -1 offset fixes it exactly.
# ---------------------------------------------------------------------------
from mutant_cache.data.mutations import parse_mutant, suggest_offsets
from mutant_cache.data.schema import load_targets, load_variants

v = load_variants('mavedb')
t = load_targets('mavedb')
seqs = dict(zip(t['target_id'], t['sequence']))
bad, tot = Counter(), Counter()
for m, tid in zip(v['mutant'].to_list(), v['target_id'].to_list()):
    s = seqs[tid]
    for p, w, _ in parse_mutant(m):
        tot[tid] += 1
        if p > len(s) or s[p - 1] != w:
            bad[tid] += 1
T, B = sum(tot.values()), sum(bad.values())
print(f'targets with any mismatch: {len(bad)} of {len(tot)}; subs {T:,} bad {B:,} {100*B/T:.2f}%')
for tid, n in bad.most_common(12):
    print(f'  {n:>9,}  {100*n/tot[tid]:5.1f}%  {tid}')
offs = suggest_offsets(v['mutant'].to_list(), v['target_id'].to_list(), seqs)
fixed = sum(bad[x] for x in offs)
after = B - fixed + sum(offs[x][2] * tot[x] for x in offs)
print(f'offset-explained: {len(offs)} targets, {fixed:,}/{B:,} bad subs; '
      f'applying them -> {100*after/T:.2f}% overall')

# ---------------------------------------------------------------------------
# 4. Why is mean epsilon +0.89 instead of 0?
#    Answer: assay dynamic-range compression. Monotone in the additive
#    prediction, and it survives dropping the saturated rows, so it is not
#    hard clipping. Tsuboyama needs a monotone link too.
# ---------------------------------------------------------------------------
from mutant_cache.data.epsilon import load, noise_ceiling

e = load().with_columns((pl.col('ddg_a') + pl.col('ddg_b')).alias('sum_singles'))
print('corr(eps, sum_singles)', e.select(pl.corr('epsilon', 'sum_singles')).item())
bb = e.with_columns(pl.col('sum_singles').round(0).clip(-7, 2).alias('bin'))
print(bb.group_by('bin').agg(pl.len(),
                             pl.col('epsilon').mean().round(3).alias('mean_eps'),
                             pl.col('ddg_double').min().round(3).alias('min_dd')).sort('bin'))
print('frac ddg_double < -4.8:', round(float((e['ddg_double'] < -4.8).mean()), 4))
ns = e.filter((pl.col('ddg_double') > -4.0) & (pl.col('ddg_a') > -4.0)
              & (pl.col('ddg_b') > -4.0) & (pl.col('sum_singles') > -4.0))
nc = noise_ceiling(ns)
print('non-saturated n=%d mean=%.3f sd=%.3f r_max=%.3f' % (nc['n'], nc['mean'], nc['sd'], nc['r_max']))
