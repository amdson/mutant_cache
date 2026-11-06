"""Summarize smalln_exact_vs_cache.sh. Raw script."""
import glob, json
import pandas as pd

OUT = '/home/andrew/mutant_cache_scratch/runs/smalln_exact_vs_cache/'
rows = []
for f in glob.glob(OUT + '*_n*_s*/results.json'):
    r = json.load(open(f))
    tag = f.split('/')[-2]
    be, n, s = tag.split('_')
    vs = {(d['depth'], d['metric']): d['macro'] for d in r['extra']['val_summary']}
    t = r['extra']['timing']
    rows.append(dict(backend=be, n=int(n[1:]), seed=int(s[1:]),
                     single_rho=vs[('single', 'spearman')], double_rho=vs[('double', 'spearman')],
                     eps_rho=vs[('epistasis', 'spearman')], single_rmse=vs[('single', 'rmse')],
                     train_s=t['train_s'], total_s=t['cold_total_s']))
d = pd.DataFrame(rows).sort_values(['n', 'backend', 'seed'])
d.to_csv('/home/andrew/mutant_cache/experiments/smalln_exact_vs_cache_runs.csv', index=False)
pd.set_option('display.width', 200)
print(d.round(3).to_string(index=False))
m = d.groupby(['n', 'backend'])[['single_rho', 'double_rho', 'eps_rho', 'single_rmse', 'train_s']].mean()
print('\nmean over seeds'); print(m.round(3).to_string())
p = d.pivot_table(index=['n', 'seed'], columns='backend',
                  values=['single_rho', 'double_rho', 'eps_rho'])
diff = pd.DataFrame({c: p[(c, 'exact')] - p[(c, 'cache')] for c in ('single_rho', 'double_rho', 'eps_rho')})
print('\nexact - cache, paired by seed'); print(diff.round(3).to_string())
print('\nmean exact - cache'); print(diff.groupby('n').mean().round(3).to_string())
