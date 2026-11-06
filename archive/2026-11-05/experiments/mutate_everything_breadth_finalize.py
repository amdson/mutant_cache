"""Wait for the five frozen-recipe repeats, score them, and summarize all seeds."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import pandas as pd
from mutant_cache.comparison import compare
from mutant_cache.eval.mutate_everything import evaluate

scratch = Path('/home/andrew/mutant_cache_scratch')
root = scratch / 'runs/mutate_everything_breadth_20261104'
external = scratch / 'mutate_everything/external'
pilot = scratch / 'runs/mutate_everything_esm2_pilot_20261104'
status = root / 'finalizer_status.json'
py = '/home/andrew/anaconda3/envs/mutant_cache/bin/python'
jobs = [('joint', 1), ('joint', 2)] + [('additive', s) for s in range(3)]
tags = ['s669','ptmul','cdna1','s669_reverse','cdna2']
labels = [external/f'{t}_labels.csv' for t in tags[:-1]]
labels.append(scratch/'processed/mutate_everything_cdna/comparison_labels_test.csv')
def update(phase, **kwargs):
    status.write_text(json.dumps(dict(phase=phase, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kwargs),indent=2))
try:
    for _ in range(720):
        pending = [f'{a}_s{s}' for a,s in jobs if not (root/f'{a}_s{s}/model.pt').exists()]
        update('waiting_for_training',pending=pending)
        if not pending: break
        time.sleep(30)
    else:
        raise RuntimeError('training did not finish within six hours')
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1')
    for arm, seed in jobs:
        dest = root/f'eval_{arm}_s{seed}'
        if (dest/'provenance.json').exists(): continue
        update('scoring',arm=arm,seed=seed)
        subprocess.run([py,'-m','mutant_cache.benchmark','--checkpoint',str(root/f'{arm}_s{seed}/model.pt'),
            '--labels',*[str(p) for p in labels],'--output',str(dest),'--device','cuda:0'],env=env,check=True)
    rows = []
    for tag in tags:
        baseline = (scratch/'mutate_everything/baseline/predictions_cdna2.csv' if tag=='cdna2'
                    else external/f'baseline/{tag}_single_only_predictions.csv' if tag=='s669'
                    else external/f'baseline/{tag}_predictions.csv')
        theirs = pd.read_csv(baseline)
        for arm in ['joint','additive']:
            fits = []
            for seed in range(3):
                if arm=='joint' and seed==0:
                    path = (pilot/'joint_s0_cdna2_fixed/predictions_test.csv' if tag=='cdna2' else
                            external/f'ours_joint_s0/{tag}_labels_predictions.csv')
                else:
                    stem = 'comparison_labels_test' if tag=='cdna2' else f'{tag}_labels'
                    path = root/f'eval_{arm}_s{seed}/{stem}_predictions.csv'
                f = pd.read_csv(path)
                fits.append((seed,f))
                for r in evaluate(f)['table'].to_dict('records'):
                    rows.append(dict(benchmark=tag,arm=arm,seed=seed,**r))
            expected = {'all':{'ndcg30':0.38}} if tag=='cdna2' else ({'all':{'spearman':0.47}} if tag=='s669' else None)
            compare(theirs,fits,root/f'reports/{tag}_{arm}',expected=expected,
                    expected_source='https://github.com/jozhang97/MutateEverything#main-results' if expected else None)
    pd.DataFrame(rows).to_csv(root/'all_seed_metrics.csv',index=False)
    allrows = pd.DataFrame(rows).query("depth == 'all'")
    summary = allrows.groupby(['benchmark','arm']).agg(spearman_mean=('spearman','mean'),
        spearman_sd=('spearman','std'),rmse_mean=('rmse','mean'),ndcg30_mean=('ndcg30','mean')).reset_index()
    summary.to_csv(root/'summary.csv',index=False)
    update('complete',summary=str(root/'summary.csv'))
    print(summary.to_string(index=False),flush=True)
except Exception as exc:
    update('failed',error=str(exc),traceback=traceback.format_exc())
    raise
