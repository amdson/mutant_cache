"""Raw frozen seven-protein pilot; first BCHB, then six without outcome selection."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

repo=Path('/home/andrew/mutant_cache')
scratch=Path('/home/andrew/mutant_cache_scratch')
root=scratch/'runs/stabilityarc_pilot_v1'
prior=scratch/'runs/mutate_everything_accuracy_v2_diagnostics_fix'
py='/home/andrew/anaconda3/envs/mutant_cache/bin/python'
study=Path('experiments/stability_paper/stabilityarc_pilot_v1')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def compare(fold):
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    run=root/'runs'/fold['heldout']
    predictions=pd.read_csv(run/'predictions_test.csv')
    labels=pd.read_parquet(scratch/'processed'/fold['dataset']/'variants.parquet')
    labels=labels[labels.target_id==fold['heldout']]
    if predictions.mutant.duplicated().any() or set(predictions.target_id)!={fold['heldout']}:
        raise RuntimeError('duplicate rows or unexpected test target')
    joined=labels.merge(predictions,on=['target_id','mutant'],validate='one_to_one',suffixes=('_expected','_pred'),how='outer',indicator=True)
    if len(joined)!=fold['test_rows'] or not (joined['_merge']=='both').all():
        raise RuntimeError('incomplete test coverage')
    if not np.isfinite(joined.pred).all() or not np.allclose(joined.score_expected,joined.score_pred,rtol=0,atol=1e-12):
        raise RuntimeError('nonfinite prediction or changed test labels')
    rows=[]
    for subset,depth,key in [('all',None,'all_variant_spearman'),('single',1,'single_mutant_spearman'),('double',2,'multi_mutant_spearman')]:
        frame=predictions if depth is None else predictions[predictions.n_mut==depth]
        rho=float(spearmanr(frame.score,frame.pred).statistic) if len(frame)>1 else None
        baseline=fold['published'][key]
        rows.append(dict(protein=fold['heldout'],subset=subset,n=len(frame),our_spearman=rho,
                         published_stabilityarc_spearman=baseline,
                         difference=None if rho is None or baseline is None else rho-baseline))
    (run/'comparison.json').write_text(json.dumps(rows,indent=2))
    return rows

if '--coordinate' in sys.argv:
    pid=json.loads((prior/'restart_launch.json').read_text())['coordinator_pid']
    while True:
        try: live=str(prior).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError: live=False
        if not live: break
        time.sleep(30)
    for dependency in [prior,Path(str(prior)+'_holdout1'),Path(str(prior)+'_holdout2')]:
        if json.loads((dependency/'complete.json').read_text()).get('failed'):
            raise RuntimeError(f'dependency failed: {dependency}')
    for path,digest in json.loads((root/'data_sha256.json').read_text()).items():
        if sha(path)!=digest: raise RuntimeError(f'data changed: {path}')
    manifest=json.loads((root/'manifest.json').read_text())
    for jobs in ['pilot_jobs.json','remaining_jobs.json']:
        subprocess.run([py,'-m','mutant_cache.queue_runner','run',str(study/jobs),
                        '--root',str(root),'--gpus','0,2'],cwd=root/'source_snapshot',check=True)
        completed=[f for f in manifest['folds'] if (root/'runs'/f['heldout']/'results.json').exists()]
        rows=[r for fold in completed for r in compare(fold)]
        import pandas as pd
        pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
        (root/'STATUS.json').write_text(json.dumps(dict(completed=[f['heldout'] for f in completed],
            comparison='selected seven-assay pilot; public count reconciliation, not exact private packed-data reproduction'),indent=2))
    (root/'COMPLETE.json').write_text(json.dumps({'completed':True,'folds':7,'seed':0}))
else:
    if root.exists(): raise RuntimeError('refusing to overwrite pilot')
    root.mkdir(parents=True)
    snapshot=root/'source_snapshot'
    shutil.copytree(repo/'mutant_cache',snapshot/'mutant_cache',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(repo/study,snapshot/study)
    shutil.copy2(__file__,snapshot/'launch.py')
    manifest=scratch/'stabilityarc/pilot_v1/manifest.json'
    shutil.copy2(manifest,root/'manifest.json')
    folds=json.loads(manifest.read_text())['folds']
    paths=[p for f in folds for p in (scratch/'processed'/f['dataset']).rglob('*') if p.is_file()]
    (root/'data_sha256.json').write_text(json.dumps({str(p):sha(p) for p in paths},indent=2))
    (root/'source_sha256.json').write_text(json.dumps({str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()},indent=2))
    env=dict(os.environ,PYTHONPATH=str(snapshot),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
    for jobs in ['pilot_jobs.json','remaining_jobs.json']:
        subprocess.run([py,'-m','mutant_cache.queue_runner','run',str(study/jobs),
                       '--root',str(root),'--gpus','0,2','--dry-run'],cwd=snapshot,env=env,check=True)
    with (root/'coordinator.log').open('xb') as log:
        proc=subprocess.Popen([py,str(snapshot/'launch.py'),'--coordinate'],cwd=snapshot,env=env,
                              stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    record=dict(coordinator_pid=proc.pid,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                dependency=str(prior),root=str(root),first='BCHB_CHLTE',remaining=6)
    (root/'launch.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
