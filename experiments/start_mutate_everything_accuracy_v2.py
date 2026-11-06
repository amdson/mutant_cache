"""Raw launch of the E02 accuracy screen plus automatic extra protein holdouts."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import pandas as pd
import numpy as np
from mutant_cache.tuning import initialize

repo=Path('/home/andrew/mutant_cache')
root=Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2')
py='/home/andrew/anaconda3/envs/mutant_cache/bin/python'
if root.exists(): raise RuntimeError('refusing to overwrite accuracy-v2')
data=Path('/home/andrew/mutant_cache_scratch/processed/mutate_everything_cdna')
sp=pd.read_parquet(data/'splits/mutate_everything_cdna.parquet')
targets=pd.read_parquet(data/'targets.parquet').set_index('target_id').sequence.to_dict()
dev=sp.loc[sp['split']!='test','target_id']
for seed in [1,2]:
 sequences=sorted({targets[t] for t in dev})
 np.random.default_rng(seed).shuffle(sequences)
 val=set(sequences[:round(len(sequences)*.1)])
 extra=sp.copy()
 extra.loc[extra['split']!='test','split']=['val' if targets[t] in val else 'train' for t in dev]
 path=data/'splits'/f'mutate_everything_cdna_holdout{seed}.parquet'
 if path.exists():assert pd.read_parquet(path).equals(extra)
 else:extra.to_parquet(path,index=False)
 assert set(extra.loc[extra['split']=='test','target_id'])==set(sp.loc[sp['split']=='test','target_id'])
template=Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v2_tracking/runs/E02/attempt_1/model.pt')
initialize(root,repo/'experiments/mutate_everything_accuracy_v2/plan.json',template,repo/'experiments/wandb_tracking.json')
shutil.copy2(repo/'experiments/mutate_everything_accuracy_v2_holdouts.py',root/'source_snapshot/holdouts.py')
env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONPATH=str(root/'source_snapshot'))
workers=[]
for gpu in [0,2]:
 with (root/f'worker_gpu{gpu}.log').open('xb') as log:
  p=subprocess.Popen([py,'-m','mutant_cache.tuning','worker','--root',str(root),'--gpu',str(gpu)],cwd=root/'source_snapshot',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 workers.append(dict(gpu=gpu,pid=p.pid))
with (root/'holdout_coordinator.log').open('xb') as log:
 p=subprocess.Popen([py,str(root/'source_snapshot/holdouts.py')],cwd=root/'source_snapshot',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=workers,holdout_coordinator_pid=p.pid)
(root/'launch.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
