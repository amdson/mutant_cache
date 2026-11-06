"""Raw launch script: continue tuning v1 as v2_tracking (W&B + diagnostics), GPU 0 only (GPU 2 kept free)."""
import datetime
import json
from pathlib import Path
import subprocess

repo = Path('/home/andrew/mutant_cache')
source = Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1')
root = Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v2_tracking')
py = '/home/andrew/anaconda3/envs/mutant_cache/bin/python'
subprocess.run([py,'-m','mutant_cache.tuning','continue','--root',str(root),
    '--source-root',str(source),'--tracking-config',str(repo/'experiments/wandb_tracking.json')],
    cwd=repo,check=True)
workers = []
for gpu in [0]:
    with (root/f'worker_gpu{gpu}.log').open('xb') as log:
        process = subprocess.Popen([py,'-m','mutant_cache.tuning','worker','--root',str(root),'--gpu',str(gpu)],
            cwd=root/'source_snapshot',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    workers.append(dict(physical_gpu=gpu,pid=process.pid))
record = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=workers,
    run_root=str(root),source_root=str(source),plan=str(repo/'experiments/mutate_everything_tuning/plan.json'),
    status=str(repo/'experiments/mutate_everything_tuning/STATUS.md'))
(root/'launch.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
