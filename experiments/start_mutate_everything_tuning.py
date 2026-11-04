"""Raw launch script for systematic tuning v1; run once after protocol checks."""
import datetime
import json
from pathlib import Path
import subprocess

repo = Path('/home/andrew/mutant_cache')
root = Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1')
py = '/home/andrew/anaconda3/envs/mutant_cache/bin/python'
template = '/home/andrew/mutant_cache_scratch/runs/mutate_everything_esm2_pilot_20261104/joint_s0/model.pt'
subprocess.run([py,'-m','mutant_cache.tuning','init','--root',str(root),
    '--plan',str(repo/'experiments/mutate_everything_tuning/plan.json'),
    '--template',template],check=True)
workers = []
for gpu in [0,2]:
    with (root/f'worker_gpu{gpu}.log').open('xb') as log:
        process = subprocess.Popen([py,'-m','mutant_cache.tuning','worker','--root',str(root),'--gpu',str(gpu)],
            cwd=root/'source_snapshot',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    workers.append(dict(physical_gpu=gpu,pid=process.pid))
record = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=workers,
    run_root=str(root),plan=str(repo/'experiments/mutate_everything_tuning/plan.json'),
    status=str(repo/'experiments/mutate_everything_tuning/STATUS.md'))
(root/'launch.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
