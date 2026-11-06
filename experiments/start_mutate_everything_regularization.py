"""Raw launch of the independent regularization screen on free GPU 2."""
import datetime
import json
from pathlib import Path
import subprocess

repo = Path('/home/andrew/mutant_cache')
root = Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_regularization_v1')
py = '/home/andrew/anaconda3/envs/mutant_cache/bin/python'
template = '/home/andrew/mutant_cache_scratch/runs/mutate_everything_esm2_pilot_20261104/joint_s0/model.pt'
subprocess.run([py, '-m', 'mutant_cache.tuning', 'init', '--root', str(root),
    '--plan', str(repo/'experiments/mutate_everything_regularization/plan.json'),
    '--template', template, '--tracking-config', str(repo/'experiments/wandb_tracking.json')],
    cwd=repo, check=True)
with (root/'worker_gpu2.log').open('xb') as log:
    process = subprocess.Popen([py, '-m', 'mutant_cache.tuning', 'worker', '--root', str(root), '--gpu', '2'],
        cwd=root/'source_snapshot', stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
record = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    workers=[dict(physical_gpu=2, pid=process.pid)], run_root=str(root),
    plan=str(repo/'experiments/mutate_everything_regularization/plan.json'))
(root/'launch.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
