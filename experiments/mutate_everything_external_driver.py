"""GPU-0 breadth evaluation, then detached worker continues additive repeats."""
import json
import os
from pathlib import Path
import subprocess
import sys

scratch = Path('/home/andrew/mutant_cache_scratch')
repo = Path('/home/andrew/mutant_cache')
root = scratch / 'mutate_everything'
external = root / 'external'
baseline = external / 'baseline'
baseline.mkdir(exist_ok=True)
py = '/home/andrew/anaconda3/envs/mutant_cache/bin/python'
fair = str(root / 'venv/bin/python')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4',
           PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1')
tags = ['s669', 'ptmul', 'cdna1', 's669_reverse']
# cDNA2 has the existing dense oracle, so first verify real-weight parity.
jobs = [('cdna2_oracle', scratch / 'processed/mutate_everything_cdna/comparison_labels_test.csv')]
jobs += [(tag, external / f'{tag}_labels.csv') for tag in tags]
for tag, labels in jobs:
    dest = baseline / f'{tag}_predictions.csv'
    if dest.exists():
        continue
    print('BASELINE START', tag, flush=True)
    subprocess.run([fair, str(repo / 'experiments/mutate_everything_predict.py'),
        '--repo', str(root / 'repo'), '--labels', str(labels), '--predictions', str(dest),
        '--resume', str(root / 'mutate_everything_esm.pth'), '--device', 'cuda:0',
        '--checkpoint-only-backbone', '--requested-decoding'], env=env, check=True)
    print('BASELINE COMPLETE', tag, flush=True)
pilot = scratch / 'runs/mutate_everything_esm2_pilot_20261104/joint_s0/model.pt'
subprocess.run([py, '-m', 'mutant_cache.benchmark', '--checkpoint', str(pilot),
    '--labels', *[str(external / f'{tag}_labels.csv') for tag in tags],
    '--output', str(external / 'ours_joint_s0'), '--device', 'cuda:0'], env=env, check=True)
subprocess.run([py, str(repo / 'experiments/mutate_everything_external_report.py')], env=env, check=True)
(external / 'evaluation_complete.json').write_text(json.dumps(dict(tags=tags, status='complete'), indent=2))
print('CORE EXTERNAL EVALUATION COMPLETE; STARTING ADDITIVE REPEATS', flush=True)
env['ME_ARM'] = 'additive'
subprocess.run([py, str(repo / 'experiments/mutate_everything_replicas.py')], env=env, check=True)
