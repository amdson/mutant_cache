"""Raw launch: freeze E02/ESM-MSR three-seed test; wait for accuracy-v2 GPUs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

repo = Path('/home/andrew/mutant_cache')
scratch = Path(os.environ.get('MUTANT_CACHE_SCRATCH', '/home/andrew/mutant_cache_scratch'))
root = scratch/'runs/esm_msr_e02_test_v1'
prior = scratch/'runs/mutate_everything_accuracy_v2_diagnostics_fix'
py = '/home/andrew/anaconda3/envs/mutant_cache/bin/python'
study = Path('experiments/stability_paper/msr_esm2_test_v1')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

if '--coordinate' in sys.argv:
    launch = json.loads((root/'launch.json').read_text())
    pid = launch['waiting_for_pid']
    while True:
        try:
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
            live = str(prior).encode() in cmd
        except FileNotFoundError:
            live = False
        if not live:
            break
        time.sleep(30)
    # A terminated coordinator may have failed; require successful follow-up.
    for directory in [prior, Path(str(prior)+'_holdout1'), Path(str(prior)+'_holdout2')]:
        complete = json.loads((directory/'complete.json').read_text())
        if complete.get('failed'):
            raise RuntimeError(f'dependency failed: {directory}')
    for path, digest in json.loads((root/'data_sha256.json').read_text()).items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f'frozen dataset changed: {path}')
    (root/'RUNNING.json').write_text(json.dumps({'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}))
    subprocess.run([py, '-m', 'mutant_cache.queue_runner', 'run', str(study/'jobs.json'),
                    '--root', str(root), '--gpus', '0,2'], cwd=root/'source_snapshot', check=True)
    # Report every seed; no best-seed selection.
    import pandas as pd
    rows = []
    for seed in [0, 1, 2]:
        run = root/f'runs/msr_esm2_test_v1_s{seed}'
        result = json.loads((run/'results.json').read_text())
        if result['extra'].get('test_summary') is None:
            raise RuntimeError(f'missing final test metrics: {run}')
        summary = result['extra']['test_summary']
        row = {'seed': seed, 'run': str(run)}
        for metric in summary['summary']:
            row[f"{metric['depth']}_{metric['metric']}"] = metric['macro']
        for metric in summary['retrieval']:
            for key, value in metric.items():
                if key != 'subset':
                    row[f"retrieval_{metric['subset']}_{key}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(root/'test_summary_by_seed.csv', index=False)
    (root/'COMPLETE.json').write_text(json.dumps({'seeds': [0, 1, 2], 'comparator_reproduced': False}))
else:
    if root.exists():
        raise RuntimeError('refusing to overwrite frozen study')
    prior_launch = json.loads((prior/'restart_launch.json').read_text())
    root.mkdir(parents=True)
    snapshot = root/'source_snapshot'
    shutil.copytree(repo/'mutant_cache', snapshot/'mutant_cache', ignore=shutil.ignore_patterns('__pycache__'))
    for relative in [study, Path('datasets/protocols')]:
        shutil.copytree(repo/relative, snapshot/relative)
    shutil.copy2(__file__, snapshot/'launch.py')
    hashes = {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}
    (root/'source_sha256.json').write_text(json.dumps(hashes, indent=2))
    data = scratch/'processed/esm_msr_mega'
    paths = [data/'variants.parquet', data/'targets.parquet', data/'splits/esm_msr_hyperopt.parquet']
    (root/'data_sha256.json').write_text(json.dumps({str(p): sha(p) for p in paths}, indent=2))
    env = dict(os.environ, PYTHONPATH=str(snapshot), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1')
    subprocess.run([py, '-m', 'mutant_cache.queue_runner', 'run', str(study/'jobs.json'),
                    '--root', str(root), '--gpus', '0,2', '--dry-run'], cwd=snapshot, env=env, check=True)
    record = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'waiting_for_pid': prior_launch['coordinator_pid'], 'dependency': str(prior),
              'root': str(root), 'protocol': 'frozen E02; validation selection; released test; all three seeds'}
    (root/'launch.json').write_text(json.dumps(record, indent=2))
    with (root/'coordinator.log').open('xb') as log:
        proc = subprocess.Popen([py, str(snapshot/'launch.py'), '--coordinate'], cwd=snapshot,
                                env=env, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    record['coordinator_pid'] = proc.pid
    (root/'launch.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
