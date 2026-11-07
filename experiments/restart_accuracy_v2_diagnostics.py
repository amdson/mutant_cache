"""Raw launch: preserve completed fits, retry I01 with corrected diagnostics."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from mutant_cache.tuning import initialize_continuation, locked, refresh

repo = Path('/home/andrew/mutant_cache')
old = Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2')
root = old.with_name(old.name + '_diagnostics_fix')
py = '/home/andrew/anaconda3/envs/mutant_cache/bin/python'

if '--coordinate' in sys.argv:
    def workers(label):
        processes = []
        for gpu in [0, 2]:
            with (root / f'{label}_gpu{gpu}.log').open('xb') as log:
                processes.append(subprocess.Popen(
                    [py, '-m', 'mutant_cache.tuning', 'worker', '--root', str(root), '--gpu', str(gpu)],
                    cwd=root/'source_snapshot', stdout=log, stderr=subprocess.STDOUT))
        if any([p.wait() for p in processes]):
            raise RuntimeError('worker exited unsuccessfully')

    workers('retry')
    state = json.loads((root/'registry.json').read_text())
    if any(t['status'] == 'failed' for t in state['trials']):
        raise RuntimeError('retry failed; inspect before holdouts')
    # Imported selected confirmations belong to S02. Refit them if I01 wins.
    changed = False
    with locked(root) as state:
        for trial in state['trials']:
            if trial.get('role') == 'selected' and trial.get('parent_id') != state['selected_recipe']:
                prior = dict(trial)
                trial.clear()
                trial.update(id=prior['id'], stage='G', role='selected', seed=prior['seed'],
                             question=prior['question'], status='planned', attempts=0,
                             prior_attempts=[prior])
                changed = True
    if changed:
        (root/'complete.json').unlink(missing_ok=True)
        refresh(root)
        workers('confirmation')
    subprocess.run([py, str(root/'source_snapshot/holdouts.py')],
                   cwd=root/'source_snapshot', check=True)
else:
    if root.exists():
        raise RuntimeError('refusing to overwrite continuation')
    # The primary queue has terminated, including its failed holdout coordinator.
    processes = subprocess.check_output(['ps', '-eo', 'args'], text=True)
    if any(str(old) in line and ('mutant_cache.tuning' in line or 'holdouts.py' in line)
           for line in processes.splitlines()):
        raise RuntimeError('source queue still has live processes')
    (old/'STOPPED.json').write_text(json.dumps(dict(
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reason='Queue terminated with I01 diagnostics failure; continuation preserves all artifacts.'), indent=2))
    initialize_continuation(root, old, old/'tracking_config.json')
    coordinator = root/'source_snapshot/restart.py'
    shutil.copy2(__file__, coordinator)
    holdouts = (repo/'experiments/mutate_everything_accuracy_v2_holdouts.py').read_text()
    (root/'source_snapshot/holdouts.py').write_text(holdouts.replace(str(old), str(root)))
    with locked(root) as state:
        state['continuation']['restarted'] = ['I01']
        (root/'continuation.json').write_text(json.dumps(state['continuation'], indent=2))
    env = dict(os.environ, PYTHONPATH=str(root/'source_snapshot'),
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4')
    with (root/'restart_coordinator.log').open('xb') as log:
        proc = subprocess.Popen([py, str(coordinator), '--coordinate'],
                                cwd=root/'source_snapshot', env=env,
                                stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(coordinator_pid=proc.pid, root=str(root),
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (root/'restart_launch.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
