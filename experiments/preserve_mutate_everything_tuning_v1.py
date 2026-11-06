"""Exact archive/stop step before the tracking continuation, 2026-11-04."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

from mutant_cache.tuning import alive, score_result, write_json

root = Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v1')
successor = Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v2_tracking')
assert not successor.exists(), 'continuation already exists; inspect it before restarting'
assert not (root/'STOPPED.json').exists(), 'original sweep was already stopped'
launch = json.loads((root/'launch.json').read_text())
groups = []
for worker in launch['workers']:
    pid = worker['pid']
    if alive(pid):
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        assert b'mutant_cache.tuning' in command and b'worker' in command and str(root).encode() in command
        assert os.getpgid(pid) == pid, 'unexpected worker process group'
        groups.append(pid)
assert groups, 'no original workers found; inspect the registry before proceeding'

stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
archive = root/'archives'/f'before_tracking_restart_{stamp}'
archive.mkdir(parents=True)
frozen = []
try:
    for group in groups:
        os.killpg(group, signal.SIGSTOP)
        frozen.append(group)
    state = json.loads((root/'registry.json').read_text())
    for name in ('registry.json', 'plan.json', 'source_sha256.json', 'data_provenance.json',
                 'launch.json', 'summary.csv', 'checkpoint_choices.csv', 'STATUS.md', 'frozen_recipe.json'):
        if (root/name).exists():
            shutil.copy2(root/name, archive/name)
    artifacts = []
    for trial in state['trials']:
        if not trial.get('run_dir'):
            continue
        directory = Path(trial['run_dir'])
        saved = archive/trial['id']
        saved.mkdir()
        for name in ('history.csv', 'progress.json', 'selection_checkpoints.json', 'training_protocol.json',
                     'resolved_experiment.json', 'results.json'):
            if (directory/name).exists():
                shutil.copy2(directory/name, saved/name)
        for path in sorted(directory.rglob('*.pt')):
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024*1024), b''):
                    digest.update(block)
            artifacts.append(dict(trial=trial['id'], path=str(path), bytes=path.stat().st_size,
                                  sha256=digest.hexdigest()))
        # A child can have finished before its worker records completion.
        if trial['status']=='running' and (directory/'results.json').exists() and (directory/'model.pt').exists():
            trial.update(status='completed', metrics=score_result(json.loads((directory/'results.json').read_text())))
    write_json(archive/'checkpoint_inventory.json', artifacts)
    write_json(archive/'reconciled_registry.json', state)
except BaseException:
    for group in frozen:
        os.killpg(group, signal.SIGCONT)
    raise

# Both group leaders and their training/compiler children are stopped together.
# Existing checkpoint files and logs stay in their original directories.
for group in frozen:
    os.killpg(group, signal.SIGTERM)
    os.killpg(group, signal.SIGCONT)
targets = groups + [t['child_pid'] for t in state['trials'] if t.get('child_pid') and t['status']=='running']
deadline = time.monotonic()+20
while any(alive(pid) for pid in targets) and time.monotonic()<deadline:
    time.sleep(.2)
for group in frozen:
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass
assert not any(alive(pid) for pid in targets), 'an original training process has not exited'
for trial in state['trials']:
    if trial['status']=='running':
        trial.update(status='interrupted', interruption_reason='user-requested restart with tracking',
                     interrupted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    elif trial['status']=='planned':
        trial.update(status='deferred', deferred_to=str(successor))
record = dict(stopped_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), archive=str(archive),
    successor=str(successor), stopped_process_groups=groups,
    completed=[t['id'] for t in state['trials'] if t['status']=='completed'],
    interrupted=[t['id'] for t in state['trials'] if t['status']=='interrupted'],
    checkpoints_preserved=len(artifacts), exact_optimizer_resume=False)
state['superseded_by'] = str(successor)
write_json(root/'registry.json', state)
write_json(root/'STOPPED.json', record)
old = (root/'STATUS.md').read_text()
(root/'STATUS.md').write_text(f'Archived: continued at `{successor}`. See STOPPED.json.\n\n'+old.replace('| running |','| interrupted |').replace('| planned |','| deferred |'))
print(json.dumps(record, indent=2))
