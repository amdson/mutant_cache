"""Raw queue takeover: retain active fits; add loss logging to pending fits."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

repo=Path('/home/andrew/mutant_cache')
root=Path('/home/andrew/mutant_cache_scratch/runs/stabilityarc_pilot_v1')
py='/home/andrew/anaconda3/envs/mutant_cache/bin/python'
snapshot=root/'assay_logging_snapshot'
shutil.copytree(root/'source_snapshot',snapshot,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['diagnostics.py','megascale.py']:
    shutil.copy2(repo/'mutant_cache'/name,snapshot/'mutant_cache'/name)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
(root/'assay_logging_source_sha256.json').write_text(json.dumps({str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()},indent=2))
old=json.loads((root/'concurrent_launch.json').read_text())['coordinator_pid']
cmd=Path(f'/proc/{old}/cmdline').read_bytes()
if str(root/'concurrent_coordinator.py').encode() not in cmd:raise RuntimeError('coordinator identity changed')
children=Path(f'/proc/{old}/task/{old}/children').read_text().split()
queues=[]
for pid in children:
    command=Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'mutant_cache.queue_runner' in command and str(root).encode() in command:queues.append(int(pid))
if len(queues)!=1:raise RuntimeError(f'expected one queue supervisor, got {queues}')
state=json.loads((root/'queue_state.json').read_text())
preserved={name:e['pid'] for name,e in state['jobs'].items() if e.get('status')=='running'}
source=(root/'concurrent_coordinator.py').read_text().replace("root/'source_snapshot'","root/'assay_logging_snapshot'")
coordinator=root/'assay_logging_coordinator.py'
with coordinator.open('x') as f:f.write(source)
# Stop only supervisors. Queue workers have their own sessions and survive.
os.kill(old,signal.SIGTERM)
for pid in queues:os.kill(pid,signal.SIGTERM)
for _ in range(50):
    if all(not Path(f'/proc/{pid}').exists() or Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z' for pid in queues):break
    time.sleep(.1)
else:raise RuntimeError('old queue supervisor did not exit')
for pid in preserved.values():os.kill(pid,0)
env=dict(os.environ,PYTHONPATH=str(snapshot),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
with (root/'assay_logging_coordinator.log').open('xb') as log:
    proc=subprocess.Popen([py,str(coordinator),'--coordinate'],cwd=snapshot,env=env,
                          stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),coordinator_pid=proc.pid,
            preserved_running=preserved,logging_snapshot=str(snapshot),
            scientific_configuration_changed=False,metrics='train_assay/<assay>/loss and window_rows every diagnostic logging window')
(root/'assay_logging_launch.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
