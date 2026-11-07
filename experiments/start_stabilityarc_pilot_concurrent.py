"""Raw operational restart: user authorized sharing GPUs with accuracy jobs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

root=Path('/home/andrew/mutant_cache_scratch/runs/stabilityarc_pilot_v1')
py='/home/andrew/anaconda3/envs/mutant_cache/bin/python'
record=json.loads((root/'launch.json').read_text())
pid=record['coordinator_pid']
command=Path(f'/proc/{pid}/cmdline').read_bytes()
if str(root/'source_snapshot/launch.py').encode() not in command:
    raise RuntimeError('waiting coordinator identity changed')
if (root/'queue_state.json').exists():
    raise RuntimeError('queue already started; inspect before replacing coordinator')
source=(root/'source_snapshot/launch.py').read_text()
start=source.index("    pid=json.loads((prior/'restart_launch.json')")
end=source.index('    for path,digest in ',start)
concurrent=source[:start]+"    # User authorized concurrent GPU sharing; dependency wait removed.\n"+source[end:]
path=root/'concurrent_coordinator.py'
with path.open('x') as f:f.write(concurrent)
os.kill(pid,signal.SIGTERM)
env=dict(os.environ,PYTHONPATH=str(root/'source_snapshot'),PYTHONUNBUFFERED='1',
         OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
with (root/'concurrent_coordinator.log').open('xb') as log:
    proc=subprocess.Popen([py,str(path),'--coordinate'],cwd=root/'source_snapshot',env=env,
                          stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
update=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            old_coordinator_pid=pid,coordinator_pid=proc.pid,
            authorization='User requested more concurrent GPU jobs; share GPUs 0/2 with accuracy holdouts.',
            coordinator_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scientific_configuration_changed=False,first='BCHB_CHLTE',remaining=6)
(root/'concurrent_launch.json').write_text(json.dumps(update,indent=2))
print(json.dumps(update,indent=2))
