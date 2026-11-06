"""Snapshot and detach the first training pilot. Run once, after the smoke."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

repo = Path('/home/andrew/mutant_cache')
out = Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_esm2_pilot_20261104')
out.mkdir(parents=True, exist_ok=True)
if (out / 'launch.json').exists():
    raise RuntimeError('pilot has already been launched; inspect it before starting another')
snapshot = out / 'source_snapshot'
snapshot.mkdir()
shutil.copytree(repo / 'mutant_cache', snapshot / 'mutant_cache',
                ignore=shutil.ignore_patterns('__pycache__'))
for name in ('mutate_everything_train.sh', 'mutate_everything_smoke.sh',
             'mutate_everything_prepare.py'):
    shutil.copy2(repo / 'experiments' / name, snapshot / name)
data = Path('/home/andrew/mutant_cache_scratch/processed/mutate_everything_cdna')
for name in ('manifest.json', 'sign_and_overlap_audit.json'):
    shutil.copy2(data / name, out / name)
split = data / 'splits' / 'mutate_everything_cdna.parquet'
shutil.copy2(split, out / split.name)
(out / 'working_tree.diff').write_bytes(subprocess.check_output(['git', 'diff'], cwd=repo))
files = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(snapshot.rglob('*')) if p.is_file()}
(out / 'source_sha256.json').write_text(json.dumps(files, indent=2))
log = out / 'driver.log'
with log.open('xb') as f:
    process = subprocess.Popen(['bash', str(repo / 'experiments' / 'mutate_everything_train.sh')],
        cwd=repo, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True)
record = dict(pid=process.pid, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    physical_gpu=2, log=str(log), output=str(out / 'joint_s0'),
    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    source_snapshot=str(snapshot), purpose='validation-only seed-0 pilot; no cDNA2 scoring',
    command=['bash', str(repo / 'experiments' / 'mutate_everything_train.sh')])
(out / 'launch.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
