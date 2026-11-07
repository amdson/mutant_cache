"""Raw launch: import seven completed fits, run remaining 59 with frozen recipe."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

repo=Path('/home/andrew/mutant_cache')
scratch=Path('/home/andrew/mutant_cache_scratch')
root=scratch/'runs/stabilityarc_full66_v1'
py='/home/andrew/anaconda3/envs/mutant_cache/bin/python'
study=Path('experiments/stability_paper/stabilityarc_full66_v1')

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

if '--coordinate' in sys.argv:
    from mutant_cache.eval.stabilityarc import report
    for path,digest in json.loads((root/'data_sha256.json').read_text()).items():
        if sha(path)!=digest:raise RuntimeError(f'frozen data changed: {path}')
    workers=[]
    # Separate queue roots permit two concurrent jobs per GPU without sharing
    # registry locks; each queue itself schedules one job per GPU.
    for lane in [0,1]:
        with (root/f'lane{lane}.log').open('xb') as log:
            workers.append(subprocess.Popen([py,'-m','mutant_cache.queue_runner','run',str(study/f'lane{lane}_jobs.json'),
                         '--root',str(root/f'lane{lane}'),'--gpus','0,2'],cwd=root/'source_snapshot',
                         stdout=log,stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in workers):
        report(root,scratch)
        time.sleep(30)
    result=report(root,scratch)
    if any(p.returncode for p in workers) or not result['complete']:
        raise RuntimeError('one or more full-comparison jobs failed; inspect lane logs')
    (root/'COMPLETE.json').write_text(json.dumps(dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),folds=66,seed=0)))
else:
    if root.exists():raise RuntimeError('refusing to overwrite full study')
    root.mkdir(parents=True)
    snapshot=root/'source_snapshot'
    shutil.copytree(repo/'mutant_cache',snapshot/'mutant_cache',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(repo/study,snapshot/study)
    shutil.copy2(__file__,snapshot/'launch.py')
    pilot=json.loads((scratch/'stabilityarc/pilot_v1/manifest.json').read_text())
    extension=json.loads((scratch/'stabilityarc/full66_v1/manifest.json').read_text())
    if pilot['inputs']!=extension['inputs']:raise RuntimeError('source inputs changed since pilot')
    folds=[];data_paths=[]
    for old,manifest in [(True,pilot),(False,extension)]:
        for index,fold in enumerate(manifest['folds']):
            run=(scratch/'runs/stabilityarc_pilot_v1/runs'/fold['heldout']) if old else root/f'lane{index%2}'/'runs'/fold['heldout']
            folds.append({**fold,'run_dir':str(run),'imported_pilot':old})
            data_paths.extend(p for p in (scratch/'processed'/fold['dataset']).rglob('*') if p.is_file())
    if len(folds)!=66 or len({f['heldout'] for f in folds})!=66:raise RuntimeError('incomplete/duplicate fold inventory')
    (root/'study.json').write_text(json.dumps(dict(seed=0,folds=folds,training_recipe_frozen=True,pilot_imports=7,new_fits=59),indent=2))
    imports={str(Path(f['run_dir'])/name):sha(Path(f['run_dir'])/name) for f in folds if f['imported_pilot'] for name in ['results.json','predictions_test.csv','model.pt','resolved_config.json']}
    (root/'imported_sha256.json').write_text(json.dumps(imports,indent=2))
    (root/'data_sha256.json').write_text(json.dumps({str(p):sha(p) for p in data_paths},indent=2))
    (root/'source_sha256.json').write_text(json.dumps({str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()},indent=2))
    env=dict(os.environ,PYTHONPATH=str(snapshot),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
    for lane in [0,1]:
        subprocess.run([py,'-m','mutant_cache.queue_runner','run',str(study/f'lane{lane}_jobs.json'),
                       '--root',str(root/f'lane{lane}'),'--gpus','0,2','--dry-run'],cwd=snapshot,env=env,check=True,
                       stdout=subprocess.DEVNULL)
    from mutant_cache.eval.stabilityarc import report
    report(root,scratch)
    with (root/'coordinator.log').open('xb') as log:
        proc=subprocess.Popen([py,str(snapshot/'launch.py'),'--coordinate'],cwd=snapshot,env=env,
                              stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    record=dict(coordinator_pid=proc.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                imported=7,new_fits=59,concurrent_jobs=4,gpus=[0,2],wandb_group='stabilityarc_full66_v1')
    (root/'launch.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
