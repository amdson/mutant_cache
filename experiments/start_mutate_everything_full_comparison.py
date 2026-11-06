"""Raw launch: frozen control/E02, three seeds, all 116 development proteins."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from mutant_cache.megascale import build_parser, prepare, describe

repo = Path('/home/andrew/mutant_cache')
root = Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_full_comparison_v1')
py = '/home/andrew/anaconda3/envs/mutant_cache/bin/python'
if root.exists():
    raise RuntimeError('refusing to overwrite an existing final comparison')
state = json.loads(Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v2_tracking/registry.json').read_text())
trials = {t['id']: t for t in state['trials']}
tracking = json.loads((repo/'experiments/wandb_tracking.json').read_text())
plan = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    selection='predeclared final update; no test-dependent selection',
    training='all 116 released development proteins; canonical replicate medians',
    budgets='original screened fixed budgets: control 3 exposures; E02 6 exposures',
    diagnostic_warning='val/* metrics are in-sample on former validation proteins, not held-out performance',
    fits=[])
for recipe, tid, gpu in [('control','A00',0),('E02','E02',2)]:
    for seed in range(3):
        args = vars(build_parser().parse_args([]))
        args.update(trials[tid]['resolved_args'])
        args.update(tracking)
        fit_id = f'{recipe}_s{seed}'
        args.update(seed=seed, device='cuda:0', output=str(root/'runs'/fit_id),
            full_train_refit=True, final_eval_only=True, patience=0, save_selection_rules=[],
            final_test=False, benchmark_inference=False, load_trainable=None,
            max_train_variants=0, max_val_proteins=0, save_model=True,
            wandb_group=root.name, wandb_name=fit_id,
            wandb_tags=['mutate-everything','full-116-refit','fixed-final','val-is-in-sample'])
        from argparse import Namespace
        frames = prepare(Namespace(**args))
        assert frames['train'].target_id.nunique() == 116
        assert len(frames['train']) == 210842 and frames['test'].empty
        plan['fits'].append(dict(id=fit_id,recipe=recipe,gpu=gpu,seed=seed,args=args,
            train=describe(frames['train'])))
root.mkdir(parents=True)
(root/'plan.json').write_text(json.dumps(plan,indent=2))
snapshot=root/'source_snapshot'
shutil.copytree(repo/'mutant_cache',snapshot/'mutant_cache',ignore=shutil.ignore_patterns('__pycache__'))
for name in ['mutate_everything_full_comparison_driver.py','mutate_everything_full_comparison_lane.py']:
    shutil.copy2(repo/'experiments'/name,snapshot/name)
(root/'source_sha256.json').write_text(json.dumps({str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*.py')},indent=2))
shutil.copy2('/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v2_tracking/data_provenance.json',root/'data_provenance.json')
suite=json.loads((repo/'experiments/mutate_everything_matched/checkpoints.json').read_text())
suite['models']=[dict(id=recipe,training_protocol=f'Full 116 proteins, fixed final weights, {3 if recipe=="control" else 6} exposures; three fresh seeds; no test calibration',fits=[dict(seed=s,checkpoint=str(root/'runs'/f'{recipe}_s{s}'/'model.pt')) for s in range(3)]) for recipe in ['control','E02']]
(root/'comparison.json').write_text(json.dumps(suite,indent=2))
env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',PYTHONPATH=str(snapshot))
with (root/'driver.log').open('xb') as log:
    process=subprocess.Popen([py,str(snapshot/'mutate_everything_full_comparison_driver.py')],cwd=snapshot,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'launch.json').write_text(json.dumps(dict(pid=process.pid,started_utc=plan['created_utc']),indent=2))
print(json.dumps(dict(root=str(root),pid=process.pid),indent=2))
