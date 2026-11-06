"""Raw coordinator: confirm frozen screen winner on two additional protein holdouts."""
import json
from pathlib import Path
import subprocess
import sys
import time
from mutant_cache.tuning import initialize

root=Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2')
while not (root/'complete.json').exists():time.sleep(30)
state=json.loads((root/'registry.json').read_text())
if any(t['status']=='failed' for t in state['trials']):raise RuntimeError('screen has failed trials; inspect before holdout confirmation')
selected=state['selected_recipe']
byid={t['id']:t for t in state['trials']}
base=json.loads((root/'plan.json').read_text())['base_overrides']
# No external panels are read here. The recipe is frozen before extra holdouts.
for holdout in [1,2]:
 dest=Path(str(root)+f'_holdout{holdout}')
 spec=dict(name=f'Accuracy-v2 fixed recipe confirmation, protein holdout {holdout}',
  report_path=f'/home/andrew/mutant_cache/experiments/mutate_everything_accuracy_v2/HOLDOUT{holdout}_STATUS.md',
  base_overrides={**base,'split':f'mutate_everything_cdna_holdout{holdout}'},
  stages=[dict(id='A',name='Frozen E02 versus frozen selected recipe')],
  trials=[dict(id='A00',stage='A',seed=0,question='E02 on alternate protein holdout',overrides={}),
          dict(id='selected',stage='A',seed=0,question=f'Frozen {selected} on alternate protein holdout',overrides={**byid[selected]['resolved_args'],'split':f'mutate_everything_cdna_holdout{holdout}'})])
 plan=root/f'holdout{holdout}_plan.json';plan.write_text(json.dumps(spec,indent=2))
 initialize(dest,plan,state['template_args_source'],root/'tracking_config.json')
 (dest/'confirmation_protocol.json').write_text(json.dumps(dict(screen_root=str(root),frozen_recipe=selected,holdout=holdout,external_results_used=False),indent=2))
 workers=[]
 for gpu in [0,2]:
  with (dest/f'worker_gpu{gpu}.log').open('xb') as log:
   p=subprocess.Popen([sys.executable,'-m','mutant_cache.tuning','worker','--root',str(dest),'--gpu',str(gpu)],cwd=dest/'source_snapshot',stdout=log,stderr=subprocess.STDOUT)
  workers.append(p)
 codes=[p.wait() for p in workers]
 if any(codes):raise RuntimeError(f'holdout workers failed: {codes}')
 print('COMPLETE holdout',holdout,flush=True)
