"""Raw final-fit lane: sequential seeds on one physical GPU, frozen JSON args."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
from mutant_cache.data.mutate_everything import sha256

root=Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_full_comparison_v1')
gpu=int(sys.argv[1])
plan=json.loads((root/'plan.json').read_text())
for path, expected in json.loads((root/'data_provenance.json').read_text()).items():
    assert sha256(path)==expected, f'data changed: {path}'
for fit in plan['fits']:
    if fit['gpu']!=gpu: continue
    dest=Path(fit['args']['output'])
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'resolved_experiment.json').write_text(json.dumps(fit,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
    code='import argparse,json; from mutant_cache.megascale import run; run(argparse.Namespace(**json.load(open('+repr(str(dest/'resolved_experiment.json'))+'))["args"]))'
    print('START',fit['id'],flush=True)
    with (dest/'train.log').open('xb') as log:
        subprocess.run([sys.executable,'-c',code],cwd=root/'source_snapshot',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((dest/'results.json').read_text())
    assert result['extra']['data']['train']['proteins']==116
    assert result['extra']['selection']['rule']=='fixed_final'
    assert result['extra']['selection']['step']==result['extra']['timing']['total_steps']
    print('COMPLETE',fit['id'],flush=True)
