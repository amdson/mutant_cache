"""Raw driver: two final-fit lanes, then complete matched-panel evaluation."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback

root=Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_full_comparison_v1')
def status(phase,**extra):
    (root/'status.json').write_text(json.dumps(dict(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2))
try:
    lanes=[]
    for gpu in [0,2]:
        with (root/f'lane_gpu{gpu}.log').open('xb') as log:
            p=subprocess.Popen([sys.executable,str(root/'source_snapshot/mutate_everything_full_comparison_lane.py'),str(gpu)],cwd=root/'source_snapshot',stdout=log,stderr=subprocess.STDOUT)
        lanes.append((gpu,p))
    status('training',workers=[dict(gpu=g,pid=p.pid) for g,p in lanes])
    codes=[(g,p.wait()) for g,p in lanes]
    if any(code for _,code in codes): raise RuntimeError(f'training lane failed: {codes}')
    status('evaluating')
    subprocess.run([sys.executable,'-m','mutant_cache.matched_suite','--config',str(root/'comparison.json'),'--output',str(root/'matched_report'),'--device','cuda:0'],cwd=root/'source_snapshot',check=True)
    status('complete',report=str(root/'matched_report'))
except Exception as exc:
    status('failed',error=str(exc),traceback=traceback.format_exc())
    raise
