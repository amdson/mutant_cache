"""Check the independently released single-only model on S669."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import requests

root = Path('/home/andrew/mutant_cache_scratch/mutate_everything')
checkpoint = root / 'mutate_everything_esm_singles_only.pth'
url = ('https://utexas.box.com/index.php?rm=box_download_shared_file&'
       'shared_name=xl8fytg6rbmturu1zd5mus3sy3c2y4wn&file_id=f_1582598891728')
if not checkpoint.exists():
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    if 'text/html' in r.headers.get('content-type',''): raise RuntimeError('checkpoint unavailable')
    partial = checkpoint.with_suffix('.pth.partial')
    with partial.open('wb') as f:
        for block in r.iter_content(1 << 20): f.write(block)
    if partial.stat().st_size != 2631742331: raise RuntimeError('checkpoint size mismatch')
    partial.replace(checkpoint)
h = hashlib.sha256()
with checkpoint.open('rb') as f:
    for block in iter(lambda:f.read(1 << 20),b''):h.update(block)
(root/'single_checkpoint.json').write_text(json.dumps(dict(path=str(checkpoint), source=url, sha256=h.hexdigest()),indent=2))
env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
subprocess.run([str(root/'venv/bin/python'),'experiments/mutate_everything_predict.py',
    '--repo',str(root/'repo'),'--labels',str(root/'external/s669_labels.csv'),
    '--predictions',str(root/'external/baseline/s669_single_only_predictions.csv'),
    '--resume',str(checkpoint),'--device','cuda:0','--checkpoint-only-backbone','--requested-decoding'],env=env,check=True)
