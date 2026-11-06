"""Download the sequence-only comparator linked by upstream's evaluation recipe."""
import hashlib
import json
from pathlib import Path
import requests

root = Path('/home/andrew/mutant_cache_scratch/mutate_everything')
path = root / 'mutate_everything_esm.pth'
url = ('https://utexas.box.com/index.php?rm=box_download_shared_file&'
       'shared_name=giodqufy5idqfv3apgbxtevfd8q7nytw&file_id=f_1336050981541')
if not path.exists():
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    if 'text/html' in r.headers.get('content-type', ''):
        raise RuntimeError('Box returned a page instead of a checkpoint')
    expected = int(r.headers['content-length'])
    partial = path.with_suffix('.pth.partial')
    size = 0
    with partial.open('wb') as f:
        for block in r.iter_content(1 << 20):
            f.write(block)
            size += len(block)
            if size % (256 << 20) == 0:
                print('downloaded MiB', size >> 20, 'of', expected >> 20, flush=True)
    if size != expected:
        raise RuntimeError(f'truncated checkpoint: {size} != {expected}')
    partial.replace(path)
h = hashlib.sha256()
with path.open('rb') as f:
    for block in iter(lambda: f.read(1 << 20), b''):
        h.update(block)
record = {'path': str(path), 'source': url, 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}
(root / 'checkpoint.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2), flush=True)
