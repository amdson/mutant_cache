"""Raw preparation script for the first cDNA training run (2026-11-04)."""
import hashlib
from pathlib import Path
import zipfile

import requests

ROOT = Path('/home/andrew/mutant_cache_scratch/mutate_everything')
ROOT.mkdir(parents=True, exist_ok=True)
archive = ROOT / 'mutate_everything_data.zip'
url = 'https://drive.usercontent.google.com/download'
if not archive.exists():
    response = requests.get(url, params={
        'id': '1akfHt1WLbC345DgZwZ5kLeMUADq9XB7y',
        'export': 'download', 'confirm': 't'}, stream=True, timeout=120)
    response.raise_for_status()
    if 'text/html' in response.headers.get('content-type', ''):
        raise RuntimeError('Drive returned HTML instead of the release archive')
    partial = archive.with_suffix('.zip.partial')
    with partial.open('wb') as f:
        for chunk in response.iter_content(1 << 20):
            f.write(chunk)
    partial.replace(archive)
print('archive', archive, archive.stat().st_size, flush=True)
h = hashlib.sha256()
with archive.open('rb') as f:
    for block in iter(lambda: f.read(1 << 20), b''):
        h.update(block)
(ROOT / 'archive.sha256').write_text(f'{h.hexdigest()}  {archive.name}\n')
with zipfile.ZipFile(archive) as z:
    # Only cDNA mutation tables are needed for sequence-only training.
    members = [m for m in z.infolist() if m.filename.endswith(
        ('/cdna_train.csv', '/cdna2_test.csv')) and '__MACOSX' not in m.filename]
    if len(members) != 2:
        raise RuntimeError(f'expected two cDNA tables, got {[m.filename for m in members]}')
    dest = ROOT / 'data'
    dest.mkdir(exist_ok=True)
    for m in members:
        output = dest / Path(m.filename).name
        output.write_bytes(z.read(m))
        print('extracted', m.filename, '->', output, output.stat().st_size, flush=True)
print('archive_sha256', h.hexdigest(), flush=True)
