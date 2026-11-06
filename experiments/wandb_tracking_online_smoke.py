"""Exact online W&B authentication/integration check, 2026-11-04.

Reads only the literal WANDB_API_KEY export when a noninteractive shell has
not inherited it. The SDK stores the login; no credential is printed or put
in experiment arguments, artifacts, or source.
"""
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time

repo = Path('/home/andrew/mutant_cache')
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo/'tests'))
import wandb
from fixtures import make_synthetic_dataset
from mutant_cache.megascale import build_parser, run

key = os.environ.get('WANDB_API_KEY')
if not key:
    for line in (Path.home()/'.bashrc').read_text().splitlines():
        if re.match(r'^\s*export\s+WANDB_API_KEY\s*=', line):
            words = shlex.split(line, comments=True)
            if len(words) == 2 and words[1].startswith('WANDB_API_KEY='):
                key = words[1].split('=', 1)[1]
    if not key:
        raise RuntimeError('WANDB_API_KEY was not found in the environment or a literal bashrc export')
    os.environ['WANDB_API_KEY'] = key
if not wandb.login(key=key, verify=True):
    raise RuntimeError('W&B authentication failed')
del key

root = Path('/home/andrew/mutant_cache_scratch/runs/wandb_tracking_online_preflight_20261104')
root.mkdir(parents=True, exist_ok=True)
data, _ = make_synthetic_dataset(root/'data', n_targets=5, seq_len=24,
    n_singles=16, n_doubles=16, n_triples=0)
args = build_parser().parse_args(['--data-dir', str(data), '--dataset', 'synthetic',
    '--split', 'protein', '--backbone', 'random:64:4:2', '--device', 'cpu', '--precision', 'fp32',
    '--lora-r', '2', '--exposures', '1', '--micro-batch', '16', '--proteins-per-step', '2',
    '--eval-min-n', '3', '--selection', 'spearman', '--save-model',
    '--save-selection-rules', 'double_rmse', 'double_spearman', 'ndcg30',
    '--wandb-mode', 'online', '--wandb-project', 'mutant_cache', '--wandb-entity', 'amdickson128-usda',
    '--wandb-group', 'tracking-verification', '--wandb-name', 'tracking-check-synthetic-cpu',
    '--wandb-tags', 'integration-test', 'synthetic', '--wandb-log-every', '1',
    '--output', str(root/'run')])
args.experiment_context = dict(sweep='tracking-verification', trial_id='synthetic-smoke',
    stage='verification', attempt=1)
result = run(args)
record = json.loads((root/'run/tracking.json').read_text())
assert record['status'] == 'finished' and record['mode'] == 'online'
assert 'logging_error' not in record and 'finalization_error' not in record
assert record['url']
assert result['extra']['test_summary'] is None
assert not (root/'run/predictions_test.csv').exists()

# Check the server, not just local files, before claiming online logging works.
api = wandb.Api(timeout=30)
for attempt in range(6):
    remote = api.run(f"{record['entity']}/{record['project']}/{record['id']}")
    if remote.state == 'finished' and remote.summary.get('training/status') == 'finished':
        break
    if attempt < 5:
        time.sleep(2)
assert remote.state == 'finished'
assert remote.summary['training/status'] == 'finished'
assert remote.summary['selected/step'] == result['extra']['selection']['step']
assert 'val/all_spearman' in remote.summary
assert 'lr/head' in remote.summary
assert remote.config['dataset'] == 'synthetic'
summary = dict(status='passed', sdk_version=record['sdk_version'], url=record['url'],
    run_id=record['id'], remote_state=remote.state, selected_step=remote.summary['selected/step'])
(root/'smoke_result.json').write_text(json.dumps(summary, indent=2))
print('W&B ONLINE PREFLIGHT PASSED', json.dumps(summary, indent=2))
