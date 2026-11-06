"""Refresh the online preflight and check persisted authentication, 2026-11-04.

The training/upload script's first completion check saw a cached running state.
This follow-up verifies that same run without creating another dashboard run.
"""
import json
import os
from pathlib import Path
import time
import wandb

root = Path('/home/andrew/mutant_cache_scratch/runs/wandb_tracking_online_preflight_20261104')
record = json.loads((root/'run/tracking.json').read_text())
result = json.loads((root/'run/results.json').read_text())
# Verify a new process can use the saved login without sourcing bashrc.
os.environ.pop('WANDB_API_KEY', None)
api = wandb.Api(timeout=30)
remote = api.run(f"{record['entity']}/{record['project']}/{record['id']}")
for attempt in range(6):
    remote.load(force=True)
    if remote.state == 'finished':
        break
    if attempt < 5:
        time.sleep(2)
assert remote.state == 'finished'
assert remote.summary['training/status'] == 'finished'
assert remote.summary['selected/step'] == result['extra']['selection']['step']
assert 'val/all_spearman' in remote.summary and 'lr/head' in remote.summary
assert remote.config['dataset'] == 'synthetic'
assert record['status'] == 'finished' and record['mode'] == 'online'
assert 'logging_error' not in record and 'finalization_error' not in record
summary = dict(status='passed', sdk_version=record['sdk_version'], url=record['url'],
    run_id=record['id'], remote_state=remote.state, selected_step=remote.summary['selected/step'],
    persisted_authentication_verified=True, server_metrics_verified=True)
(root/'smoke_result.json').write_text(json.dumps(summary, indent=2))
print('W&B ONLINE VERIFICATION PASSED', json.dumps(summary, indent=2))
