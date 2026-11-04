"""Exact CPU/offline W&B integration preflight, 2026-11-04."""
import json
from pathlib import Path
import sys

repo = Path('/home/andrew/mutant_cache')
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo/'tests'))
from fixtures import make_synthetic_dataset
from mutant_cache.megascale import build_parser, run

root = Path('/home/andrew/mutant_cache_scratch/runs/wandb_tracking_preflight_20261104')
root.mkdir(parents=True, exist_ok=True)
data, _ = make_synthetic_dataset(root/'data', n_targets=5, seq_len=24,
    n_singles=16, n_doubles=16, n_triples=0)
args = build_parser().parse_args(['--data-dir', str(data), '--dataset', 'synthetic',
    '--split', 'protein', '--backbone', 'random:64:4:2', '--device', 'cpu', '--precision', 'fp32',
    '--lora-r', '2', '--exposures', '1', '--micro-batch', '16', '--proteins-per-step', '2',
    '--eval-min-n', '3', '--selection', 'spearman', '--save-model',
    '--save-selection-rules', 'double_rmse', 'double_spearman', 'ndcg30',
    '--wandb-mode', 'offline', '--wandb-project', 'mutant_cache', '--wandb-entity', 'amdickson128-usda',
    '--wandb-group', 'tracking-preflight', '--wandb-log-every', '1', '--output', str(root/'run')])
args.experiment_context = dict(sweep='tracking-preflight', trial_id='smoke', stage='smoke', attempt=1)
result = run(args)
record = json.loads((root/'run/tracking.json').read_text())
assert record['status'] == 'finished' and record['mode'] == 'offline'
assert 'logging_error' not in record and 'finalization_error' not in record
assert result['extra']['selection']['step'] > 0
assert result['extra']['test_summary'] is None
assert not (root/'run/predictions_test.csv').exists()
files = list((root/'run/wandb').glob('offline-run-*/run-*.wandb'))
assert files and all(p.stat().st_size > 0 for p in files)
summary = dict(status='passed', sdk_version=record['sdk_version'], tracking=record,
    selected_step=result['extra']['selection']['step'], offline_files=[str(p) for p in files])
(root/'smoke_result.json').write_text(json.dumps(summary, indent=2))
print('W&B OFFLINE PREFLIGHT PASSED', json.dumps(summary, indent=2))
