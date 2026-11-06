"""Exact online diagnostics integration preflight, synthetic CPU, 2026-11-04."""
import json
from pathlib import Path
import sys
import time

repo = Path('/home/andrew/mutant_cache')
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo/'tests'))
import wandb
from fixtures import make_synthetic_dataset
from mutant_cache.megascale import build_parser, run

root = Path('/home/andrew/mutant_cache_scratch/runs/wandb_diagnostics_preflight_20261104_verified')
root.mkdir(parents=True, exist_ok=True)
data, _ = make_synthetic_dataset(root/'data', n_targets=5, seq_len=24,
    n_singles=40, n_doubles=40, n_triples=0)
args = build_parser().parse_args(['--data-dir', str(data), '--dataset', 'synthetic',
    '--split', 'protein', '--backbone', 'random:64:4:2', '--device', 'cpu', '--precision', 'fp32',
    '--lora-r', '2', '--shared-lora-r', '2', '--head', 'wt_relative', '--exposures', '1',
    '--micro-batch', '16', '--proteins-per-step', '2', '--eval-min-n', '3', '--selection', 'spearman',
    '--evals-per-exposure', '2', '--save-model', '--reference-augmentation', 'background',
    '--augmentation-fraction', '1', '--loss', 'huber_dg', '--single-weight', '.1', '--task-reduction', 'depth',
    '--wandb-mode', 'online', '--wandb-project', 'mutant_cache', '--wandb-entity', 'amdickson128-usda',
    '--wandb-group', 'tracking-verification', '--wandb-name', 'diagnostics-check-synthetic-cpu-verified',
    '--wandb-tags', 'integration-test', 'synthetic', '--wandb-log-every', '2',
    '--diagnostics', '--diagnostic-log-every', '2', '--diagnostic-profile-every', '1',
    '--diagnostic-probe-proteins', '2', '--diagnostic-probe-rows', '32', '--diagnostic-plot-rows', '32',
    '--diagnostic-activation-proteins', '1', '--diagnostic-activation-rows', '8',
    '--output', str(root/'run')])
args.experiment_context = dict(sweep='tracking-verification', trial_id='diagnostics-smoke',
    stage='verification', attempt=1)
result = run(args)
record = json.loads((root/'run/tracking.json').read_text())
assert record['status'] == 'finished' and 'logging_error' not in record and 'finalization_error' not in record
api = wandb.Api(timeout=30)
remote = api.run(f"{record['entity']}/{record['project']}/{record['id']}")
for attempt in range(20):
    remote.load(force=True)
    if remote.state == 'finished':
        break
    time.sleep(2)
assert remote.state == 'finished'
for key in ['val/loss', 'train_probe/loss', 'train_probe/all_spearman_pooled',
    'optimization/head/grad_norm_pre_clip', 'optimization/variant_lora/update_rms_sampled',
    'activation/layer_0/variant_q/delta_to_base_rms', 'activation/head/input/rms',
    'sampling/augmented_reference_fraction', 'selected/train_probe/loss',
    'val/predicted_vs_observed_table', 'train_probe/per_protein']:
    assert key in remote.summary, key
assert remote.summary['training/status'] == 'finished'
assert result['extra']['test_summary'] is None
summary = dict(status='passed', url=record['url'], sdk_version=record['sdk_version'],
    remote_state=remote.state, local_diagnostics=str(root/'run/diagnostics'),
    metrics_and_media_verified=True)
(root/'smoke_result.json').write_text(json.dumps(summary, indent=2))
print('ONLINE DIAGNOSTICS PREFLIGHT PASSED', json.dumps(summary, indent=2))
