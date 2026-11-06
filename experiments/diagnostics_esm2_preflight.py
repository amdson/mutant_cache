"""Exact 650M bf16 diagnostic preflight alongside the active queue, 2026-11-04."""
import json
from pathlib import Path
import torch
from mutant_cache.megascale import build_parser, run

root = Path('/home/andrew/mutant_cache_scratch/runs/diagnostics_esm2_preflight_20261104')
template = Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_esm2_pilot_20261104/joint_s0/model.pt')
args = build_parser().parse_args([])
vars(args).update(torch.load(template, map_location='cpu', weights_only=False)['args'])
vars(args).update(max_train_variants=1024, max_val_proteins=1, exposures=1,
    micro_batch=32, proteins_per_step=2, evals_per_exposure=2, patience=0,
    selection='spearman', save_selection_rules=['double_rmse','double_spearman','ndcg30'],
    final_test=False, load_trainable=None, output=str(root/'run'), out_root=str(root), device='cuda:0',
    diagnostics=True, diagnostic_log_every=4, diagnostic_profile_every=2,
    diagnostic_probe_proteins=2, diagnostic_probe_rows=32,
    diagnostic_activation_proteins=1, diagnostic_activation_rows=8,
    diagnostic_plot_rows=32, wandb_mode='offline', wandb_project='mutant_cache',
    wandb_entity='amdickson128-usda', wandb_group='tracking-verification',
    wandb_name='diagnostics-650m-bf16-smoke', wandb_log_every=4)
result = run(args)
rows = [json.loads(s) for s in (root/'run/diagnostics/history.jsonl').read_text().splitlines()]
assert all(torch.isfinite(p).all() for p in result['model'].parameters() if p.requires_grad)
assert any('activation/layer_32/variant_q/delta_to_base_rms' in r for r in rows)
assert any('train_probe/loss' in r for r in rows)
assert any('optimization/variant_lora/update_rms_sampled' in r for r in rows)
assert not (root/'run/predictions_test.csv').exists()
tracking = json.loads((root/'run/tracking.json').read_text())
assert 'logging_error' not in tracking and 'finalization_error' not in tracking
summary = dict(status='passed', layers=33, precision=args.precision,
    memory=result['extra']['memory'], timing=result['extra']['timing'], diagnostics_records=len(rows))
(root/'smoke_result.json').write_text(json.dumps(summary, indent=2))
print('REAL 650M DIAGNOSTICS PREFLIGHT PASSED', json.dumps(summary, indent=2))
