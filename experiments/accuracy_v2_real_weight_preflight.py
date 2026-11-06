"""Raw two-update real ESM2 preflight of residual, shared K/V and partial backbone."""
import argparse
import gc
import json
from pathlib import Path
import torch
from mutant_cache.megascale import build_parser, run

state=json.loads(Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_v2_tracking/registry.json').read_text())
base=next(t for t in state['trials'] if t['id']=='E02')['resolved_args']
root=Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_accuracy_v2_preflight')
for name,changes in [('residual',dict(interaction_residual=True)),('shared_kv',dict(shared_lora_targets=['q','k','v','o','fc1','fc2'])),('partial4',dict(train_backbone=True,backbone_train_last=4,precision='fp32',backbone_lr=1e-5))]:
 args=vars(build_parser().parse_args([]));args.update(base)
 args.update(device='cuda:0',output=str(root/name),exposures=1,micro_batch=16,proteins_per_step=2,max_train_variants=64,max_val_proteins=1,final_eval_only=True,save_selection_rules=[],patience=0,final_test=False,load_trainable=None,wandb_mode='disabled',diagnostics=False,save_model=False)
 args.update(changes)
 result=run(argparse.Namespace(**args))
 assert result['extra']['timing']['total_steps']==2
 assert all(torch.isfinite(p).all() for p in result['model'].parameters())
 print('PASSED',name,flush=True)
 del result
 gc.collect();torch.cuda.empty_cache()
