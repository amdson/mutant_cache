"""Real-weight bf16 smoke checks for the non-default training paths."""
import argparse
import gc
from pathlib import Path
import torch
from mutant_cache.megascale import run

template = Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_esm2_pilot_20261104/joint_s0/model.pt')
base = torch.load(template,map_location='cpu',weights_only=False)['args']
root = Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_tuning_preflight')
cases = {
    'balanced_curriculum':dict(loss='huber_dg',task_reduction='depth',single_weight=.1,
        double_weight=1.,stabilizing_double_weight=4.,double_subsample_ratio=8.,single_only_exposures=.5),
    'shared_wt':dict(shared_lora_r=8,head='wt_relative'),
    'background_huber':dict(loss='huber_dg',reference_augmentation='background',augmentation_fraction=1.,
        backgrounds_per_protein=2,task_reduction='depth',single_weight=.1,double_weight=1.,stabilizing_double_weight=4.)
}
for name,changes in cases.items():
    from mutant_cache.megascale import build_parser
    args=build_parser().parse_args([])
    vars(args).update(base)
    vars(args).update(max_train_variants=512,max_val_proteins=1,exposures=1,
        micro_batch=32,proteins_per_step=2,evals_per_exposure=2,patience=0,
        selection='spearman',save_selection_rules=['double_rmse','double_spearman','ndcg30'],
        final_test=False,output=str(root/name),out_root=str(root),device='cuda:0')
    vars(args).update(changes)
    print('PREFLIGHT',name,flush=True)
    result=run(args)
    assert all(torch.isfinite(p).all() for p in result['model'].parameters() if p.requires_grad)
    assert not (root/name/'predictions_test.csv').exists()
    print('PREFLIGHT PASSED',name,flush=True)
    del result
    gc.collect();torch.cuda.empty_cache()
