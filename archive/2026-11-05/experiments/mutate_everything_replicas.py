"""Frozen-recipe repeats: ME_ARM=joint seeds 1/2; additive seeds 0/1/2.

CUDA_VISIBLE_DEVICES selects the physical GPU. Internal validation alone
selects checkpoints; external predictions never affect training decisions.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import torch
from mutant_cache.megascale import run

scratch = Path('/home/andrew/mutant_cache_scratch')
base = scratch / 'runs/mutate_everything_esm2_pilot_20261104/joint_s0/model.pt'
root = scratch / 'runs/mutate_everything_breadth_20261104'
arm = os.environ['ME_ARM']
assert arm in ('joint', 'additive')
template = torch.load(base, map_location='cpu', weights_only=False)['args']
for seed in ([1, 2] if arm == 'joint' else [0, 1, 2]):
    output = root / f'{arm}_s{seed}'
    if (output / 'results.json').exists():
        print('already complete', output, flush=True)
        continue
    args = argparse.Namespace(**template)
    args.seed = seed
    args.composition = arm
    args.device = 'cuda:0'
    args.output = str(output)
    args.out_root = str(root)
    args.tag = f'me_breadth_{arm}_s{seed}'
    args.final_test = False
    args.benchmark_inference = False
    args.load_trainable = None
    print('START', arm, seed, flush=True)
    result = run(args)
    assert result['status'] == 'ok'
    del result
    gc.collect()
    torch.cuda.empty_cache()
    print('COMPLETE', arm, seed, flush=True)
print('WORKER COMPLETE', arm, flush=True)
