"""One cDNA2 evaluation of the frozen, validation-selected seed-0 pilot.

No retraining or test-dependent selection. Later model choices must still be
made on internal validation; this is a preliminary single-seed comparison.
"""
import argparse
from pathlib import Path
import torch
from mutant_cache.megascale import run

root = Path('/home/andrew/mutant_cache_scratch/runs/mutate_everything_esm2_pilot_20261104')
checkpoint = root / 'joint_s0' / 'model.pt'
frozen = torch.load(checkpoint, map_location='cpu', weights_only=False)
args = argparse.Namespace(**frozen['args'])
args.exposures = 0
args.final_test = True
args.final_eval_only = True
args.benchmark_inference = True
args.load_trainable = str(checkpoint)
args.output = str(root / 'joint_s0_cdna2_fixed')
args.tag = 'me_esm2_pilot_joint_s0_cdna2_fixed'
result = run(args)
assert result['status'] == 'ok'
