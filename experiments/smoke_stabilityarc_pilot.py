"""Raw one-update real-weight smoke: training subset/inner validation only."""
from mutant_cache.train import resolve
from mutant_cache.training import train

args,_=resolve(['--experiment','experiments/stability_paper/stabilityarc_pilot_v1/BCHB_CHLTE_experiment.json',
                '--seed','0','--output','/home/andrew/mutant_cache_scratch/runs/stabilityarc_pilot_smoke_v1'])
args.full_train_refit=False
args.final_test=False
args.max_train_variants=256
args.max_val_proteins=1
args.exposures=1
args.wandb_mode='disabled'
args.diagnostic_probe_proteins=1
args.diagnostic_probe_rows=16
args.diagnostic_activation_rows=8
args.diagnostic_plot_rows=32
result=train(args)
assert result['status']=='ok'
from pathlib import Path
assert not (Path(args.output)/'predictions_test.csv').exists()
print('SMOKE PASSED: no outer test predictions')
