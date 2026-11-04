#!/usr/bin/env bash
# First cDNA ESM-2 accuracy pilot. Full train partition, validation selection,
# no test scoring. Hyperparameters are a starting point, not a selected recipe.
set -euo pipefail
cd /home/andrew/mutant_cache
PY=/home/andrew/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/mutate_everything_esm2_pilot_20261104
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
exec "$PY" -m mutant_cache.megascale \
  --dataset mutate_everything_cdna --split mutate_everything_cdna \
  --backbone esm2_t33_650M_UR50D --composition joint --precision bf16 --seed 0 \
  --device cuda:0 --lora-r 8 --lora-alpha 16 --lora-lr 3e-4 --lr 1e-3 \
  --exposures 3 --evals-per-exposure 10 --patience 10 --selection double_rmse \
  --micro-batch 64 --proteins-per-step 4 --eval-batch-size 512 \
  --weight-decay 0.01 --grad-clip 1 --schedule cosine --warmup-frac 0.05 \
  --save-model --tag me_esm2_pilot_joint_s0 --output "$OUT/joint_s0" --out-root "$OUT"
