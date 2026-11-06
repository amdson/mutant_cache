#!/usr/bin/env bash
set -euo pipefail
cd /home/andrew/mutant_cache
ROOT=/home/andrew/mutant_cache_scratch/mutate_everything
PY="$ROOT/venv/bin/python"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
"$PY" experiments/mutate_everything_predict.py \
  --repo "$ROOT/repo" \
  --labels /home/andrew/mutant_cache_scratch/processed/mutate_everything_cdna/comparison_labels_test.csv \
  --predictions "$ROOT/baseline/predictions_cdna2.csv" \
  --resume "$ROOT/mutate_everything_esm.pth" \
  --backbone esm2_t33_650M_UR50D --device cuda:0 --checkpoint-only-backbone
