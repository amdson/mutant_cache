#!/usr/bin/env bash
# metl_gb1_lr_probe.sh -- LR x loss probe for METL GB1 Rosetta, selected on VAL
# (never test). One GPU, serial. ds16000 (14,400 train) as the mid size.
#
#   cd ~/mutant_cache && bash experiments/metl_gb1_lr_probe.sh
set -u
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/metl_gb1_probe
mkdir -p $OUT/logs

for loss in rank mse_rank; do
  for lr in 1e-3 3e-3 1e-2; do
    tag=probe_${loss}_lr${lr}
    echo "$(date +%T) start $tag"
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.harness \
      --dataset metl_gb1_rosetta --split metl_gb1_rosetta_ds16000 \
      --eval-split val --lora-r 8 --epochs 15 --batch-size 256 \
      --eval-batch-size 1024 --lr $lr --loss $loss \
      --baselines none --epsilon-depths --seed 0 \
      --tag $tag --output $OUT/$tag --out-root $OUT > $OUT/logs/$tag.log 2>&1
    rc=$?
    best=$(grep -oP '(?<=best ρ\(all\)=)[-0-9.]+' $OUT/logs/$tag.log | tail -1)
    echo "$(date +%T) done  $tag rc=$rc best_val_rho=$best"
  done
done
