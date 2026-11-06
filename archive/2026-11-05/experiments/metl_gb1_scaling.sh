#!/usr/bin/env bash
# metl_gb1_scaling.sh -- METL GB1 Rosetta synthetic: train-size scaling of the
# variant cache (head + variant LoRA r=8 over a frozen bf16 ESM-C 300M).
#
# Train sizes are METL's own resampled_1 grid (ds<N> = train+val, 0.9/0.1), test
# is a fixed 50k subsample of METL's shared supertest holdout, so sizes are
# mutually comparable. Hyperparameters were chosen on VAL only
# (experiments/metl_gb1_lr_probe.sh: loss=rank, lr=3e-3 >> mse_rank).
# Fixed epoch budget + --no-select-best: nothing is selected on test.
# Epochs give ~240-1350 optimizer steps per size (the val curve plateaus by ~570).
#
#   cd ~/mutant_cache && bash experiments/metl_gb1_scaling.sh
# One GPU, serial (one A6000 at a time).
set -u
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/metl_gb1_scaling
mkdir -p $OUT/logs

declare -A EPOCHS=( [1000]=60 [4000]=60 [16000]=21 [64000]=6 [128000]=3 )

for seed in 0 1 2; do
  for n in 1000 4000 16000 64000 128000; do
    tag=ds${n}_s${seed}
    # the exact-backend control (full forward of every mutant) only on seed 0:
    # it is ~60x slower than the cache, and one seed is enough for the delta
    ctl=none; [ $seed -eq 0 ] && ctl=exact
    echo "$(date +%T) start $tag (epochs ${EPOCHS[$n]}, control $ctl)"
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.harness \
      --dataset metl_gb1_rosetta --split metl_gb1_rosetta_ds${n} \
      --eval-split test --no-select-best \
      --lora-r 8 --loss rank --lr 3e-3 \
      --epochs ${EPOCHS[$n]} --batch-size 256 --eval-batch-size 1024 \
      --control-backend $ctl --baselines zero_shot --epsilon-depths \
      --seed $seed --tag $tag --output $OUT/$tag --out-root $OUT \
      > $OUT/logs/$tag.log 2>&1
    rc=$?
    # NB: take the MODEL's line, not the control's (which prints after it)
    rho=$(grep -oP '(?<=ρ@all=)[-0-9.]+' $OUT/logs/$tag.log | sed -n '$!p' | tail -1)
    [ "$ctl" = none ] && rho=$(grep -oP '(?<=ρ@all=)[-0-9.]+' $OUT/logs/$tag.log | tail -1)
    echo "$(date +%T) done  $tag rc=$rc test_rho=$rho"
  done
done
echo all done
