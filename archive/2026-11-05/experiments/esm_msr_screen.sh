#!/usr/bin/env bash
# esm_msr_screen.sh -- TODO_ESM_MSR_EXPERIMENT.md §3 small screen, seed 0: arm S (joint) and
# A (additive) x adapter LR {3e-4, 1e-3} x head LR {1e-3, 3e-3} = 8 runs. ESM-C 300M frozen bf16
# base, fp32 variant LoRA r8/a16 (q,o,gate,up,down), sum_mean_count zero-anchored head, MSE on
# raw ddG_ML, protein-uniform sampling (4 proteins x 64 rows per step), 10 exposures, cosine,
# 5% warmup, val once per exposure, select lowest macro val double RMSE. Val only; no test.
# Calibrated ~70 min per run on one A6000. GPU 0, serial, resumable.
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/esm_msr_screen
mkdir -p $OUT/logs
for comp in joint additive; do
  for alr in 3e-4 1e-3; do
    for hlr in 1e-3 3e-3; do
      tag=${comp}_alr${alr}_hlr${hlr}_s0
      [ -f $OUT/$tag/results.json ] && continue
      echo "$(date +%T) start $tag"
      CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.megascale --composition $comp \
        --precision bf16 --seed 0 --lora-lr $alr --lr $hlr --exposures 10 \
        --micro-batch 64 --proteins-per-step 4 --eval-batch-size 512 \
        --weight-decay 0.01 --grad-clip 1 --schedule cosine --warmup-frac 0.05 --save-model \
        --tag $tag --output $OUT/$tag --out-root $OUT > $OUT/logs/$tag.log 2>&1
      echo "$(date +%T) done  $tag rc=$?"
    done
  done
done
echo all done
