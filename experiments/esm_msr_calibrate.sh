#!/usr/bin/env bash
# esm_msr_calibrate.sh -- throughput calibration for the ESM-MSR screen: S (joint), seed 0,
# 20,000 train rows, 1 exposure (~78 steps), 4 val proteins. Cost estimate only. GPU 0.
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/esm_msr_smoke
mkdir -p $OUT/logs
for mb in 64 256; do
  tag=calib_joint_mb${mb}
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.megascale --composition joint \
    --precision bf16 --seed 0 --max-train-variants 20000 --max-val-proteins 4 --exposures 1 \
    --micro-batch $mb --proteins-per-step $((256 / mb)) --eval-batch-size 512 \
    --tag $tag --output $OUT/$tag --out-root $OUT > $OUT/logs/$tag.log 2>&1
  echo "$tag rc=$?"
done
