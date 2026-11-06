#!/usr/bin/env bash
# esm_msr_smoke.sh -- TODO_ESM_MSR_EXPERIMENT.md §3 smoke: one seed of S (joint) and A (additive)
# on a fixed train subset (4,000 rows, seed 0) and the first 4 val proteins. bf16 frozen base,
# fp32 adapters/head. Checks plumbing + calibrates throughput. Never touches test. GPU 0.
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/esm_msr_smoke
mkdir -p $OUT/logs
for comp in joint additive; do
  tag=smoke_${comp}_s0
  echo "$(date +%T) start $tag"
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.megascale --composition $comp \
    --precision bf16 --seed 0 --max-train-variants 4000 --max-val-proteins 4 --exposures 2 \
    --micro-batch 64 --proteins-per-step 4 --eval-batch-size 512 \
    --tag $tag --output $OUT/$tag --out-root $OUT > $OUT/logs/$tag.log 2>&1
  echo "$(date +%T) done  $tag rc=$?"
done
