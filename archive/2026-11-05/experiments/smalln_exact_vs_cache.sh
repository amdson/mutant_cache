#!/usr/bin/env bash
# smalln_exact_vs_cache.sh -- does training through the exact full forward beat the sparse
# cache at small label budgets? Upper-bound proxy for what sparse-teacher distillation could buy
# (TODO_ARCHITECTURE_MISMATCH_PRETRAINING.md). ESM-MSR hyperopt train subsampled to N rows
# (across proteins), 10 exposures, final weights only (--final-eval-only, no selection),
# first 8 val proteins. bf16 base, default megascale config (variant LoRA r8, sum_mean_count).
# GPU 0, serial. Never touches test.
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/smalln_exact_vs_cache
mkdir -p $OUT/logs
for seed in 0 1; do
  for n in 256 1024 4096; do
    for be in cache exact; do
      tag=${be}_n${n}_s$seed
      [ -f $OUT/$tag/results.json ] && continue
      echo "$(date +%T) start $tag"
      CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.megascale --backend $be \
        --precision bf16 --seed $seed --max-train-variants $n --max-val-proteins 8 --exposures 10 \
        --final-eval-only --micro-batch 64 --proteins-per-step 4 --eval-batch-size 512 \
        --tag $tag --output $OUT/$tag --out-root $OUT > $OUT/logs/$tag.log 2>&1
      echo "$(date +%T) done  $tag rc=$?"
    done
  done
done
echo all done
