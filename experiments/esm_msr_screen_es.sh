#!/usr/bin/env bash
# esm_msr_screen_es.sh -- TODO_ESM_MSR_EXPERIMENT.md §3 small screen, seed 0, WITH EARLY STOPPING.
# Replaces esm_msr_screen.sh (stopped 2026-11-02 after 4 S runs + 1 A run): every run there peaked
# at the first once-per-exposure validation and then overfit, so selection resolution was too coarse
# and 9 of 10 exposures were wasted. Deviations from the TODO recipe, applied to both arms:
#   --exposures 3 (cosine + 5% warmup span 3 exposures, not 10)
#   --evals-per-exposure 10 (validation every ~72 steps)
#   --patience 10 (stop after one exposure of validations without a new best val double RMSE)
# Everything else as esm_msr_screen.sh: S (joint) / A (additive) x adapter LR {3e-4,1e-3} x head LR
# {1e-3,3e-3}; ESM-C 300M frozen bf16 base, fp32 variant LoRA r8/a16, sum_mean_count zero-anchored
# head, MSE on raw ddG, 4 proteins x 64 rows per step. Selection: lowest macro val double RMSE.
# Val only; no test. GPU 0, serial, resumable.
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/esm_msr_screen_es
mkdir -p $OUT/logs
for comp in joint additive; do
  for alr in 3e-4 1e-3; do
    for hlr in 1e-3 3e-3; do
      tag=${comp}_alr${alr}_hlr${hlr}_s0
      [ -f $OUT/$tag/results.json ] && continue
      echo "$(date +%T) start $tag"
      CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.megascale --composition $comp \
        --precision bf16 --seed 0 --lora-lr $alr --lr $hlr --exposures 3 \
        --evals-per-exposure 10 --patience 10 \
        --micro-batch 64 --proteins-per-step 4 --eval-batch-size 512 \
        --weight-decay 0.01 --grad-clip 1 --schedule cosine --warmup-frac 0.05 --save-model \
        --tag $tag --output $OUT/$tag --out-root $OUT > $OUT/logs/$tag.log 2>&1
      echo "$(date +%T) done  $tag rc=$?"
    done
  done
done
echo all done
