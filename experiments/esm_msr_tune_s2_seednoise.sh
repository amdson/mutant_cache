#!/usr/bin/env bash
# esm_msr_tune_s2_seednoise.sh -- stage 2 of best-effort S tuning (2026-11-04): seed noise.
# Stage 1 (esm_msr_tune_s1_lr.sh) gave macro val all-depth Spearman 0.6524-0.6763 across 6 LR
# configs at seed 0 -- a 0.024 spread with no noise estimate, so the ranking of the top four is
# uninterpretable. This runs seeds 1 and 2 of the stage-1 best config (adapter 1e-3, head 3e-3) to
# get a 3-seed spread; every later lever (schedule, LoRA rank, 600M backbone, ranking loss,
# two-phase LoRA) is judged against that spread, not against seed 0 alone.
# Identical recipe and matched ESM-MSR eval otherwise. Val only; no test. GPU 0, serial, resumable.
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/esm_msr_tune_s2_seednoise
mkdir -p $OUT/logs
for seed in 1 2; do
  tag=joint_alr1e-3_hlr3e-3_s${seed}
  [ -f $OUT/$tag/results.json ] && continue
  echo "$(date +%T) start $tag"
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.megascale --composition joint \
    --precision bf16 --seed $seed --lora-lr 1e-3 --lr 3e-3 --exposures 3 \
    --evals-per-exposure 10 --patience 10 \
    --eval-protein-col code_wt --eval-min-n 2 --selection spearman \
    --micro-batch 64 --proteins-per-step 4 --eval-batch-size 512 \
    --weight-decay 0.01 --grad-clip 1 --schedule cosine --warmup-frac 0.05 --save-model \
    --tag $tag --output $OUT/$tag --out-root $OUT > $OUT/logs/$tag.log 2>&1
  echo "$(date +%T) done  $tag rc=$?"
done
echo all done
