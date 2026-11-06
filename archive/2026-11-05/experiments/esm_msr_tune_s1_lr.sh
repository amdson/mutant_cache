#!/usr/bin/env bash
# esm_msr_tune_s1_lr.sh -- best-effort tuning of S on ESM-MSR's eval stack (2026-11-04), stage 1: LR.
# Goal changed 2026-11-04: train S as well as possible on ESM-MSR's data/eval stack; A is one
# control run at S's tuned recipe later. Eval matched to ESM-MSR: --eval-protein-col code_wt,
# --eval-min-n 2, --selection spearman (macro all-depth within-protein Spearman, their
# val_rho_combined_avg quantity). Otherwise as esm_msr_screen_es.sh: ESM-C 300M frozen bf16 base,
# fp32 variant LoRA r8/a16, sum_mean_count zero-anchored head, MSE on raw ddG, 4 proteins x 64
# rows per step, 3 exposures cosine + 5% warmup, 10 validations/exposure, patience 10.
# Grid: adapter LR {3e-4,1e-3,3e-3} x head LR {1e-3,3e-3}. Seed 0. Val only; no test. GPU 0.
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/esm_msr_tune_s1_lr
mkdir -p $OUT/logs
for alr in 3e-4 1e-3 3e-3; do
  for hlr in 1e-3 3e-3; do
    tag=joint_alr${alr}_hlr${hlr}_s0
    [ -f $OUT/$tag/results.json ] && continue
    echo "$(date +%T) start $tag"
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.megascale --composition joint \
      --precision bf16 --seed 0 --lora-lr $alr --lr $hlr --exposures 3 \
      --evals-per-exposure 10 --patience 10 \
      --eval-protein-col code_wt --eval-min-n 2 --selection spearman \
      --micro-batch 64 --proteins-per-step 4 --eval-batch-size 512 \
      --weight-decay 0.01 --grad-clip 1 --schedule cosine --warmup-frac 0.05 --save-model \
      --tag $tag --output $OUT/$tag --out-root $OUT > $OUT/logs/$tag.log 2>&1
    echo "$(date +%T) done  $tag rc=$?"
  done
done
echo all done
