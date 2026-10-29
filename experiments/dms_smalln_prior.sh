#!/usr/bin/env bash
# dms_smalln_prior.sh -- does removing the masked-marginal prior recover small N?
# ds80/ds320, reps 0-2, 5k DEV set, select on val loss, 300 epochs (as dms_tune_sweep.sh).
# With-prior LoRA r8 at these sizes = B0 runs already in runs/dms_tune.
#   L_nomarg  LoRA r8, no prior
#   F_prior   full FT (fp32, bb lr 1e-5), prior on
#   F_nomarg  full FT, no prior
set -u
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/dms_tune
mkdir -p $OUT/logs
FT="--train-backbone --precision fp32 --backbone-lr 1e-5"
declare -A CFG=(
  [L_nomarg]="--lora-r 8 --lr 3e-3 --no-marginals"
  [F_prior]="$FT --lr 3e-3"
  [F_nomarg]="$FT --lr 3e-3 --no-marginals"
)
for c in L_nomarg F_nomarg F_prior; do
  for n in 80 320; do
    for r in 0 1 2; do
      tag=${c}_ds${n}_r${r}
      [ -f $OUT/$tag/results.json ] && continue
      echo "$(date +%T) start $tag"
      CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.harness \
        --dataset metl_gb1_dms --split metl_gb1_dms_tune_ds${n}_r${r} \
        --eval-split test --select-on-val --min-variants-per-set 1 \
        --loss rank --epochs 300 --batch-size 256 --eval-batch-size 1024 \
        --control-backend none --baselines none --epsilon-depths \
        ${CFG[$c]} --seed $r --tag $tag --output $OUT/$tag --out-root $OUT \
        > $OUT/logs/$tag.log 2>&1
      rc=$?
      echo "$(date +%T) done  $tag rc=$rc"
    done
  done
done
echo all done
