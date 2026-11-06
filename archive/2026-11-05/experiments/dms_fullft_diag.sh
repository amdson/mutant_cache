#!/usr/bin/env bash
# dms_fullft_diag.sh -- why does full fine-tuning trail ESM-2 35M FT on GB1 DMS?
# Smoke at ds1280: full FT through the cache 0.893, through EXACT 0.884 (dev) vs
# ESM-2 35M FT 0.925 (METL test) -> the cache is not the gap; the setup is.
# One knob at a time off full FT (F). ds1280, reps 0-1, 5k DEV set (never the
# supertest), select on val loss, 40 epochs (best epoch was ~18 in the smokes).
set -u
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/dms_fullft_diag
mkdir -p $OUT/logs
FT="--train-backbone --precision fp32"
declare -A CFG=(
  [F]="$FT --backbone-lr 1e-5 --lr 3e-3"
  [bb3e-6]="$FT --backbone-lr 3e-6 --lr 3e-3"
  [bb3e-5]="$FT --backbone-lr 3e-5 --lr 3e-3"
  [bb1e-4]="$FT --backbone-lr 1e-4 --lr 3e-3"
  [head3e-4]="$FT --backbone-lr 1e-5 --lr 3e-4"
  [mse]="$FT --backbone-lr 1e-5 --lr 3e-3 --loss mse_rank"
  [nomarg]="$FT --backbone-lr 1e-5 --lr 3e-3 --no-marginals"
  [wd0]="$FT --backbone-lr 1e-5 --lr 3e-3 --weight-decay 0"
  [drop1]="$FT --backbone-lr 1e-5 --lr 3e-3 --dropout 0.1"
  [const]="$FT --backbone-lr 1e-5 --lr 3e-3 --schedule constant"
  [sum]="$FT --backbone-lr 1e-5 --lr 3e-3 --pool sum"
  [lora8]="--lora-r 8 --lr 3e-3"
)
ORDER="F bb3e-6 bb3e-5 bb1e-4 head3e-4 mse nomarg wd0 drop1 const sum lora8"
for c in $ORDER; do
  for r in 0 1; do
    tag=${c}_r${r}
    [ -f $OUT/$tag/results.json ] && continue
    echo "$(date +%T) start $tag"
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.harness \
      --dataset metl_gb1_dms --split metl_gb1_dms_tune_ds1280_r${r} \
      --eval-split test --select-on-val --min-variants-per-set 1 \
      --loss rank --epochs 40 --batch-size 256 --eval-batch-size 1024 \
      --control-backend none --baselines none --epsilon-depths \
      ${CFG[$c]} --seed $r --tag $tag --output $OUT/$tag --out-root $OUT \
      > $OUT/logs/$tag.log 2>&1
    rc=$?
    echo "$(date +%T) done  $tag rc=$rc"
  done
done
echo all done
