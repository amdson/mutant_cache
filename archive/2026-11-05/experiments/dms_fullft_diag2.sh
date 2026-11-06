#!/usr/bin/env bash
# dms_fullft_diag2.sh -- round 2 of the full-FT diagnostic (see dms_fullft_diag.sh).
# Round 1: every knob within +-0.01 of F (0.895 dev); MSE collapsed to 0.33 with
# train loss 27.5 -> 0.16 (worse than a constant, 0.083 on uniform ranks): the
# masked-marginal prior starts predictions at LLR scale (~+-10) vs a [0,1] target.
# Round 2: get plain MSE regression working (prior off / not zero-init / head lr),
# combine the best rank knobs, and add rep 2 for F and nomarg.
set -u
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/dms_fullft_diag
mkdir -p $OUT/logs
FT="--train-backbone --precision fp32 --backbone-lr 1e-5"
declare -A CFG=(
  [mse_nomarg]="$FT --lr 3e-3 --loss mse_rank --no-marginals"
  [mse_nomarg_lr1e-2]="$FT --lr 1e-2 --loss mse_rank --no-marginals"
  [mse_nomarg_lr1e-3]="$FT --lr 1e-3 --loss mse_rank --no-marginals"
  [mse_noz]="$FT --lr 3e-3 --loss mse_rank --no-zero-init-additive"
  [mse_lr1e-2]="$FT --lr 1e-2 --loss mse_rank"
  [nomarg_bb3e-5]="--train-backbone --precision fp32 --backbone-lr 3e-5 --lr 3e-3 --no-marginals"
  [F]="$FT --lr 3e-3"
  [nomarg]="$FT --lr 3e-3 --no-marginals"
)
RUNS="mse_nomarg:0 mse_nomarg:1 mse_nomarg_lr1e-2:0 mse_nomarg_lr1e-2:1 mse_nomarg_lr1e-3:0 mse_nomarg_lr1e-3:1
      mse_noz:0 mse_noz:1 mse_lr1e-2:0 mse_lr1e-2:1 nomarg_bb3e-5:0 nomarg_bb3e-5:1 F:2 nomarg:2"
for cr in $RUNS; do
  c=${cr%:*}; r=${cr#*:}
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
echo all done
