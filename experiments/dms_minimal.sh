#!/usr/bin/env bash
# dms_minimal.sh -- the MINIMAL model (--minimal: cache -> mean-pool mutated tokens
# -> one MLP; no marginal prior, no added token embeddings, no reference summary,
# no assay terms) on GB1 DMS. Everything from here starts from this model.
# ds80/ds320/ds1280, reps 0-2, 5k DEV set (never the supertest), select on val loss.
# Epochs: 300 at ds80/320 (1 batch/epoch), 40 at ds1280 (best epoch ~15-20 before).
#   mL / mF        LoRA r8 / full FT (fp32, bb lr 1e-5), rank loss, head lr 3e-3
#   mL_mse / mF_mse  same with MSE on the rank target
# Ordered so the rank-loss results land first.
set -u
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/dms_tune
mkdir -p $OUT/logs
FT="--train-backbone --precision fp32 --backbone-lr 1e-5"
declare -A CFG=(
  [mL]="--minimal --lora-r 8 --lr 3e-3 --loss rank"
  [mF]="--minimal $FT --lr 3e-3 --loss rank"
  [mL_mse]="--minimal --lora-r 8 --lr 3e-3 --loss mse_rank"
  [mF_mse]="--minimal $FT --lr 3e-3 --loss mse_rank"
)
declare -A EP=( [80]=300 [320]=300 [1280]=40 )
for c in mL mF mL_mse mF_mse; do
  for n in 1280 320 80; do
    for r in 0 1 2; do
      tag=${c}_ds${n}_r${r}
      [ -f $OUT/$tag/results.json ] && continue
      echo "$(date +%T) start $tag"
      CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.harness \
        --dataset metl_gb1_dms --split metl_gb1_dms_tune_ds${n}_r${r} \
        --eval-split test --select-on-val --min-variants-per-set 1 \
        --epochs ${EP[$n]} --batch-size 256 --eval-batch-size 1024 \
        --control-backend none --baselines none --epsilon-depths \
        ${CFG[$c]} --seed $r --tag $tag --output $OUT/$tag --out-root $OUT \
        > $OUT/logs/$tag.log 2>&1
      rc=$?
      echo "$(date +%T) done  $tag rc=$rc"
    done
  done
done
echo all done
