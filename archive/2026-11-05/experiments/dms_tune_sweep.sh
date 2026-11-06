#!/usr/bin/env bash
# dms_tune_sweep.sh -- GB1 DMS settings sweep vs fine-tuned ESM-2 35M (METL Fig. 2).
# Splits: experiments/build_dms_tune_splits.py -- METL's train/val for reps 0-2 of
# ds80/ds320/ds1280, scored on a fixed 5k DEV set (never the supertest).
# Selection on val loss (--select-on-val). Winners get run on the real test after.
#   B0  current: LoRA r8, head lr 3e-3, frozen bf16 base
#   A1/A2  head lr 1e-3 / 3e-4
#   H   head only (no LoRA)
#   R32 LoRA r32
#   F5/F4  full weights (no LoRA), fp32, backbone lr 1e-5 / 1e-4
#   M   LoRA r8 + full weights (backbone lr 1e-5), fp32
#   X   full weights through the EXACT backend (plain ESM-C 300M fine-tuning)
# Resumable. One GPU, serial.
set -u
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/dms_tune
mkdir -p $OUT/logs
declare -A CFG=(
  [B0]="--lora-r 8 --lr 3e-3"
  [A1]="--lora-r 8 --lr 1e-3"
  [A2]="--lora-r 8 --lr 3e-4"
  [H]="--lora-r 0 --lr 3e-3"
  [R32]="--lora-r 32 --lr 3e-3"
  [F5]="--lr 3e-3 --train-backbone --backbone-lr 1e-5 --precision fp32"
  [F4]="--lr 3e-3 --train-backbone --backbone-lr 1e-4 --precision fp32"
  [M]="--lora-r 8 --lr 3e-3 --train-backbone --backbone-lr 1e-5 --precision fp32"
  [X]="--lr 3e-3 --train-backbone --backbone-lr 1e-5 --precision fp32 --backend exact"
)
ORDER="B0 A1 A2 H R32 F5 F4 M X"
for c in $ORDER; do
  for n in 80 320 1280; do
    ntr=$(( n * 8 / 10 )); spe=$(( (ntr + 255) / 256 ))
    ep=$(( (400 + spe - 1) / spe )); [ $ep -gt 300 ] && ep=300; [ $ep -lt 20 ] && ep=20
    for r in 0 1 2; do
      tag=${c}_ds${n}_r${r}
      [ -f $OUT/$tag/results.json ] && continue
      echo "$(date +%T) start $tag (epochs $ep)"
      CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.harness \
        --dataset metl_gb1_dms --split metl_gb1_dms_tune_ds${n}_r${r} \
        --eval-split test --select-on-val --min-variants-per-set 1 \
        --loss rank --epochs $ep --batch-size 256 --eval-batch-size 1024 \
        --control-backend none --baselines none --epsilon-depths \
        ${CFG[$c]} --seed $r --tag $tag --output $OUT/$tag --out-root $OUT \
        > $OUT/logs/$tag.log 2>&1
      rc=$?
      echo "$(date +%T) done  $tag rc=$rc"
    done
  done
done
echo all done
