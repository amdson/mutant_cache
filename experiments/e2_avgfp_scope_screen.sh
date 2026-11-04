#!/usr/bin/env bash
# e2_avgfp_scope_screen.sh -- TODO.md 2026-11-01, E2 screen: training/adapter scope on avGFP
# DEVELOPMENT split (metl_avgfp_dms_tune_ds1280_r0: METL train/val, 5k dev holdout outside
# supertest + every queue train/val). Original minimal mean head held fixed. ESM-C 300M,
# sparse ragged cache, fp32 base AND adapters, uniform RankNet (sigma 1, margin 0, min-gap 0)
# on raw `score`, head lr 3e-3, AdamW wd 0.01, clip 1, batch 256, accum 1, cosine with
# warmup min(50, 400//10)=40, EXACTLY 400 optimizer steps (--max-steps; historical avGFP
# ds1280 was 100 epochs x 4 batches = 400 steps too, but evaluated every epoch). Val every 20
# steps + final; restore lowest val rank loss. Dev scored at init and after selection only.
#   F  full backbone,        backbone lr 3e-6 1e-5 3e-5   (F_1e-5 = anchor, run first)
#   V  variant LoRA r8 a16,  adapter lr 1e-4 3e-4 1e-3 3e-3
#   S  shared LoRA r8 a16,   adapter lr 1e-4 3e-4 1e-3 3e-3
# Default targets q o gate up down. Replicate 0 only. Resumable. GPU 0, serial.
set -u
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/pilot_e2
mkdir -p $OUT/logs
r=0
COMMON="--dataset metl_avgfp_dms --split metl_avgfp_dms_tune_ds1280_r$r --eval-split test --eval-label development
  --select-on-val --min-variants-per-set 1 --minimal --head mean --precision fp32
  --loss rank --target-col score --rank-sigma 1 --rank-margin 0 --rank-min-gap 0
  --lr 3e-3 --weight-decay 0.01 --grad-clip 1 --batch-size 256 --accum 1 --schedule cosine --warmup 50
  --max-steps 400 --eval-every 20 --eval-batch-size 1024
  --control-backend none --baselines none --epsilon-depths --seed $r"
declare -A CFG=(
  [F_3e-6]="--train-backbone --backbone-lr 3e-6"
  [F_1e-5]="--train-backbone --backbone-lr 1e-5"
  [F_3e-5]="--train-backbone --backbone-lr 3e-5"
  [V_1e-4]="--lora-r 8 --lora-alpha 16 --lora-lr 1e-4"
  [V_3e-4]="--lora-r 8 --lora-alpha 16 --lora-lr 3e-4"
  [V_1e-3]="--lora-r 8 --lora-alpha 16 --lora-lr 1e-3"
  [V_3e-3]="--lora-r 8 --lora-alpha 16 --lora-lr 3e-3"
  [S_1e-4]="--shared-lora-r 8 --lora-alpha 16 --lora-lr 1e-4"
  [S_3e-4]="--shared-lora-r 8 --lora-alpha 16 --lora-lr 3e-4"
  [S_1e-3]="--shared-lora-r 8 --lora-alpha 16 --lora-lr 1e-3"
  [S_3e-3]="--shared-lora-r 8 --lora-alpha 16 --lora-lr 3e-3"
)
ORDER=${ORDER:-"F_1e-5 F_3e-6 F_3e-5 V_1e-4 V_3e-4 V_1e-3 V_3e-3 S_1e-4 S_3e-4 S_1e-3 S_3e-3"}
for c in $ORDER; do
  tag=avgfp_ds1280_${c}_r$r
  [ -f $OUT/$tag/results.json ] && continue
  echo "$(date +%T) start $tag"
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.harness $COMMON ${CFG[$c]} \
    --tag $tag --output $OUT/$tag --out-root $OUT > $OUT/logs/$tag.log 2>&1
  echo "$(date +%T) done  $tag rc=$?"
done
echo all done
