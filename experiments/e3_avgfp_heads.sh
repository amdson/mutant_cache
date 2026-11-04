#!/usr/bin/env bash
# e3_avgfp_heads.sh -- TODO.md 2026-11-01, E3: isolate the readout. avGFP ds1280 dev manifests,
# replicates 0-2, the E2-selected training configuration REF (e.g. REF=F_1e-5) and EXACTLY the
# same schedule / loss / selection as E2. Heads (final MLP 128 hidden, 2 layers):
#   H1 mean_count   H2 sum_mean_count   H3 wt_relative (phi width 128, 16-d ref/alt embeddings)
# H0 (--head mean) is reused from E2 (same config, same replicates). Backbone init identical
# across heads (same pretrained weights; adapters seeded by --seed). GPU 0, serial, resumable.
# Optional: HEADS="mut_proj" TOKDIM=... for the H3 capacity control (only if H3 wins).
set -u
: "${REF:?set REF to the E2-selected config, e.g. REF=F_1e-5}"
HEADS=${HEADS:-"mean_count sum_mean_count wt_relative"}
TOKDIM=${TOKDIM:-128}
N=${N:-1280}; TASK=${TASK:-avgfp}; REPS=${REPS:-"0 1 2"}; STAGE=${STAGE:-pilot_e3}
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/$STAGE
mkdir -p $OUT/logs
echo "REF=$REF HEADS=$HEADS TOKDIM=$TOKDIM N=$N TASK=$TASK REPS=$REPS"
arm=${REF%%_*}; lr=${REF#*_}
case $arm in
  F) X="--train-backbone --backbone-lr $lr" ;;
  V) X="--lora-r 8 --lora-alpha 16 --lora-lr $lr" ;;
  S) X="--shared-lora-r 8 --lora-alpha 16 --lora-lr $lr" ;;
esac
for r in $REPS; do
  COMMON="--dataset metl_${TASK}_dms --split metl_${TASK}_dms_tune_ds${N}_r$r --eval-split test --eval-label development
    --select-on-val --min-variants-per-set 1 --minimal --precision fp32
    --loss rank --target-col score --rank-sigma 1 --rank-margin 0 --rank-min-gap 0
    --lr 3e-3 --weight-decay 0.01 --grad-clip 1 --batch-size 256 --accum 1 --schedule cosine --warmup 50
    --max-steps 400 --eval-every 20 --eval-batch-size 1024
    --head-hidden 128 --head-layers 2 --token-proj-dim $TOKDIM --residue-embed-dim 16
    --control-backend none --baselines none --epsilon-depths --seed $r"
  for h in $HEADS; do
    tag=${TASK}_ds${N}_${REF}_${h}$([ $TOKDIM != 128 ] && echo _w$TOKDIM)_r$r
    [ -f $OUT/$tag/results.json ] && continue
    echo "$(date +%T) start $tag"
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.harness $COMMON $X --head $h \
      --tag $tag --output $OUT/$tag --out-root $OUT > $OUT/logs/$tag.log 2>&1
    echo "$(date +%T) done  $tag rc=$?"
  done
done
echo all done
