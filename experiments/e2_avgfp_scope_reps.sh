#!/usr/bin/env bash
# e2_avgfp_scope_reps.sh -- TODO.md 2026-11-01, E2 replicates: the one LR per arm (F, V, S)
# selected on replicate 0 by lowest best-val rank loss (experiments/summarize_pilot.py), run
# on replicates 1 and 2 with everything else identical to e2_avgfp_scope_screen.sh.
# SEL="F_xx V_xx S_xx" must be given (logged in the driver log). GPU 0, serial, resumable.
set -u
: "${SEL:?set SEL to the selected configs, e.g. SEL='F_1e-5 V_1e-3 S_1e-3'}"
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/pilot_e2
mkdir -p $OUT/logs
echo "SEL=$SEL"
for r in 1 2; do
  COMMON="--dataset metl_avgfp_dms --split metl_avgfp_dms_tune_ds1280_r$r --eval-split test --eval-label development
    --select-on-val --min-variants-per-set 1 --minimal --head mean --precision fp32
    --loss rank --target-col score --rank-sigma 1 --rank-margin 0 --rank-min-gap 0
    --lr 3e-3 --weight-decay 0.01 --grad-clip 1 --batch-size 256 --accum 1 --schedule cosine --warmup 50
    --max-steps 400 --eval-every 20 --eval-batch-size 1024
    --control-backend none --baselines none --epsilon-depths --seed $r"
  for c in $SEL; do
    arm=${c%%_*}; lr=${c#*_}
    case $arm in
      F) X="--train-backbone --backbone-lr $lr" ;;
      V) X="--lora-r 8 --lora-alpha 16 --lora-lr $lr" ;;
      S) X="--shared-lora-r 8 --lora-alpha 16 --lora-lr $lr" ;;
    esac
    tag=avgfp_ds1280_${c}_r$r
    [ -f $OUT/$tag/results.json ] && continue
    echo "$(date +%T) start $tag"
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.harness $COMMON $X \
      --tag $tag --output $OUT/$tag --out-root $OUT > $OUT/logs/$tag.log 2>&1
    echo "$(date +%T) done  $tag rc=$?"
  done
done
echo all done
