#!/usr/bin/env bash
# esm_msr_seeds.sh -- TODO_ESM_MSR §3: repeat each arm's validation-selected configuration
# (summarize_esm_msr.py --select) with seeds 1 and 2; seed 0 comes from the screen.
# SEL_S / SEL_A = "<adapter lr> <head lr>", e.g. SEL_S="1e-3 3e-3". Val only. GPU 0, serial.
set -u
: "${SEL_S:?}"; : "${SEL_A:?}"
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/esm_msr_screen
mkdir -p $OUT/logs
echo "SEL_S=$SEL_S SEL_A=$SEL_A"
for seed in 1 2; do
  for comp in joint additive; do
    if [ $comp = joint ]; then set -- $SEL_S; else set -- $SEL_A; fi
    alr=$1; hlr=$2
    tag=${comp}_alr${alr}_hlr${hlr}_s$seed
    [ -f $OUT/$tag/results.json ] && continue
    echo "$(date +%T) start $tag"
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.megascale --composition $comp \
      --precision bf16 --seed $seed --lora-lr $alr --lr $hlr --exposures 10 \
      --micro-batch 64 --proteins-per-step 4 --eval-batch-size 512 \
      --weight-decay 0.01 --grad-clip 1 --schedule cosine --warmup-frac 0.05 --save-model \
      --tag $tag --output $OUT/$tag --out-root $OUT > $OUT/logs/$tag.log 2>&1
    echo "$(date +%T) done  $tag rc=$?"
  done
done
echo all done
