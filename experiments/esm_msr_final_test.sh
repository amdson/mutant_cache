#!/usr/bin/env bash
# esm_msr_final_test.sh -- TODO_ESM_MSR §4: the ONE batch of final test analyses. Loads each
# frozen, validation-selected checkpoint (seed 0 from the screen, seeds 1-2 from
# esm_msr_seeds.sh; model.pt holds the selected adapters + head) and scores val + test once
# (--exposures 0: no training). Run only after SEL_S / SEL_A are frozen. GPU 0, serial.
set -u
: "${SEL_S:?}"; : "${SEL_A:?}"
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
SRC=/home/andrew/mutant_cache_scratch/runs/esm_msr_screen
OUT=/home/andrew/mutant_cache_scratch/runs/esm_msr_final
mkdir -p $OUT/logs
echo "SEL_S=$SEL_S SEL_A=$SEL_A"
for seed in 0 1 2; do
  for comp in joint additive; do
    if [ $comp = joint ]; then set -- $SEL_S; else set -- $SEL_A; fi
    alr=$1; hlr=$2
    tag=${comp}_alr${alr}_hlr${hlr}_s$seed
    [ -f $OUT/$tag/results.json ] && continue
    [ -f $SRC/$tag/model.pt ] || { echo "missing $SRC/$tag/model.pt"; exit 1; }
    echo "$(date +%T) start $tag"
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.megascale --composition $comp \
      --precision bf16 --seed $seed --lora-lr $alr --lr $hlr --exposures 0 \
      --eval-batch-size 512 --load-trainable $SRC/$tag/model.pt --final-test \
      --tag $tag --output $OUT/$tag --out-root $OUT > $OUT/logs/$tag.log 2>&1
    echo "$(date +%T) done  $tag rc=$?"
  done
done
echo all done
