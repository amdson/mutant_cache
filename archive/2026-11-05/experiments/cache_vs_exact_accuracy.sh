#!/usr/bin/env bash
# cache_vs_exact_accuracy.sh -- PLAN.md 22: what the cache approximation costs.
# Each run trains through one backend (cache = ragged sparse cache, exact = full
# forward of every mutant) and scores the same final weights through the other.
# Exact-trained runs on GPU 0, cache-trained on GPU 2, in parallel.
#
#   cd ~/mutant_cache && bash experiments/cache_vs_exact_accuracy.sh
# Prereq: experiments/build_accuracy_splits.py
set -u
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/cache_vs_exact
mkdir -p $OUT/logs

COMMON="--lora-r 8 --precision bf16 --batch-size 32 --eval-batch-size 512 \
  --eval-split test --no-select-best --baselines zero_shot --out-root $OUT"

declare -A DATA
DATA[tsub4]="--dataset tsuboyama --split acc_tsuboyama4_random --epochs 3"
DATA[brca1]="--dataset mavedb --score-sets urn:mavedb:00000003-b-1 --split acc_brca1_random --epochs 3"
DATA[gb1]="--dataset flip_gb1 --split flip_gb1_two_vs_rest --epochs 20"

lane() {   # $1 = train backend, $2 = control backend, $3 = gpu
  for seed in 0 1 2; do
    for ds in tsub4 brca1 gb1; do
      tag=${ds}_train-$1_s$seed
      echo "$(date +%T) start $tag"
      CUDA_VISIBLE_DEVICES=$3 HF_HUB_OFFLINE=1 $PY -m mutant_cache.harness \
        ${DATA[$ds]} $COMMON --backend $1 --control-backend $2 --seed $seed \
        --tag $tag --output $OUT/$tag > $OUT/logs/$tag.log 2>&1
      rc=$?
      echo "$(date +%T) done  $tag rc=$rc"
    done
  done
}

lane exact cache 0 > $OUT/logs/lane_exact.log 2>&1 &
lane cache exact 2 > $OUT/logs/lane_cache.log 2>&1 &
wait
echo all done
