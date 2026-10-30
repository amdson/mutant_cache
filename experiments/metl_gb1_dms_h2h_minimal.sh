#!/usr/bin/env bash
# metl_gb1_dms_h2h_minimal.sh -- head-to-head vs METL on GB1 DMS, REAL test set
# (METL's supertest, 53,609), every resampled replicate, with the config chosen on
# the 5k DEV set (experiments/dms_minimal.sh): --minimal, full fine-tuning (fp32,
# backbone lr 1e-5), rank loss, head lr 3e-3, select on val loss.
# Epochs as metl_gb1_dms_h2h.sh: ~400 steps, clamped to [20, 300].
# Ordered rep-major (rep 0 of every size, then rep 1, ...) so all sizes fill early.
# Resumable. One GPU, serial.
set -u
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/metl_gb1_dms_h2h_minimal
SPLITS=/home/andrew/mutant_cache_scratch/processed/splits
mkdir -p $OUT/logs
SIZES="10 20 40 80 160 320 640 1280 2560 5120 10240 20480"
for r in $(seq 0 100); do
  for n in $SIZES; do
    [ -f $SPLITS/metl_gb1_dms_ds${n}_r${r}.parquet ] || continue
    ntr=$(( n * 8 / 10 )); spe=$(( (ntr + 255) / 256 ))
    ep=$(( (400 + spe - 1) / spe )); [ $ep -gt 300 ] && ep=300; [ $ep -lt 20 ] && ep=20
    tag=ds${n}_r${r}
    [ -f $OUT/$tag/results.json ] && continue
    echo "$(date +%T) start $tag (epochs $ep)"
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.harness \
      --dataset metl_gb1_dms --split metl_gb1_dms_ds${n}_r${r} \
      --eval-split test --select-on-val --min-variants-per-set 1 \
      --minimal --train-backbone --precision fp32 --backbone-lr 1e-5 \
      --loss rank --lr 3e-3 \
      --epochs $ep --batch-size 256 --eval-batch-size 1024 \
      --control-backend none --baselines none --epsilon-depths \
      --seed $r --tag $tag --output $OUT/$tag --out-root $OUT \
      > $OUT/logs/$tag.log 2>&1
    rc=$?
    echo "$(date +%T) done  $tag rc=$rc"
  done
done
echo all done
