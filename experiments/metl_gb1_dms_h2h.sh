#!/usr/bin/env bash
# metl_gb1_dms_h2h.sh -- head-to-head vs METL on its own GB1 DMS benchmark
# (HANDOFF_2026-10-24 §4.1, the "cheaper alternative": scalar task, no model change).
#
# Splits: METL's resampled grid, every replicate (ds10 x101 ... ds20480 x3),
# built by mutant_cache.data.metl.build_dms_resampled. Test = METL's full GB1 DMS
# supertest (53,609), identical for every size/rep. Metric = median test Spearman
# over replicates, as in the paper.
# Protocol: --select-on-val (best epoch chosen on METL's val rows, test scored once).
# Hyperparameters carried over from the Rosetta probe (rank loss, lr 3e-3, LoRA r=8),
# NOT tuned on DMS. Epochs give ~400 optimizer steps, clamped to [20, 300] epochs.
# Also fits the one-hot ridge baseline on the same train rows (METL's "Linear").
#
#   cd ~/mutant_cache && bash experiments/metl_gb1_dms_h2h.sh
# Resumable (skips finished runs). One GPU, serial. MAX_REPS=N to cap replicates.
set -u
cd ~/mutant_cache
PY=~/anaconda3/envs/mutant_cache/bin/python
OUT=/home/andrew/mutant_cache_scratch/runs/metl_gb1_dms_h2h
SPLITS=/home/andrew/mutant_cache_scratch/processed/splits
MAX_REPS=${MAX_REPS:-1000}
mkdir -p $OUT/logs

for n in 10 20 40 80 160 320 640 1280 2560 5120 10240 20480; do
  ntr=$(( n * 8 / 10 ))
  spe=$(( (ntr + 255) / 256 ))
  ep=$(( (400 + spe - 1) / spe )); [ $ep -gt 300 ] && ep=300; [ $ep -lt 20 ] && ep=20
  for f in $(ls $SPLITS/metl_gb1_dms_ds${n}_r*.parquet | sort -V); do
    r=$(basename $f .parquet); r=${r##*_r}
    [ $r -ge $MAX_REPS ] && continue
    tag=ds${n}_r${r}
    [ -f $OUT/$tag/results.json ] && continue
    echo "$(date +%T) start $tag (epochs $ep)"
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -m mutant_cache.harness \
      --dataset metl_gb1_dms --split metl_gb1_dms_ds${n}_r${r} \
      --eval-split test --select-on-val --min-variants-per-set 1 \
      --lora-r 8 --loss rank --lr 3e-3 \
      --epochs $ep --batch-size 256 --eval-batch-size 1024 \
      --control-backend none --baselines zero_shot onehot_ridge --epsilon-depths \
      --seed $r --tag $tag --output $OUT/$tag --out-root $OUT \
      > $OUT/logs/$tag.log 2>&1
    rc=$?
    echo "$(date +%T) done  $tag rc=$rc"
  done
done
echo all done
