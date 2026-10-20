# LoRA pilot (2026-10-20): head-only vs +variant LoRA vs +variant+shared LoRA on
# Tsuboyama, 20k train variants, 1 epoch, val split. As run (GPU 1 was occupied).
S=~/mutant_cache_scratch; P=~/anaconda3/envs/mutant_cache/bin/python
COMMON="--dataset tsuboyama --split protein_level --eval-split val --backbone esmc_300m --max-train-variants 20000 --epochs 1 --eval-every 200 --baselines zero_shot --seed 0"
run() { /usr/bin/time -f "WALL %e s  MAXRSS %M KB" $P -m mutant_cache.harness $COMMON --tag "$1" --output $S/runs/lora_pilot_$1 "${@:2}" > $S/logs/lora_pilot_$1.log 2>&1; echo "exit=$?" >> $S/logs/lora_pilot_$1.log; }
( CUDA_VISIBLE_DEVICES=0 run head; CUDA_VISIBLE_DEVICES=0 run shared --lora-r 8 --shared-lora-r 8 ) &
( CUDA_VISIBLE_DEVICES=2 run variant --lora-r 8 ) &
wait
# Results (val, 17 domains, 25,675 variants; Spearman depth1 / depth2 / all; eps rho; wall):
#   zero-shot      0.343 / 0.293 / 0.368
#   head only      0.625 / 0.434 / 0.631   eps 0.012   279 s
#   + variant r8   0.668 / 0.501 / 0.671   eps 0.094   401 s
#   + shared r8    0.681 / 0.497 / 0.688   eps 0.016   585 s
