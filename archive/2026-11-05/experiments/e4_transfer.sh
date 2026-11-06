#!/usr/bin/env bash
# e4_transfer.sh -- TODO.md 2026-11-01, E4: frozen E2 config (REF) and E3 head (HEAD) vs H0 on
# avGFP-320, GB1-320, GB1-1280, replicates 0-2, copied WITHOUT retuning (robustness to
# dataset / label-budget change). Dev manifests *_tune_ds{N}_r{r}. Same schedule (400 steps),
# loss, selection as E2/E3. Reuses e3_avgfp_heads.sh with STAGE=pilot_e4. GPU 0, serial.
set -u
: "${REF:?set REF, e.g. F_1e-5}"; : "${HEAD:?set HEAD, e.g. sum_mean_count}"
cd ~/mutant_cache
for tn in "avgfp 320" "gb1 320" "gb1 1280"; do
  set -- $tn
  TASK=$1 N=$2 STAGE=pilot_e4 HEADS="mean $HEAD" REF=$REF bash experiments/e3_avgfp_heads.sh
done
