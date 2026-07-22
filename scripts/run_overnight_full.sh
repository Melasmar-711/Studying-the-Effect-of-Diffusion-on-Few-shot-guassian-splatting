#!/usr/bin/env bash
# Unattended overnight runner: the FULL object-centric grid (16 runs) at 30k.
# Order: few-shot baselines -> full upper bound -> 12 inpaint ratio experiments.
# Thermal cooldown (25 min) after every ~2 h of accumulated runtime. Continues
# past failures; clears each run dir first; builds the dashboard at the end.
set -uo pipefail
cd /home/asmar/GS_VR
# shellcheck disable=SC1091
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr

LOG=results/run_overnight.log
BREAK_EVERY=7200; COOLDOWN=1500
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

: > results/registry.jsonl            # fresh registry (drop throwaway 7k rows)
log "=== overnight FULL grid start ==="

BASELINES=(scene01_obj__n5_r0 scene01_obj__n10_r0 scene01_obj__n20_r0 scene01_obj__full)
mapfile -t AUG < <(tail -n +2 configs/experiments/index.csv | cut -d, -f1 \
                   | grep '^scene01_obj__' | grep -vE '_r0$|__full$')
EXPS=("${BASELINES[@]}" "${AUG[@]}")
log "queue (${#EXPS[@]}): ${EXPS[*]}"

last_break=$(date +%s); i=0
for exp in "${EXPS[@]}"; do
  i=$((i+1))
  rm -rf "experiments/$exp/train"      # avoid nerfstudio 'timestamp exists' errors
  log "[$i/${#EXPS[@]}] START $exp"
  if python scripts/run_experiment.py --exp "$exp" >> "$LOG" 2>&1; then
    log "[$i/${#EXPS[@]}] DONE  $exp"
  else
    log "[$i/${#EXPS[@]}] FAIL  $exp"
  fi
  now=$(date +%s)
  if [ $(( now - last_break )) -ge "$BREAK_EVERY" ] && [ "$i" -lt "${#EXPS[@]}" ]; then
    log "thermal cooldown ${COOLDOWN}s"; sleep "$COOLDOWN"; last_break=$(date +%s)
  fi
done

python scripts/compare.py >> "$LOG" 2>&1 || true
log "=== overnight FULL grid ALL DONE ==="
