#!/usr/bin/env bash
# scene02 grid: 3 few-shot baselines + 12 inpaint runs @30k, PLY init (default).
# full is already done (random init, 25.0). Thermal cooldown every ~2h.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/scene02_grid.log
BREAK_EVERY=7200; COOLDOWN=1500
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== scene02 grid start ==="

BASELINES=(scene02__n5_r0 scene02__n10_r0 scene02__n20_r0)
mapfile -t AUG < <(tail -n +2 configs/experiments/index.csv | cut -d, -f1 \
                   | grep '^scene02__' | grep '_inpaint$')
EXPS=("${BASELINES[@]}" "${AUG[@]}")
log "queue (${#EXPS[@]}): ${EXPS[*]}"

last_break=$(date +%s); i=0
for exp in "${EXPS[@]}"; do
  i=$((i+1))
  rm -rf "experiments/$exp/train"
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
log "=== scene02 grid ALL DONE ==="
