#!/usr/bin/env bash
# Focused zero123 test on n5/n10 at r100,r200 (+ zero123+inpaint combo). Compares
# genuine novel viewpoints against inpaint's same-pose regularization.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/scene02_zero123.log
BREAK_EVERY=7200; COOLDOWN=1200
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
EXPS=(
  scene02__n5_r100_zero123 scene02__n5_r200_zero123
  "scene02__n5_r100_zero123+inpaint" "scene02__n5_r200_zero123+inpaint"
  scene02__n10_r100_zero123 scene02__n10_r200_zero123
  "scene02__n10_r100_zero123+inpaint" "scene02__n10_r200_zero123+inpaint"
)
log "=== scene02 zero123 focused run start (${#EXPS[@]} exps) ==="
last=$(date +%s); i=0
for exp in "${EXPS[@]}"; do
  i=$((i+1)); rm -rf "experiments/$exp/train"
  log "[$i/${#EXPS[@]}] START $exp"
  if python scripts/run_experiment.py --exp "$exp" >> "$LOG" 2>&1; then log "[$i/${#EXPS[@]}] DONE  $exp"; else log "[$i/${#EXPS[@]}] FAIL  $exp"; fi
  now=$(date +%s); if [ $((now-last)) -ge "$BREAK_EVERY" ] && [ "$i" -lt "${#EXPS[@]}" ]; then log "cooldown ${COOLDOWN}s"; sleep "$COOLDOWN"; last=$(date +%s); fi
done
python scripts/compare.py >> "$LOG" 2>&1 || true
log "=== scene02 zero123 focused run ALL DONE ==="
