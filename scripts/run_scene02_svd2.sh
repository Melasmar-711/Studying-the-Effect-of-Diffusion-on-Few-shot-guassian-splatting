#!/usr/bin/env bash
# Complete the SVD curves. n10 fits UNCAPPED (clean apples-to-apples vs n10
# inpaint/baseline). n20 OOMs uncapped (novel-view densification runaway) so it
# uses the CAPPED config (compare to the capped n20_r0=6.34). Most-informative
# ratio first. r100 already done: n10=6.95 (uncapped), n20=6.28 (capped).
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/scene02_svd.log
BREAK_EVERY=7200; COOLDOWN=1500
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== scene02 SVD curves (part 2) ==="

# exp : config  (empty config = default/uncapped)
declare -a EXPS=(
  "scene02__n10_r200_svd|"
  "scene02__n20_r200_svd|configs/project_scene02_capped.yaml"
  "scene02__n10_r50_svd|"
  "scene02__n20_r50_svd|configs/project_scene02_capped.yaml"
  "scene02__n10_r25_svd|"
  "scene02__n20_r25_svd|configs/project_scene02_capped.yaml"
)
last_break=$(date +%s); i=0
for row in "${EXPS[@]}"; do
  exp="${row%%|*}"; conf="${row#*|}"; i=$((i+1))
  rm -rf "experiments/$exp/train"
  cfgarg=(); [ -n "$conf" ] && cfgarg=(--config "$conf")
  log "[$i/${#EXPS[@]}] START $exp ${conf:-(uncapped)}"
  if python scripts/run_experiment.py --exp "$exp" "${cfgarg[@]}" >> "$LOG" 2>&1; then
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
log "=== scene02 SVD curves (part 2) ALL DONE ==="
