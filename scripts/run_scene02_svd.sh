#!/usr/bin/env bash
# scene02 SVD-augmentation arm: 8 runs (n10 & n20 x ratios 25/50/100/200) @30k,
# PLY init. Synthetic pool = COLMAP-registered SVD novel neighbours.
# CAPPED config (stop-split 3000): novel-view synthetic frames trigger the same
# densification runaway the bare baseline did (inpaint frames do NOT) -> without
# the cap n20 OOMs at ~step 3800. Same cap that recovered n20_r0 (=6.34).
# Baselines (r0) already exist. Thermal cooldown every ~2h (8 GB laptop GPU).
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/scene02_svd.log
BREAK_EVERY=7200; COOLDOWN=1500
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== scene02 SVD arm start ==="

# most-informative first, so a systemic problem shows early
EXPS=(
  scene02__n20_r100_svd scene02__n10_r100_svd
  scene02__n20_r200_svd scene02__n10_r200_svd
  scene02__n20_r50_svd  scene02__n10_r50_svd
  scene02__n20_r25_svd  scene02__n10_r25_svd
)
log "queue (${#EXPS[@]}): ${EXPS[*]}"

last_break=$(date +%s); i=0
for exp in "${EXPS[@]}"; do
  i=$((i+1))
  rm -rf "experiments/$exp/train"
  log "[$i/${#EXPS[@]}] START $exp"
  if python scripts/run_experiment.py --exp "$exp" \
        --config configs/project_scene02_capped.yaml >> "$LOG" 2>&1; then
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
log "=== scene02 SVD arm ALL DONE ==="
