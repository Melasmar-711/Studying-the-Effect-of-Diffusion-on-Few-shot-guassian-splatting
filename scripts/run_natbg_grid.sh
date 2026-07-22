#!/usr/bin/env bash
# NATURAL-BACKGROUND re-run of the whole study (segmentation was crippling few-shot).
# Training = natural-bg images; eval = masked (object). Inpaint unchanged (object-
# focused). Capped config (object ply + fabric). `full` handled separately.
# Ordered most-informative-first: baselines -> r100 aug -> r200 aug -> r50 -> r25.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/project_natbg.yaml
LOG=results/natbg_grid.log
BREAK_EVERY=7200; COOLDOWN=1500
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== natural-bg grid start ==="

EXPS=(
  scene02__n20_r0 scene02__n10_r0 scene02__n5_r0
  scene02__n20_r100_inpaint scene02__n10_r100_inpaint scene02__n5_r100_inpaint
  scene02__n20_r100_svd scene02__n10_r100_svd
  scene02__n20_r200_inpaint scene02__n10_r200_inpaint scene02__n5_r200_inpaint
  scene02__n20_r200_svd scene02__n10_r200_svd
  scene02__n20_r50_inpaint scene02__n10_r50_inpaint scene02__n5_r50_inpaint
  scene02__n20_r25_inpaint scene02__n10_r25_inpaint scene02__n5_r25_inpaint
  scene02__n20_r50_svd scene02__n10_r50_svd
  scene02__n20_r25_svd scene02__n10_r25_svd
)
log "queue (${#EXPS[@]}): ${EXPS[*]}"

last_break=$(date +%s); i=0
for exp in "${EXPS[@]}"; do
  i=$((i+1))
  rm -rf "experiments/$exp/train" "experiments/$exp/data"
  log "[$i/${#EXPS[@]}] START $exp"
  if python scripts/run_experiment.py --exp "$exp" --config "$CFG" >> "$LOG" 2>&1; then
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
log "=== natural-bg grid ALL DONE ==="
