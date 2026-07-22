#!/usr/bin/env bash
# natural-bg `full` upper bound with PLY INIT (user's choice) to showcase how much
# a ply + all views enhances. Capped (stop-split 3000) to survive the many-view
# densification runaway. Random-init fallback if ply-init still OOMs.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/natbg_grid.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

rm -rf experiments/scene02__full/{data,train}
log "START scene02__full (natural-bg, PLY init, capped)"
if python scripts/run_experiment.py --exp scene02__full --config configs/project_natbg.yaml >> "$LOG" 2>&1; then
  log "DONE  scene02__full (ply init)"
else
  log "FAIL scene02__full ply-init (likely OOM) -> fallback random init"
  TJ=scenes/scene02/nerf/transforms.json
  BAK=/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad/tj.full.json
  cp "$TJ" "$BAK"; trap 'cp "$BAK" "$TJ"' EXIT INT TERM
  python -c "import json;p='$TJ';d=json.load(open(p));d.pop('ply_file_path',None);json.dump(d,open(p,'w'))"
  rm -rf experiments/scene02__full/{data,train}
  if python scripts/run_experiment.py --exp scene02__full --config configs/project.yaml >> "$LOG" 2>&1; then
    log "DONE  scene02__full (random-init fallback)"
  else
    log "FAIL scene02__full (both)"
  fi
fi
python scripts/compare.py >> "$LOG" 2>&1 || true
