#!/usr/bin/env bash
# `full` upper bound with densification FROZEN at step 1500 (before the
# exponential runaway that OOMs: splits double every 100 steps after ~1500,
# reaching 3.3M gaussians by step 2700). ~79k gaussians is ample for one object.
# ds=2 to match the grid's resolution. Default everything else, 7000 iters.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/rerun_full.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== full bounded (stop-split-at 1500, ds=2, 7k) start ==="
rm -rf experiments/scene01_obj__full/train
if python scripts/run_experiment.py --exp scene01_obj__full --iterations 7000 --config configs/project_full_bounded.yaml >> "$LOG" 2>&1; then
  log "DONE full bounded"
else
  log "FAIL full bounded"
fi
log "=== full bounded ALL DONE ==="
