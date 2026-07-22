#!/usr/bin/env bash
# Reproduce the good full: DEFAULT densification at ds=1 (matches the 1920x1080
# render), stopped EARLY at 2000 iters — capturing many healthy COLMAP-seeded
# gaussians BEFORE the many-view densification runaway crosses the 8 GB ceiling
# (ds=1 OOM'd at step ~2700 / 3.3M GSs, so 2000 (~1.3M) should fit). Uses the
# object-cropped ply that is the current scene default.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/rerun_full.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== full SHORT test (ds=1, default densification, 2000 iters) start ==="
rm -rf experiments/scene01_obj__full/train
if python scripts/run_experiment.py --exp scene01_obj__full --iterations 2000 --config configs/project_full_ds1.yaml >> "$LOG" 2>&1; then
  log "DONE full short"
else
  log "FAIL full short"
fi
log "=== full short ALL DONE ==="
