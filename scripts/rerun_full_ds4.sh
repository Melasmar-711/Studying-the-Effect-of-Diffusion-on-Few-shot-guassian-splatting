#!/usr/bin/env bash
# `full` upper-bound CEILING REFERENCE at ds=4 (half-res). 281 distinct views
# do not fit in 8 GB at ds=2 across uncapped + 2 densification caps, so this is
# a lower-res, NOT strictly comparable, upper reference (see VERDICT.md).
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/rerun_full.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== re-run full @ ds=4 (ceiling reference) start ==="
rm -rf experiments/scene01_obj__full/train
if python scripts/run_experiment.py --exp scene01_obj__full --config configs/project_full_ds4.yaml >> "$LOG" 2>&1; then
  log "DONE full ds4"
else
  log "FAIL full ds4"
fi
log "=== re-run full ds4 ALL DONE ==="
