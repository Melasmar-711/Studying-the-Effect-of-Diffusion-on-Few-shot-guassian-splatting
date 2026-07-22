#!/usr/bin/env bash
# Reproduce `full` at ds=1 (FULL RES) — matches the surviving good run (19.57
# PSNR, 1920x1080 renders). Counterintuitively, ds=1 keeps densification BOUNDED
# (finer per-gaussian gradients -> fewer splits), while ds=2/4 caused a runaway
# to millions of gaussians -> OOM. Default densification, 7000 iters (=memory's "@7k").
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/rerun_full.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== reproduce full @ ds=1, 7k (default densification) start ==="
rm -rf experiments/scene01_obj__full/train
if python scripts/run_experiment.py --exp scene01_obj__full --iterations 7000 --config configs/project_full_ds1.yaml >> "$LOG" 2>&1; then
  log "DONE full ds1"
else
  log "FAIL full ds1"
fi
log "=== full ds1 ALL DONE ==="
