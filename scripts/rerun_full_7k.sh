#!/usr/bin/env bash
# Reproduce the GOOD `full` upper bound at 7k iterations (DEFAULT densification,
# ds=2). full converges by ~7k (~19.6 masked PSNR, sharp render); 30k OOMs purely
# because densification keeps spawning gaussians (3.65M by step 7k) past the 8 GB
# ceiling — the extra steps add no quality. This is the honest upper bound on 8 GB.
set -uo pipefail
cd /home/asmar/GS_VR
# shellcheck disable=SC1091
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/rerun_full.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== reproduce full @ 7k (default densification) start ==="
rm -rf experiments/scene01_obj__full/train
if python scripts/run_experiment.py --exp scene01_obj__full --iterations 7000 >> "$LOG" 2>&1; then
  log "DONE full 7k"
else
  log "FAIL full 7k"
fi
log "=== full 7k ALL DONE ==="
