#!/usr/bin/env bash
# Re-run ONLY the `full` upper bound with a FIRMER densification cap
# (project_capped_full.yaml). n20_r0 already completed (PSNR 3.99) at the milder
# cap; full's 281 distinct poses need more trimming to fit 8 GB at ds=2.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/rerun_full.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== re-run full (firmer cap) start ==="
rm -rf experiments/scene01_obj__full/train
if python scripts/run_experiment.py --exp scene01_obj__full --config configs/project_capped_full.yaml >> "$LOG" 2>&1; then
  log "DONE scene01_obj__full"
else
  log "FAIL scene01_obj__full"
fi
python scripts/compare.py >> "$LOG" 2>&1 || true
log "=== re-run full ALL DONE ==="
