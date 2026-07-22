#!/usr/bin/env bash
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/scene02_smoke.log
echo "[start] scene02__n10_r0 smoke @3000" | tee "$LOG"
python scripts/run_experiment.py --exp scene02__n10_r0 --iterations 3000 >> "$LOG" 2>&1
echo "[done] exit $?" | tee -a "$LOG"
