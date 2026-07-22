#!/usr/bin/env bash
# Recover scene02__n20_r0 (OOM'd: ply-init many-view densification runaway).
# ply init + densification cap so it finishes -> the n20 baseline anchor.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/scene02_n20r0_recover.log
echo "[start] scene02__n20_r0 capped" | tee "$LOG"
rm -rf experiments/scene02__n20_r0/train
python scripts/run_experiment.py --exp scene02__n20_r0 --config configs/project_scene02_capped.yaml >> "$LOG" 2>&1
echo "[done] exit $?" | tee -a "$LOG"
