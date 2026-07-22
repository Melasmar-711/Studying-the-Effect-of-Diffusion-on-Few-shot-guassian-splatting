#!/usr/bin/env bash
# COLMAP on the SEGMENTED (object-on-black) frames -> object-only poses + cloud.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
CONDA=/home/asmar/miniconda3/envs/nerfstudio
export PATH="$CONDA/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA/lib:${LD_LIBRARY_PATH:-}"
LOG=results/scene02_colmap.log
echo "[$(date '+%H:%M:%S')] START ns-process-data (exhaustive, colmap)" | tee "$LOG"
ns-process-data images \
  --data scenes/scene02/frames_seg \
  --output-dir scenes/scene02/nerf \
  --matching-method exhaustive \
  --sfm-tool colmap >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%H:%M:%S')] DONE ns-process-data exit=$rc" | tee -a "$LOG"
