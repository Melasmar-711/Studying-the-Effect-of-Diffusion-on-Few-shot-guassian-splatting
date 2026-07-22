#!/usr/bin/env bash
# TEST the user's hypothesis: the good full run (19.57) used the ORIGINAL room
# point cloud (sparse_pc.ply, diag 109) as init, not the object-cropped one
# (sparse_pc_object.ply, diag 8.8). Cropping the cloud to a tiny dense blob while
# cameras stay room-scale breaks splatfacto's densification calibration -> runaway.
# Here we temporarily point the scene at sparse_pc.ply and run full with DEFAULT
# densification (no cap), then restore. If the hypothesis holds: no runaway, ~19.5 PSNR.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/rerun_full.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

TF=output_masked/transforms.json
cp "$TF" "$TF.bak_roomtest"
python - <<'PY'
import json
p="output_masked/transforms.json"; m=json.load(open(p))
m["ply_file_path"]="sparse_pc.ply"
json.dump(m, open(p,"w"), indent=2)
print("[roomtest] set ply_file_path -> sparse_pc.ply")
PY

log "=== full ROOM-PLY test (sparse_pc.ply, DEFAULT densification, ds=2, 7k) start ==="
rm -rf experiments/scene01_obj__full/train
if python scripts/run_experiment.py --exp scene01_obj__full --iterations 7000 >> "$LOG" 2>&1; then
  log "DONE full roomply"
else
  log "FAIL full roomply"
fi

# restore object-cropped ply as the scene default
mv "$TF.bak_roomtest" "$TF"
log "[roomtest] restored ply_file_path -> sparse_pc_object.ply"
log "=== full roomply ALL DONE ==="
