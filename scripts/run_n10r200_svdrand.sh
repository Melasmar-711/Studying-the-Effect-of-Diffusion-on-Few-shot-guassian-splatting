#!/usr/bin/env bash
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TJ=scenes/scene02/nerf/transforms.json
BAK=/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad/transforms.withply.json
cp "$TJ" "$BAK"
restore(){ cp "$BAK" "$TJ"; echo "[restore] transforms.json (ply) restored"; }
trap restore EXIT INT TERM
python - <<'PY'
import json
p="scenes/scene02/nerf/transforms.json"; d=json.load(open(p)); d.pop("ply_file_path",None)
json.dump(d,open(p,"w"))
print("stripped ply_file_path -> random init")
PY
rm -rf experiments/scene02__n10_r200_svdrand/{data,train}
echo "[$(date '+%H:%M:%S')] START n10_r200_svdrand (random init)"
python scripts/run_experiment.py --exp scene02__n10_r200_svdrand 2>&1 | tail -2
echo "[$(date '+%H:%M:%S')] DONE"
