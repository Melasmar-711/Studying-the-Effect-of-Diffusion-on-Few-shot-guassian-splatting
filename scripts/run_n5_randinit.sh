#!/usr/bin/env bash
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TJ=scenes/scene02/nerf/transforms.json
BAK=/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad/tj.n5rand.json
cp "$TJ" "$BAK"; trap 'cp "$BAK" "$TJ"; echo "[restore] ply restored"' EXIT INT TERM
python -c "import json;p='$TJ';d=json.load(open(p));d.pop('ply_file_path',None);json.dump(d,open(p,'w'));print('random init (no ply)')"
rm -rf experiments/scene02__n5_r0_randinit/{data,train}
echo "[$(date '+%H:%M:%S')] START n5_r0_randinit (random init, uncapped)"
python scripts/run_experiment.py --exp scene02__n5_r0_randinit --config configs/project.yaml 2>&1 | tail -3
echo "[$(date '+%H:%M:%S')] DONE"
