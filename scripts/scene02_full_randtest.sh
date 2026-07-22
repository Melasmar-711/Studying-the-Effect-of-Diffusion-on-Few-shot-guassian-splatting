#!/usr/bin/env bash
# Diagnostic: scene02 full with RANDOM init (strip ply) @7k. If poses/scale are
# correct, 280 views reconstruct a sharp object (like scene01 full=19.75). If black
# -> a geometry problem to fix before the grid. Restores ply after.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/scene02_full_randtest.log
TF=scenes/scene02/nerf/transforms.json
cp "$TF" "$TF.bak_rand"
python - <<'PY'
import json; p="scenes/scene02/nerf/transforms.json"; m=json.load(open(p)); m.pop("ply_file_path",None); json.dump(m,open(p,"w"),indent=2); print("random init (ply stripped)")
PY
echo "[start] scene02 full randinit @7k" | tee "$LOG"
rm -rf experiments/scene02__full/train
python scripts/run_experiment.py --exp scene02__full --iterations 7000 >> "$LOG" 2>&1
echo "[done] exit $?" | tee -a "$LOG"
mv "$TF.bak_rand" "$TF"
echo "[restored ply]" | tee -a "$LOG"
