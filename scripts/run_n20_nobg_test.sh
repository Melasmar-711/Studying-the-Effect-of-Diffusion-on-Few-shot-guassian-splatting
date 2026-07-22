#!/usr/bin/env bash
# THROWAWAY test (not part of the pipeline): train the n20 baseline on NATURAL-BG
# (unsegmented) images at the same poses, eval the same masked way, to see if the
# object-on-black segmentation itself hurts the baseline. Restores the n20 split.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPLIT=data/splits/scene02/n20/transforms.json
BAK=/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad/n20_split.seg.json
cp "$SPLIT" "$BAK"
restore(){ cp "$BAK" "$SPLIT"; echo "[restore] n20 split (segmented) restored"; }
trap restore EXIT INT TERM

# point the n20 split at the natural-bg images (same poses)
python - <<'PY'
import json
from pathlib import Path
p="data/splits/scene02/n20/transforms.json"; d=json.load(open(p))
base=Path("/home/asmar/GS_VR/data/svd/scene02/nobg_test/images")
for f in d["frames"]:
    k=int(Path(f["file_path"]).stem.split("_")[1])
    f["file_path"]=str(base/f"frame_{k:05d}.jpg")
json.dump(d,open(p,"w"))
print("n20 split -> natural-bg images")
PY

rm -rf experiments/scene02__n20_r0_nobg/{data,train}
echo "[$(date '+%H:%M:%S')] START n20_r0_nobg (natural-bg, capped)"
python scripts/run_experiment.py --exp scene02__n20_r0_nobg --config configs/project_scene02_capped.yaml 2>&1 | tail -3
echo "[$(date '+%H:%M:%S')] DONE n20_r0_nobg"
