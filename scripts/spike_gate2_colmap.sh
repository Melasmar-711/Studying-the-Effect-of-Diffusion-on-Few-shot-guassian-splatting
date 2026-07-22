#!/usr/bin/env bash
# Gate 2: does COLMAP register the SVD-generated frames into the REAL orbit?
# Real frames (1920x1080) and generated frames (896x512) are put in SEPARATE
# folders so `single_camera_per_folder` lets COLMAP estimate the generated
# group's intrinsics independently (do NOT force shared focal length).
set -uo pipefail
cd /home/asmar/GS_VR
CONDA=/home/asmar/miniconda3/envs/nerfstudio
export PATH="$CONDA/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA/lib:${LD_LIBRARY_PATH:-}"

WORK=/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad/gate2
IMG="$WORK/images"
rm -rf "$WORK"; mkdir -p "$IMG/real" "$IMG/gen" "$WORK/sparse"
LOG=results/spikes/gate2_colmap.log
: > "$LOG"

# --- real anchors: every 4th frame + the last few near the seed (319) ---
python - <<'PY'
import shutil
from pathlib import Path
src = Path("scenes/scene02/frames_raw")
dst = Path("/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad/gate2/images/real")
idx = sorted(set(list(range(0, 320, 4)) + [317, 318, 319]))
n = 0
for i in idx:
    f = src / f"frame_{i:05d}.jpg"
    if f.exists():
        shutil.copy(f, dst / f.name); n += 1
print(f"real anchors copied: {n}")
PY

# --- generated frames from both clips ---
python - <<'PY'
import shutil
from pathlib import Path
base = Path("results/spikes/svd")
dst = Path("/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad/gate2/images/gen")
n = 0
for clip, tagn in [("frame_00319_mb127", "A"), ("frame_00319_mb180", "B")]:
    for f in sorted((base / clip).glob("gen_*.png")):
        shutil.copy(f, dst / f"gen{tagn}_{f.stem.split('_')[1]}.png"); n += 1
print(f"generated frames copied: {n}")
PY

echo "[$(date '+%H:%M:%S')] feature_extractor" | tee -a "$LOG"
colmap feature_extractor --database_path "$WORK/db.db" --image_path "$IMG" \
  --ImageReader.single_camera_per_folder 1 --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu 1 >> "$LOG" 2>&1

echo "[$(date '+%H:%M:%S')] exhaustive_matcher" | tee -a "$LOG"
colmap exhaustive_matcher --database_path "$WORK/db.db" \
  --SiftMatching.use_gpu 1 >> "$LOG" 2>&1

echo "[$(date '+%H:%M:%S')] mapper" | tee -a "$LOG"
colmap mapper --database_path "$WORK/db.db" --image_path "$IMG" \
  --output_path "$WORK/sparse" >> "$LOG" 2>&1

MODEL="$WORK/sparse/0"
if [ -d "$MODEL" ]; then
  colmap model_converter --input_path "$MODEL" --output_path "$MODEL" --output_type TXT >> "$LOG" 2>&1
  echo "[$(date '+%H:%M:%S')] model_analyzer" | tee -a "$LOG"
  colmap model_analyzer --path "$MODEL" >> "$LOG" 2>&1
  echo "MODEL_DIR=$MODEL"
else
  echo "NO MODEL RECONSTRUCTED"; ls "$WORK/sparse" >> "$LOG" 2>&1
fi
echo "[$(date '+%H:%M:%S')] DONE" | tee -a "$LOG"
