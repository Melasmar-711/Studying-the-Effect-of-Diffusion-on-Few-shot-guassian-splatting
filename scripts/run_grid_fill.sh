#!/usr/bin/env bash
# Fill the remaining feasible grid cells:
#   A) n5 SVD, RANDOM init (completes the honest grid)
#   B) n20 SVD OWN-PLY for r25/50/200 (r100 already done; uses the n20 real+svd ply)
# n5/n10 OWN-PLY is infeasible (few natural-bg views don't register in COLMAP).
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/grid_fill.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

TJ=scenes/scene02/nerf/transforms.json
FULL=scenes/scene02/nerf/sparse_pc.ply
SP=/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad
cp "$TJ" "$SP/tj.fill.json"; cp "$FULL" "$SP/ply.fill.ply"
restore(){ cp "$SP/tj.fill.json" "$TJ"; cp "$SP/ply.fill.ply" "$FULL"; echo "[restore] transforms+ply restored"; }
trap restore EXIT INT TERM

run(){ local exp="$1" cfg="$2"; rm -rf "experiments/$exp/"{train,data}
  log "START $exp"
  python scripts/run_experiment.py --exp "$exp" --config "$cfg" >> "$LOG" 2>&1 && log "DONE  $exp" || log "FAIL  $exp"; }

# ===== A) n5 SVD, random init (strip ply) =====
log "=== Part A: n5 SVD random-init ==="
python -c "import json;p='$TJ';d=json.load(open(p));d.pop('ply_file_path',None);json.dump(d,open(p,'w'));print('random init')"
for r in 100 200 50 25; do run "scene02__n5_r${r}_svd" configs/project.yaml; done
cp "$SP/tj.fill.json" "$TJ"     # restore ply_file_path for Part B

# ===== B) n20 SVD own-ply, remaining ratios =====
PLY=data/svd/scene02/honest_ply/n20_svd_natural.ply
if [ -f "$PLY" ]; then
  log "=== Part B: n20 SVD own-ply (r25/50/200), ply=$(basename "$PLY") ==="
  cp "$PLY" "$FULL"
  for r in 25 50 200; do run "scene02__n20_r${r}_svd_ownply" configs/project_natbg.yaml; done
  cp "$SP/ply.fill.ply" "$FULL"
else
  log "Part B skipped: no $PLY"
fi
python scripts/compare.py >> "$LOG" 2>&1 || true
log "=== grid fill DONE ==="
