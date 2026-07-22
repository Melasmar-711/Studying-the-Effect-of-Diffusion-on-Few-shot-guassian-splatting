#!/usr/bin/env bash
# SVD runs initialised from the SHORTENED+SVD-COMPLETED ply (COLMAP on 40 real +
# their SVD completion, 15k object pts) instead of the full-320 ply. Baselines/
# inpaint untouched. Eval = real held-out photos (standard). n10 uncapped, n20
# capped (match the full-ply svd runs). The full ply is swapped out only for the
# duration and ALWAYS restored (trap).
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/scene02_svdshort.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

FULL=scenes/scene02/nerf/sparse_pc.ply
BAK=/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad/sparse_pc.FULL.ply
SHORT=data/svd/scene02/honest_ply/short40_svd.ply
cp "$FULL" "$BAK"
restore(){ cp "$BAK" "$FULL"; echo "[restore] full ply restored"; }
trap restore EXIT INT TERM
cp "$SHORT" "$FULL"
log "=== SVD short-ply arm start (init ply = $(sed -n 's/element vertex //p' "$SHORT"|head -1) pts) ==="

# exp : config (empty = uncapped)
declare -a EXPS=(
  "scene02__n20_r100_svdshort|configs/project_scene02_capped.yaml"
  "scene02__n10_r100_svdshort|"
  "scene02__n20_r200_svdshort|configs/project_scene02_capped.yaml"
  "scene02__n10_r200_svdshort|"
)
i=0
for row in "${EXPS[@]}"; do
  exp="${row%%|*}"; conf="${row#*|}"; i=$((i+1))
  rm -rf "experiments/$exp/train" "experiments/$exp/data"   # force re-assemble with the swapped ply
  cfgarg=(); [ -n "$conf" ] && cfgarg=(--config "$conf")
  log "[$i/${#EXPS[@]}] START $exp ${conf:-(uncapped)}"
  if python scripts/run_experiment.py --exp "$exp" "${cfgarg[@]}" >> "$LOG" 2>&1; then
    log "[$i/${#EXPS[@]}] DONE  $exp"
  else
    log "[$i/${#EXPS[@]}] FAIL  $exp"
  fi
done
python scripts/compare.py >> "$LOG" 2>&1 || true
log "=== SVD short-ply arm ALL DONE ==="
