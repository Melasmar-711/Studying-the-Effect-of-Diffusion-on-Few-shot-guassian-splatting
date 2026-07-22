#!/usr/bin/env bash
# EXTRA arm (does not touch the random-init grid): every experiment inits from a
# ply built by COLMAP on ITS OWN natural-bg data.
#   baseline / inpaint  -> ply from the N real views      (n{N}_real.ply)
#   svd                 -> ply from N real + SVD frames    (n{N}_svd.ply)
# n5/n10 real plys may not register (too sparse) -> those experiments are SKIPPED.
# Results land under *_ownply exp-ids. Capped config (ply init). GPU COLMAP.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/project_natbg.yaml
LOG=results/ownply_grid.log
HP=data/svd/scene02/honest_ply
BREAK_EVERY=7200; COOLDOWN=1500
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== own-ply grid start ==="

# 1) build per-N plys (GPU COLMAP; never halt on failure)
for n in 20 10 5; do
  log "build n$n REAL ply (COLMAP on $n real views)"
  python scripts/build_honest_ply.py --n "$n" --natural --gpu \
    --crop-ref scenes/scene02/nerf/sparse_pc.ply --out "$HP/n${n}_real.ply" 2>&1 | tail -2 | tee -a "$LOG" || true
done
for n in 20 10; do
  log "build n$n SVD ply (COLMAP on $n real + SVD)"
  python scripts/build_honest_ply.py --n "$n" --with-gen --natural --gpu \
    --crop-ref scenes/scene02/nerf/sparse_pc.ply --out "$HP/n${n}_svd.ply" 2>&1 | tail -2 | tee -a "$LOG" || true
done
log "plys available: $(ls "$HP"/n{5,10,20}_real.ply "$HP"/n{10,20}_svd.ply 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"

# 2) run each _ownply experiment from its own-data ply
FULL=scenes/scene02/nerf/sparse_pc.ply
BAK=/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad/sparse_pc.ownply.bak.ply
cp "$FULL" "$BAK"
trap 'cp "$BAK" "$FULL"; echo "[restore] full ply restored"' EXIT INT TERM

ply_for(){ local e="$1" n; n=$(echo "$e" | sed -E 's/.*n([0-9]+)_r.*/\1/')
  if [[ "$e" == *_svd_ownply ]]; then echo "$HP/n${n}_svd.ply"; else echo "$HP/n${n}_real.ply"; fi; }

EXPS=()
for n in 20 10; do
  EXPS+=("scene02__n${n}_r0_ownply")
  for r in 100 200 50 25; do EXPS+=("scene02__n${n}_r${r}_inpaint_ownply" "scene02__n${n}_r${r}_svd_ownply"); done
done
EXPS+=("scene02__n5_r0_ownply")
for r in 100 200 50 25; do EXPS+=("scene02__n5_r${r}_inpaint_ownply"); done

log "queue (${#EXPS[@]}): ${EXPS[*]}"
last_break=$(date +%s); i=0
for exp in "${EXPS[@]}"; do
  i=$((i+1)); ply=$(ply_for "$exp")
  if [ ! -f "$ply" ]; then log "[$i/${#EXPS[@]}] SKIP $exp (no ply $(basename "$ply"))"; continue; fi
  cp "$ply" "$FULL"
  rm -rf "experiments/$exp/train" "experiments/$exp/data"
  log "[$i/${#EXPS[@]}] START $exp  (ply=$(basename "$ply"))"
  if python scripts/run_experiment.py --exp "$exp" --config "$CFG" >> "$LOG" 2>&1; then
    log "[$i/${#EXPS[@]}] DONE  $exp"
  else
    log "[$i/${#EXPS[@]}] FAIL  $exp"
  fi
  now=$(date +%s)
  if [ $(( now - last_break )) -ge "$BREAK_EVERY" ] && [ "$i" -lt "${#EXPS[@]}" ]; then
    log "thermal cooldown ${COOLDOWN}s"; sleep "$COOLDOWN"; last_break=$(date +%s)
  fi
done
python scripts/compare.py >> "$LOG" 2>&1 || true
log "=== own-ply grid DONE ==="
