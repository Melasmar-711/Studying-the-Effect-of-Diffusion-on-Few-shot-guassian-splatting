#!/usr/bin/env bash
# HONEST few-shot natural-bg grid: RANDOM init (no ply -> no full-320 geometry leak)
# for all baselines + inpaint + svd. n5_r0 already done (=14.39). Uncapped (random
# init doesn't trigger the ply runaway). Then `full` with PLY init (its own runner).
# Ply is stripped for the duration and always restored (trap).
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/project.yaml            # uncapped, natural-bg
LOG=results/natbg_grid.log
BREAK_EVERY=7200; COOLDOWN=1500
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

TJ=scenes/scene02/nerf/transforms.json
BAK=/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad/tj.randgrid.json
cp "$TJ" "$BAK"
restore(){ cp "$BAK" "$TJ"; echo "[restore] ply restored"; }
trap restore EXIT INT TERM
python -c "import json;p='$TJ';d=json.load(open(p));d.pop('ply_file_path',None);json.dump(d,open(p,'w'));print('random init (ply stripped)')"
log "=== natural-bg RANDOM-INIT grid start ==="

EXPS=(
  scene02__n20_r0 scene02__n10_r0
  scene02__n20_r100_inpaint scene02__n10_r100_inpaint scene02__n5_r100_inpaint
  scene02__n20_r100_svd scene02__n10_r100_svd
  scene02__n20_r200_inpaint scene02__n10_r200_inpaint scene02__n5_r200_inpaint
  scene02__n20_r200_svd scene02__n10_r200_svd
  scene02__n20_r50_inpaint scene02__n10_r50_inpaint scene02__n5_r50_inpaint
  scene02__n20_r25_inpaint scene02__n10_r25_inpaint scene02__n5_r25_inpaint
  scene02__n20_r50_svd scene02__n10_r50_svd
  scene02__n20_r25_svd scene02__n10_r25_svd
)
last_break=$(date +%s); i=0
for exp in "${EXPS[@]}"; do
  i=$((i+1))
  rm -rf "experiments/$exp/train" "experiments/$exp/data"
  log "[$i/${#EXPS[@]}] START $exp"
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
restore; trap - EXIT INT TERM          # restore ply before the ply-init full run
python scripts/compare.py >> "$LOG" 2>&1 || true
log "=== random-init grid ALL DONE -> running full (ply init) ==="
bash scripts/run_natbg_full.sh
log "=== natural-bg re-run COMPLETE (random grid + ply full) ==="
