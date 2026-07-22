#!/usr/bin/env bash
# Compare INIT for the n10_r0 baseline: current grid used object-ply init (PSNR
# 4.30). Here we run the SAME settings (30k, ds=2) but with RANDOM init (strip
# ply), to see if it fogs/collapses differently. n10 = good judging point (not too
# many views to trigger the many-view runaway, not too few to be trivial).
# Registry backed up beforehand so we can keep the canonical ply-init row.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/rerun_n10rand.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

TF=output_masked/transforms.json
cp "$TF" "$TF.bak_n10rand"
python - <<'PY'
import json
p="output_masked/transforms.json"; m=json.load(open(p)); m.pop("ply_file_path", None)
json.dump(m, open(p,"w"), indent=2); print("[n10rand] random init (ply stripped)")
PY

log "=== n10_r0 RANDOM-INIT (30k, ds=2) start ==="
rm -rf experiments/scene01_obj__n10_r0/train
if python scripts/run_experiment.py --exp scene01_obj__n10_r0 >> "$LOG" 2>&1; then
  log "DONE n10_r0 randinit"
else
  log "FAIL n10_r0 randinit"
fi
mv "$TF.bak_n10rand" "$TF"
log "[n10rand] restored ply_file_path"
log "=== n10_r0 randinit ALL DONE ==="
