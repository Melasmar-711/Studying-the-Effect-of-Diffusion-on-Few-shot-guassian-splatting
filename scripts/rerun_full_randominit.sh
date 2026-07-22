#!/usr/bin/env bash
# TEST: the good full run (19.57 @ 02:06) predates sparse_pc_object.ply (05:02),
# so it used RANDOM init (assemble wasn't copying a ply yet). Hypothesis: ply init
# is what triggers the many-view densification runaway; random init stays bounded.
# Temporarily strip ply_file_path from the scene so splatfacto random-inits, run
# full 7k @ ds=2 (matches the good run), then restore.
set -uo pipefail
cd /home/asmar/GS_VR
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/rerun_full.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

TF=output_masked/transforms.json
cp "$TF" "$TF.bak_randtest"
python - <<'PY'
import json
p="output_masked/transforms.json"; m=json.load(open(p))
m.pop("ply_file_path", None)          # no seed cloud -> splatfacto random-inits
json.dump(m, open(p,"w"), indent=2)
print("[randtest] removed ply_file_path -> random init")
PY

log "=== full RANDOM-INIT test (no ply, DEFAULT densification, ds=2, 7k) start ==="
rm -rf experiments/scene01_obj__full/train
if python scripts/run_experiment.py --exp scene01_obj__full --iterations 7000 >> "$LOG" 2>&1; then
  log "DONE full randinit"
else
  log "FAIL full randinit"
fi
mv "$TF.bak_randtest" "$TF"
log "[randtest] restored ply_file_path"
log "=== full randinit ALL DONE ==="
