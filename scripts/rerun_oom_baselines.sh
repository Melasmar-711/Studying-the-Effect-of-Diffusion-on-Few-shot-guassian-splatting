#!/usr/bin/env bash
# Re-run the two runs that CUDA-OOM'd in the overnight grid (n20_r0, full).
# expandable_segments alone was NOT enough (it reclaimed the fragmentation but a
# true capacity wall remained: ~6.5 GiB of gaussians + ~750 MiB more needed).
# So we ALSO apply a densification cap via configs/project_capped.yaml
# (densify-grad-thresh 0.0015, stop-split-at 9000) to bound the gaussian count.
# The 14 augmented runs self-limited far below 8 GB, so the cap only bites the
# diverging baselines — documented in VERDICT.md.
set -uo pipefail
cd /home/asmar/GS_VR
# shellcheck disable=SC1091
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=results/rerun_oom.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== re-run OOM baselines (expandable_segments) start ==="

EXPS=(scene01_obj__n20_r0 scene01_obj__full)
i=0
for exp in "${EXPS[@]}"; do
  i=$((i+1))
  rm -rf "experiments/$exp/train"      # avoid nerfstudio 'timestamp exists' errors
  log "[$i/${#EXPS[@]}] START $exp"
  if python scripts/run_experiment.py --exp "$exp" --config configs/project_capped.yaml >> "$LOG" 2>&1; then
    log "[$i/${#EXPS[@]}] DONE  $exp"
  else
    log "[$i/${#EXPS[@]}] FAIL  $exp"
  fi
  [ "$i" -lt "${#EXPS[@]}" ] && { log "cooldown 300s"; sleep 300; }
done

python scripts/compare.py >> "$LOG" 2>&1 || true
log "=== re-run OOM baselines ALL DONE ==="
