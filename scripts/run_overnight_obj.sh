#!/usr/bin/env bash
# Unattended overnight runner for the object-centric inpaint ratio experiments.
# Waits for the baselines to finish, then runs the 12 augmented (inpaint) runs
# with a thermal cooldown every ~2 h of accumulated runtime. Continues past
# failures; builds the dashboard at the end.
#
# NOTE: deliberately does NOT run zero123-into-training (unverified orbit poses).
set -uo pipefail
cd /home/asmar/GS_VR
# shellcheck disable=SC1091
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6+PTX CUDA_HOME=/usr

LOG=results/run_overnight.log
BREAK_EVERY=7200      # cool down after this many seconds of accumulated running
COOLDOWN=1500         # 25-minute break to let the laptop/GPU recover
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "=== overnight runner start ==="

# 1) wait for the 30k baselines to complete
log "waiting for baselines..."
until grep -qa "baselines complete" results/run_baselines_obj.log 2>/dev/null; do sleep 60; done
log "baselines complete. cooldown ${COOLDOWN}s before augmented runs."
sleep "$COOLDOWN"

# 2) augmented inpaint experiments (scene01_obj, excluding baselines)
mapfile -t EXPS < <(tail -n +2 configs/experiments/index.csv | cut -d, -f1 \
                    | grep '^scene01_obj__' | grep -vE '_r0$|__full$')
log "running ${#EXPS[@]} augmented experiments: ${EXPS[*]}"

last_break=$(date +%s)
i=0
for exp in "${EXPS[@]}"; do
  i=$((i+1))
  log "[$i/${#EXPS[@]}] START $exp"
  if python scripts/run_experiment.py --exp "$exp" >> "$LOG" 2>&1; then
    log "[$i/${#EXPS[@]}] DONE  $exp"
  else
    log "[$i/${#EXPS[@]}] FAIL  $exp"
  fi
  now=$(date +%s)
  if [ $(( now - last_break )) -ge "$BREAK_EVERY" ] && [ "$i" -lt "${#EXPS[@]}" ]; then
    log "thermal cooldown ${COOLDOWN}s"
    sleep "$COOLDOWN"
    last_break=$(date +%s)
  fi
done

python scripts/compare.py >> "$LOG" 2>&1 || true
log "=== overnight runner ALL DONE ==="
