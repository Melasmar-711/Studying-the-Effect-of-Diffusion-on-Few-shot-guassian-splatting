#!/usr/bin/env bash
# Run the whole experiment grid sequentially (one GPU job at a time).
# Order: quick few-shot baselines -> full upper bound -> augmented cells.
# Continues past failures; each run appends to results/registry.jsonl.
#
#   bash scripts/run_grid.sh              # full config iterations (30000)
#   bash scripts/run_grid.sh 7000         # override iterations (faster first pass)
#   bash scripts/run_grid.sh "" scene01   # explicit scene (default scene01)
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

ITERS="${1:-}"
SCENE="${2:-scene01}"
IT_ARG=""; [ -n "$ITERS" ] && IT_ARG="--iterations $ITERS"
LOG="results/run_grid.log"
mkdir -p results

mapfile -t ALL < <(tail -n +2 "configs/experiments/index.csv" | cut -d, -f1 | grep "^${SCENE}__")
BASE_FS=(); FULL=(); AUG=()
for e in "${ALL[@]}"; do
  case "$e" in
    *__full) FULL+=("$e");;
    *_r0)    BASE_FS+=("$e");;
    *)       AUG+=("$e");;
  esac
done
ORDER=("${BASE_FS[@]}" "${FULL[@]}" "${AUG[@]}")

echo "=== grid start $(date) : ${#ORDER[@]} runs, iters=${ITERS:-config} ===" | tee -a "$LOG"
i=0
for exp in "${ORDER[@]}"; do
  i=$((i+1))
  echo "[$i/${#ORDER[@]}] $(date +%H:%M:%S) START $exp" | tee -a "$LOG"
  # shellcheck disable=SC2086
  if python scripts/run_experiment.py --exp "$exp" $IT_ARG >> "$LOG" 2>&1; then
    echo "[$i/${#ORDER[@]}] $(date +%H:%M:%S) DONE  $exp" | tee -a "$LOG"
  else
    echo "[$i/${#ORDER[@]}] $(date +%H:%M:%S) FAIL  $exp" | tee -a "$LOG"
  fi
done
python scripts/compare.py >> "$LOG" 2>&1 || true
echo "=== grid complete $(date) ===" | tee -a "$LOG"
