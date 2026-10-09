#!/bin/bash
# Pilot 1 (2026-10-08): communicating teams vs the same number of independent agents.
# Claude Opus 5.5, effort high, 2M total-token cap per agent, free stopping, 3 non-CP-SAT pilot tasks.
# Runs one eval at a time (clean timing) and stops if measured spend passes SPEND_CAP.
# Usage: bgjob run --name pilot1 -- bash scripts/pilot1.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

MODEL=anthropic/claude-opus-5-5
BUDGET=2000000
SPEND_CAP=150
TASKS=(algotune/cvar-projection algotune/dst-type-ii-scipy-fftpack algotune/generalized-eigenvalues-real)
COMMON=(--models "$MODEL" --reasoning-effort high --budget "$BUDGET" --budget-type all --time-limit 7200)

spent() {
  uv run python scripts/run_cost.py logs/pilot1-* 2>/dev/null | awk '/^TOTAL/ {gsub(/\$/, "", $2); print $2}'
}

check_cap() {
  local s
  s=$(spent)
  echo "[pilot1] measured spend so far: \$${s:-0}"
  if awk -v s="${s:-0}" -v c="$SPEND_CAP" 'BEGIN {exit !(s > c)}'; then
    echo "[pilot1] spend cap \$${SPEND_CAP} passed; stopping"
    exit 3
  fi
}

run() {
  echo "[pilot1] $(date +%H:%M:%S) start: $*"
  /usr/bin/caffeinate -is uv run python -u -m swarm_scaling.runner "$@" || echo "[pilot1] run failed: $*"
  check_cap
}

for task in "${TASKS[@]}"; do
  slug=${task#algotune/}
  run --arm team --n 2 "${COMMON[@]}" --sample "$task" --name "pilot1-team2-$slug"
  run --arm team --n 4 "${COMMON[@]}" --sample "$task" --name "pilot1-team4-$slug"
done
for task in "${TASKS[@]}"; do
  slug=${task#algotune/}
  run --arm independent "${COMMON[@]}" --sample "$task" --epochs 8 --name "pilot1-indep-$slug"
done
echo "[pilot1] $(date +%H:%M:%S) done"
check_cap
