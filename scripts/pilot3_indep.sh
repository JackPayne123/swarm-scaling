#!/bin/bash
# Pilot 3, independent arm only (2026-10-09): same model, budget and flags as scripts/pilot3.sh, but 4 samples
# (epochs) run at once. Timed checker runs (dev_eval, selection, scoring) stay one at a time across all of them.
# Stops if measured spend over logs/pilot3-* passes SPEND_CAP.
# Usage: bgjob run --name pilot3-indep -- bash scripts/pilot3_indep.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

MODEL=anthropic/claude-opus-5-5
BUDGET=2000000
SPEND_CAP=130
TASKS=(algotune/cvar-projection algotune/dst-type-ii-scipy-fftpack algotune/generalized-eigenvalues-real)
COMMON=(--models "$MODEL" --reasoning-effort high --budget "$BUDGET" --budget-type all --time-limit 7200 --tool-style claude_code)

spent() {
  uv run python scripts/run_cost.py logs/pilot3-* 2>/dev/null | awk '/^TOTAL/ {gsub(/\$/, "", $2); print $2}'
}

check_cap() {
  local s
  s=$(spent)
  echo "[pilot3-indep] measured spend so far: \$${s:-0}"
  if awk -v s="${s:-0}" -v c="$SPEND_CAP" 'BEGIN {exit !(s > c)}'; then
    echo "[pilot3-indep] spend cap \$${SPEND_CAP} passed; stopping"
    exit 3
  fi
}

run() {
  echo "[pilot3-indep] $(date +%H:%M:%S) start: $*"
  /usr/bin/caffeinate -is uv run python -u -m swarm_scaling.runner "$@" || echo "[pilot3-indep] run failed: $*"
  check_cap
}

check_cap
for task in "${TASKS[@]}"; do
  slug=${task#algotune/}
  run --arm independent "${COMMON[@]}" --sample "$task" --epochs 8 --parallel 4 --name "pilot3-indep-$slug"
done
echo "[pilot3-indep] $(date +%H:%M:%S) done"
