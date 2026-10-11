#!/bin/bash
# v6 (2026-10-11): run a v6 plan (plans/v6-pilot.tsv or plans/v6-main.tsv) one task at a time on one scorer, then the
# two-stage portfolio selection (scripts/select_portfolio.py) over the plan's logs, then take the scorer down.
# After each task's batch, measured spend over the plan's logs (scripts/run_cost.py) is checked; once it exceeds
# SPEND_CAP dollars no further batch starts, so the overshoot is at most one batch (24 agents, $2.40 at $0.10 each).
# Scorer: one slot per 4 agents of the largest batch, fewer if the scorer and that batch's agent VMs would not fit in
# the AWS vCPU quota together (QUOTA, default 300); backstop HOURS (default 30).
# Batches go to logs/_plans/<plan>/; rerunning refuses run names already used (fleet.py plan).
# Usage: bgjob run --name v6 --grace 60 -- /usr/bin/caffeinate -is bash scripts/v6.sh plans/v6-pilot.tsv <SPEND_CAP> [ref]
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
PLAN=${1:?usage: v6.sh <plan.tsv> <SPEND_CAP dollars> [ref]}
SPEND_CAP=${2:?usage: v6.sh <plan.tsv> <SPEND_CAP dollars> [ref]}
REF=${3:-v6}
BATCHES=logs/_plans/$(basename "$PLAN" .tsv)
mkdir -p "$BATCHES"

# one sub-plan per task (rows grouped by --sample, in plan order); prints "<slots> <machine>" for the scorer
read -r SLOTS MACHINE < <(uv run python - "$PLAN" "$BATCHES" "${QUOTA:-300}" <<'PY'
import math, sys
from pathlib import Path
sys.path.insert(0, "scripts/cloud")
import fleet

plan, out, quota = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
lines = [l for l in plan.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
batches = {}
for row, line in zip(fleet.parse_plan(plan.read_text()), lines):
    batches.setdefault(row.args[row.args.index("--sample") + 1], []).append((row, line))
agents = vcpus = 0
for k, (sample, rows) in enumerate(batches.items()):
    (out / f"{k:02d}-{sample.split('/')[-1]}.tsv").write_text("".join(line + "\n" for _, line in rows))
    agents = max(agents, sum(fleet.arg_value(r.args, "--n", 1) for r, _ in rows))
    vcpus = max(vcpus, sum(fleet.vcpus(fleet.machine_for(r, "aws")) for r, _ in rows))
slots = math.ceil(agents / fleet.AGENTS_PER_SLOT)
while slots > 1 and vcpus + slots * fleet.SLOT_VCPUS + fleet.SCORER_SERVICE_VCPUS > quota:
    slots -= 1
if vcpus + slots * fleet.SLOT_VCPUS + fleet.SCORER_SERVICE_VCPUS > quota:
    sys.exit(f"a batch needs {vcpus} vCPUs; with a 1-slot scorer that exceeds the quota of {quota}")
print(slots, fleet.scorer_machine(slots))
PY
)
[ -n "${SLOTS:-}" ] || { echo "[v6] could not split $PLAN into batches" >&2; exit 1; }
echo "[v6] $(date +%H:%M:%S) $PLAN: $(ls "$BATCHES" | wc -l | tr -d ' ') batches, scorer $MACHINE with $SLOTS slots, spend cap \$$SPEND_CAP"

LOGS=$(awk '!/^#/ && NF {print "logs/"$1}' "$PLAN")
spent() { uv run python scripts/run_cost.py $(for d in $LOGS; do [ -d "$d" ] && echo "$d"; done) 2>/dev/null | awk '/^TOTAL/ {gsub(/\$/, "", $2); print $2}'; }
over() { awk -v s="${1:-0}" -v c="$SPEND_CAP" 'BEGIN {exit !(s > c)}'; }

s=$(spent)
if over "$s"; then echo "[v6] measured spend over this plan's logs is already \$$s; nothing launched"; exit 1; fi
bash scripts/cloud/scorer.sh up --ref "$REF" --machine "$MACHINE" --hours "${HOURS:-30}" -- --slots "$SLOTS" || exit 1
TOK=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.cache/swarm-scaling/scorer.json")))["token"])')
URL=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.cache/swarm-scaling/scorer.json")))["url"])')
for b in "$BATCHES"/*.tsv; do
  echo "[v6] $(date +%H:%M:%S) batch $(basename "$b")"
  uv run python -u scripts/cloud/fleet.py plan "$b" --ref "$REF"
  s=$(spent); echo "[v6] measured spend so far: \$${s:-0} (cap \$$SPEND_CAP)"
  if over "$s"; then echo "[v6] spend cap passed; no further batches"; break; fi
done
SCORER_TOKEN=$TOK uv run python -u scripts/select_portfolio.py $(for d in $LOGS; do [ -d "$d" ] && echo "$d"; done) --scorer "$URL"
bash scripts/cloud/scorer.sh down
echo "[v6] $(date +%H:%M:%S) done"
