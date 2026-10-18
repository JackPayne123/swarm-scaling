#!/bin/bash
# v6 (2026-10-11): run a v6 plan (plans/v6-pilot.tsv or plans/v6-main.tsv) on one scorer, then take the scorer down.
# scripts/v6_drive.py does the work: `size` reads the AWS quota now and sizes the agent scorer (at most 11 slots, at
# least one per 8 agents running at once for v6 and per 2 for v7, room left for agent VMs and the selection scorer);
# `run` runs the whole plan as the quota allows, selects each task-repeat group's portfolios
# (scripts/select_portfolio.py, --concurrency 6) as soon as its runs have ended, and starts no further row once
# measured + committed spend would pass SPEND_CAP dollars. v6 plans select on the agent scorer, one group at a time;
# plans with a v7 row select on a second scorer, the selection scorer, which v6_drive brings up when the first group
# has ended; it takes the agent scorer down when the last row has ended and the selection scorer after the last
# selection (v6_drive.py's docstring). On exit this script takes both down again, whatever happened.
# Scorer slots get SCORER_MEMORY (default 16g: the verifier holds 100 sinkhorn problems, 7.0 GiB); backstop HOURS
# (default 30). SCORER_MACHINE sets the agent scorer's type instead of the sizing rule (e.g. c7a.24xlarge);
# SELECTION_SCORER the selection scorer's (default c7a.16xlarge for v7 plans; none selects on the agent scorer).
# Usage: bgjob run --name v6 --grace 60 -- /usr/bin/caffeinate -is bash scripts/v6.sh plans/v6-main.tsv 16 [ref] [run options]
# Run options go to v6_drive.py run (v7: --group-start --validity-n 1000). EXPERIMENT (default v6) picks the scorers'
# secret offset (scorer.sh up --experiment).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
PLAN=${1:?usage: v6.sh <plan.tsv> <SPEND_CAP dollars> [ref]}
SPEND_CAP=${2:?usage: v6.sh <plan.tsv> <SPEND_CAP dollars> [ref]}
REF=${3:-v6}
shift $(( $# < 3 ? $# : 3 ))
SEL=(${SELECTION_SCORER:+--selection-scorer "$SELECTION_SCORER"})
# `bgjob stop` (TERM) or Ctrl+C must not leave scorers billing until their HOURS backstop
trap 'SCORER_NAME=agent bash scripts/cloud/scorer.sh down; SCORER_NAME=select bash scripts/cloud/scorer.sh down; exit 143' TERM INT

read -r SLOTS MACHINE MAX_AGENTS < <(uv run python scripts/v6_drive.py size "$PLAN" ${SCORER_MACHINE:+--machine "$SCORER_MACHINE"} ${SEL[@]+"${SEL[@]}"})
[ -n "${MAX_AGENTS:-}" ] || { echo "[v6] could not size the scorer for $PLAN" >&2; exit 1; }
echo "[v6] $(date +%H:%M:%S) $PLAN: agent scorer $MACHINE with $SLOTS slots, at most $MAX_AGENTS agents at once, spend cap \$$SPEND_CAP"
SCORER_NAME=agent bash scripts/cloud/scorer.sh up --ref "$REF" --machine "$MACHINE" --hours "${HOURS:-30}" --experiment "${EXPERIMENT:-v6}" -- --slots "$SLOTS" --memory "${SCORER_MEMORY:-16g}" || exit 1
TOK=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.cache/swarm-scaling/scorer.json")))["token"])')
URL=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.cache/swarm-scaling/scorer.json")))["url"])')
SCORER_TOKEN=$TOK uv run python -u scripts/v6_drive.py run "$PLAN" "$SPEND_CAP" --ref "$REF" --scorer "$URL" --max-agents "$MAX_AGENTS" \
  --scorer-hours "${HOURS:-30}" --scorer-memory "${SCORER_MEMORY:-16g}" ${SEL[@]+"${SEL[@]}"} "$@"
rc=$?
SCORER_NAME=agent bash scripts/cloud/scorer.sh down
SCORER_NAME=select bash scripts/cloud/scorer.sh down
echo "[v6] $(date +%H:%M:%S) done (plan exit $rc)"
exit $rc
