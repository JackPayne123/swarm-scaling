#!/bin/bash
# v6 (2026-10-11): run a v6 plan (plans/v6-pilot.tsv or plans/v6-main.tsv) on one scorer, then take the scorer down.
# scripts/v6_drive.py does the work: `size` reads the AWS quota now and sizes the scorer (at most 11 slots, at least one
# per 8 agents running at once, room left for agent VMs); `run` runs the whole plan as the quota allows, selects each
# task-repeat group's portfolios (scripts/select_portfolio.py, --concurrency 3) as soon as its runs have ended, and
# starts no further row once measured + committed spend would pass SPEND_CAP dollars.
# Scorer slots get SCORER_MEMORY (default 16g: the verifier holds 100 sinkhorn problems, 7.0 GiB); backstop HOURS
# (default 30).
# Usage: bgjob run --name v6 --grace 60 -- /usr/bin/caffeinate -is bash scripts/v6.sh plans/v6-main.tsv 17 [ref]
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
PLAN=${1:?usage: v6.sh <plan.tsv> <SPEND_CAP dollars> [ref]}
SPEND_CAP=${2:?usage: v6.sh <plan.tsv> <SPEND_CAP dollars> [ref]}
REF=${3:-v6}

read -r SLOTS MACHINE MAX_AGENTS < <(uv run python scripts/v6_drive.py size "$PLAN")
[ -n "${MAX_AGENTS:-}" ] || { echo "[v6] could not size the scorer for $PLAN" >&2; exit 1; }
echo "[v6] $(date +%H:%M:%S) $PLAN: scorer $MACHINE with $SLOTS slots, at most $MAX_AGENTS agents at once, spend cap \$$SPEND_CAP"
bash scripts/cloud/scorer.sh up --ref "$REF" --machine "$MACHINE" --hours "${HOURS:-30}" -- --slots "$SLOTS" --memory "${SCORER_MEMORY:-16g}" || exit 1
TOK=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.cache/swarm-scaling/scorer.json")))["token"])')
URL=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.cache/swarm-scaling/scorer.json")))["url"])')
SCORER_TOKEN=$TOK uv run python -u scripts/v6_drive.py run "$PLAN" "$SPEND_CAP" --ref "$REF" --scorer "$URL" --max-agents "$MAX_AGENTS"
rc=$?
bash scripts/cloud/scorer.sh down
echo "[v6] $(date +%H:%M:%S) done (plan exit $rc)"
exit $rc
