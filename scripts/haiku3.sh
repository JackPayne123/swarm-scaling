#!/bin/bash
# Haiku 3 (2026-10-11): plans/haiku3-dst.tsv (teams of 16 and 8 on dst) on one scorer, rescore 3 times, scorer down.
# Scorer is smaller than 1 slot per 4 agents (7 slots) so two 96-vCPU team-of-16 VMs fit the 300 vCPU quota with it.
# Usage: bgjob run --name haiku3 --grace 60 -- bash scripts/haiku3.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
bash scripts/cloud/scorer.sh up --ref main --machine c7a.16xlarge --hours 12 -- --slots 7 || exit 1
TOK=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.cache/swarm-scaling/scorer.json")))["token"])')
URL=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.cache/swarm-scaling/scorer.json")))["url"])')
uv run python -u scripts/cloud/fleet.py plan plans/haiku3-dst.tsv --ref main
SCORER_TOKEN=$TOK uv run python -u scripts/rescore.py "logs/haiku3-dst-*" --repeats 3 --scorer "$URL"
uv run python scripts/run_cost.py logs/haiku3-* 2>/dev/null | grep TOTAL
bash scripts/cloud/scorer.sh down
echo "[haiku3] $(date +%H:%M:%S) done"
