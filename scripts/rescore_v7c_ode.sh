#!/bin/bash
# 2026-10-18 (Jack): re-select the v7 calibration's ode group (Opus team vs 4 diversified independents) on the fixed
# harness - the concrete-solution guard drops the team's array.array candidates, the validity pass backfills the
# shortlist, and hidden scoring runs exclusively in one scorer session - then take the scorer down.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
bash scripts/cloud/scorer.sh up --ref main --machine c7a.8xlarge --hours 4 --experiment v7 -- --slots 3 --memory 16g || exit 1
TOK=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.cache/swarm-scaling/scorer.json")))["token"])')
URL=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.cache/swarm-scaling/scorer.json")))["url"])')
SCORER_TOKEN=$TOK uv run python -u scripts/select_portfolio.py logs/v7c-ode_lorenz96_nonchaotic-r1-divteam \
  logs/v7c-ode_lorenz96_nonchaotic-r1-div-i0 logs/v7c-ode_lorenz96_nonchaotic-r1-div-i1 \
  logs/v7c-ode_lorenz96_nonchaotic-r1-div-i2 logs/v7c-ode_lorenz96_nonchaotic-r1-div-i3 \
  --scorer "$URL" --concurrency 3 --validity-n 1000 --exclusive-hidden --fresh
rc=$?
bash scripts/cloud/scorer.sh down
exit $rc
