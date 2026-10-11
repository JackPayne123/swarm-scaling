#!/bin/bash
# Haiku 2 (2026-10-11): run plans/haiku2-{dst,geneig,optadv}.tsv one task at a time on one scorer, rescore each
# task's selected solvers 3 times, stop if measured spend over logs/haiku2-* passes SPEND_CAP, take the scorer down.
# Usage: bgjob run --name haiku2 --grace 60 -- bash scripts/haiku2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
SPEND_CAP=28
spent() { uv run python scripts/run_cost.py logs/haiku2-* 2>/dev/null | awk '/^TOTAL/ {gsub(/\$/, "", $2); print $2}'; }
over() { awk -v s="${1:-0}" -v c="$SPEND_CAP" 'BEGIN {exit !(s > c)}'; }
uv run python scripts/cloud/fleet.py scorer up --plan plans/haiku2-dst.tsv --ref main || exit 1
TOK=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.cache/swarm-scaling/scorer.json")))["token"])')
URL=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.cache/swarm-scaling/scorer.json")))["url"])')
for t in dst geneig optadv; do
  echo "[haiku2] $(date +%H:%M:%S) task $t"
  uv run python -u scripts/cloud/fleet.py plan plans/haiku2-$t.tsv --ref main
  SCORER_TOKEN=$TOK uv run python -u scripts/rescore.py "logs/haiku2-$t-*" --repeats 3 --scorer "$URL"
  s=$(spent); echo "[haiku2] measured spend so far: \$${s:-0}"
  if over "$s"; then echo "[haiku2] spend cap passed; stopping"; break; fi
done
bash scripts/cloud/scorer.sh down
echo "[haiku2] $(date +%H:%M:%S) done"
