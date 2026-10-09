#!/usr/bin/env bash
# Start the remote scorer service (src/swarm_scaling/scorer_service.py) from the repo root.
# Needs SCORER_TOKEN, the experiment's data/.algotune_seed_offset (or ALGOTUNE_SEED_OFFSET), Docker with the
# hb__* task images built, and the Harbor task cache. Extra arguments go to the service, e.g. --slots 2 --port 8770.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SCORER_TOKEN:?set SCORER_TOKEN}"
exec uv run python -m swarm_scaling.scorer_service "$@"
