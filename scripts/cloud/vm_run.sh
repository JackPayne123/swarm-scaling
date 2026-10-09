#!/bin/bash
# The VM side of run_sample.sh (runs there as root).
#   vm_run.sh start <name> <git-ref> <script|-> <args...>
#     Installs the uploaded seed offset and env file (mode 600) and starts `vm_run.sh run` detached, so the run
#     survives SSH disconnects. <script> "-" runs `python -m swarm_scaling.runner <args> --name <name>`, anything
#     else runs `python <script> <args>` with INSPECT_LOG_DIR=logs/<name>.
#   vm_run.sh run: checks out <git-ref>, syncs, runs, removes the secrets, writes the exit code to
#     $STATE/exit_code. Output goes to $STATE/run.log.
set -uo pipefail
STATE=/var/lib/swarm-run
REPO=/opt/swarm-scaling
UPLOAD=${SUDO_USER:+/home/$SUDO_USER}/swarm-upload

if [ "$1" = start ]; then
  mkdir -p "$STATE" && chmod 700 "$STATE"
  if [ -e "$UPLOAD/seed_offset" ]; then  # absent for --checker remote runs: only the scorer holds it
    install -m 600 "$UPLOAD/seed_offset" "$REPO/data/.algotune_seed_offset" || exit 1
  fi
  install -m 600 "$UPLOAD/env" "$STATE/env" || exit 1
  cp "$0" "$STATE/vm_run.sh"
  rm -rf "$UPLOAD"
  shift
  printf '%s\0' "$@" > "$STATE/argv"
  setsid nohup bash "$STATE/vm_run.sh" run > "$STATE/run.log" 2>&1 < /dev/null &
  echo "started pid $!"
  exit 0
fi

mapfile -d '' argv < "$STATE/argv"
name=${argv[0]} ref=${argv[1]} script=${argv[2]}
args=("${argv[@]:3}")
cd "$REPO"
set -a
source "$STATE/env"
set +a
rm -f "$STATE/env"

run() {
  git fetch --quiet origin || return 1
  commit=$(git rev-parse --verify -q "origin/$ref^{commit}" || git rev-parse --verify "$ref^{commit}") || return 1
  git checkout --quiet --detach "$commit" || return 1
  echo "commit=$commit"
  uv sync --frozen --quiet || return 1
  local i
  for i in $(seq 1 60); do  # the runner waits for the image too; a smoke script does not
    [ -n "$(docker image ls -q --filter 'reference=hb__*')" ] && break
    sleep 5
  done
  mkdir -p "logs/$name"
  if [ "$script" = - ]; then
    uv run python -u -m swarm_scaling.runner "${args[@]}" --name "$name"
  else
    INSPECT_LOG_DIR="logs/$name" uv run python -u "$script" "${args[@]}"
  fi
}

echo "[vm_run] $(date -u +%FT%TZ) start $name"
run
rc=$?
rm -f data/.algotune_seed_offset
rm -rf data/.algotune_verifier
echo "[vm_run] $(date -u +%FT%TZ) exit $rc"
echo "$rc" > "$STATE/exit_code.tmp" && mv "$STATE/exit_code.tmp" "$STATE/exit_code"
