#!/usr/bin/env bash
# Runs one runner invocation on its own GCE VM and brings the logs back.
# Usage: run_sample.sh [--script <path>] <name> <machine-type> <git-ref> -- <runner args...>
#   Creates a VM from the newest swarm-scaling-runner image, uploads the seed offset and an env file holding only
#   the API keys the --models providers need (mode 600; removed on the VM once read), checks out <git-ref> (a
#   branch, tag or commit on GitHub; local-only commits cannot be fetched) and runs, detached on the VM,
#     python -m swarm_scaling.runner <runner args> --name <name>
#   (or, with --script, `python <path> <args>` with INSPECT_LOG_DIR=logs/<name>). When it ends, logs/<name>/ is
#   copied to the local logs/<name>/ (VM output in gcp-vm.log) and the VM is deleted, on failure too.
#   Progress is appended to logs/<name>/gcp-status as "<UTC time> <state> <detail>": created, running, copied,
#   deleted (delete-requested when stopped by a signal), exit. The exit status is the run's (or 1-2 when the infrastructure failed first).
# Env: MAX_RUN_HOURS (default 8; the VM deletes itself after this), POLL_S (default 60),
#   SEED_OFFSET_FILE (default data/.algotune_seed_offset), KEEP_VM=1 (debugging: do not delete), and
#   GCP_ZONES / CREATE_ROUNDS / CREATE_WAIT_S for stockouts (common.sh). Under bgjob, use --grace 30 or more.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/gcp/common.sh

script=-
if [ "${1:-}" = --script ]; then
  script=$2
  shift 2
fi
if [ $# -lt 4 ] || [ "$4" != -- ]; then
  sed -n '2,14p' "$0" >&2
  exit 2
fi
name=$1 mt=$2 ref=$3
shift 4
args=("$@")
max_hours=${MAX_RUN_HOURS:-8}
poll_s=${POLL_S:-60}
seed_file=${SEED_OFFSET_FILE:-data/.algotune_seed_offset}
vm=$(vm_name "$name")
logdir=logs/$name
status=$logdir/gcp-status

[ -r "$seed_file" ] || { echo "no seed offset at $seed_file (set SEED_OFFSET_FILE)" >&2; exit 2; }
if [ -e "$status" ]; then
  echo "$status exists: run name $name was used before" >&2
  exit 2
fi
mkdir -p "$logdir"
note() {
  echo "$(date -u +%FT%TZ) $*" >> "$status"
  echo "[$name] $*"
}

tmp=
ZONE= rc= copied=0 interrupted=0
copy_back() {
  [ -n "$ZONE" ] && [ "$copied" = 0 ] || return 0
  vm_ssh "$vm" "$ZONE" "sudo tar -C $REMOTE_REPO/logs -czf /tmp/run-logs.tgz '$name' && sudo cp $REMOTE_STATE/run.log /tmp/run.log && sudo chmod 644 /tmp/run-logs.tgz /tmp/run.log" &&
    vm_scp "$ZONE" "$vm:/tmp/run-logs.tgz" "$vm:/tmp/run.log" "$tmp/" &&
    tar -C logs -xzf "$tmp/run-logs.tgz" && cp "$tmp/run.log" "$logdir/gcp-vm.log" || { note "copy-failed"; return 1; }
  copied=1
  note copied "$(grep -m1 '^commit=' "$logdir/gcp-vm.log")"
}
finish() {
  rc=${rc:-$?}
  if [ -n "$ZONE" ] && [ "$interrupted" = 1 ]; then
    # Stopped (Ctrl-C, bgjob stop): no copy, and the delete runs in its own session, so the KILL that can follow
    # the TERM to this process group does not stop it (instances delete has no --async). Output: gcp-delete.log.
    python3 -c 'import subprocess, sys; subprocess.Popen(sys.argv[2:], start_new_session=True, stdin=subprocess.DEVNULL,
      stdout=open(sys.argv[1], "a"), stderr=subprocess.STDOUT)' "$logdir/gcp-delete.log" \
      gcloud --account="$GCP_ACCOUNT" --project="$GCP_PROJECT" --quiet compute instances delete "$vm" --zone "$ZONE" --delete-disks=all
    note delete-requested "vm=$vm (output in gcp-delete.log; check with scripts/gcp/gcp.py status)"
  elif [ -n "$ZONE" ]; then
    copy_back
    if [ "${KEEP_VM:-0}" = 1 ]; then
      note kept "vm=$vm zone=$ZONE"
    elif delete_vm "$vm" "$ZONE"; then
      note deleted "vm=$vm"
    else
      note delete-failed "vm=$vm zone=$ZONE (run scripts/gcp/gcp.py cleanup)"
    fi
  fi
  rm -rf "$tmp"
  if [ -e "$status" ]; then note exit "$rc"; else echo "[$name] no VM was created; exit $rc"; fi
}
trap finish EXIT
trap 'interrupted=1 rc=130; exit 130' INT TERM

# Keys: only the providers named in --models. Values come from scripts/env.sh and are never printed.
models=
for ((i = 0; i < ${#args[@]}; i++)); do
  case ${args[i]} in
    --models) models=${args[i + 1]:-} ;;
    --models=*) models=${args[i]#--models=} ;;
  esac
done
vars=()
for m in ${models//,/ }; do
  case $m in
    anthropic/*) vars+=(ANTHROPIC_API_KEY) ;;
    openai/*) vars+=(OPENAI_API_KEY) ;;
    google/*) vars+=(GEMINI_API_KEY GOOGLE_API_KEY) ;;
    openai-api/zai/*) vars+=(ZAI_API_KEY ZAI_BASE_URL) ;;
    openrouter/*) vars+=(OPENROUTER_API_KEY) ;;
    mockllm/*) ;;
    *) echo "no key mapping for model $m; add one to run_sample.sh" >&2; exit 2 ;;
  esac
done
tmp=$(mktemp -d)
chmod 700 "$tmp"
(
  source scripts/env.sh 2>/dev/null
  for v in ${vars[@]+"${vars[@]}"}; do
    [ -n "${!v:-}" ] || { echo "$v is empty after scripts/env.sh" >&2; exit 1; }
    printf 'export %s=%q\n' "$v" "${!v}"
  done
) > "$tmp/env" || exit 2
cp "$seed_file" "$tmp/seed_offset"

create_vm "$vm" "$mt" "$max_hours" "run=$vm" --image-family "$IMAGE_FAMILY" --image-project "$GCP_PROJECT"
created=$?
[ -n "$ZONE" ] && note created "vm=$vm zone=$ZONE machine=$mt max_hours=$max_hours"
[ "$created" = 0 ] || exit 1
wait_ssh "$vm" "$ZONE" || exit 1
vm_ssh "$vm" "$ZONE" "rm -rf swarm-upload && mkdir -m 700 swarm-upload" &&
  vm_scp "$ZONE" "$tmp/env" "$tmp/seed_offset" scripts/gcp/vm_run.sh "$vm:swarm-upload/" &&
  vm_ssh "$vm" "$ZONE" "sudo bash swarm-upload/vm_run.sh start $(printf '%q ' "$name" "$ref" "$script" "${args[@]}")" || exit 1
rm -f "$tmp/env" "$tmp/seed_offset"
note running "ref=$ref args=${args[*]}"

# Poll for the exit code. Tolerates SSH drops; gives up when the VM is gone (e.g. hit its max run duration).
deadline=$(( $(date +%s) + max_hours * 3600 + 600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep "$poll_s"
  if out=$(vm_ssh "$vm" "$ZONE" "sudo cat $REMOTE_STATE/exit_code 2>/dev/null || echo pending" 2>/dev/null); then
    out=${out//[$'\r\n ']/}
    [ "$out" = pending ] || { rc=$out; break; }
  elif gc compute instances describe "$vm" --zone "$ZONE" --format="value(name)" 2>&1 | grep -q "was not found"; then
    note vm-gone "the VM disappeared before the run finished"
    ZONE=
    exit 1
  fi
done
[ -n "$rc" ] || { note timeout "no exit code after ${max_hours} h"; exit 1; }
exit "$rc"
