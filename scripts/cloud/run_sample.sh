#!/usr/bin/env bash
# Runs one runner invocation on its own VM (GCP or AWS, from the machine type) and brings the logs back.
# Usage: run_sample.sh [--script <path>] <name> <machine-type> <git-ref> -- <runner args...>
#   Creates a VM from the newest runner image (build_image.sh), uploads an env file holding only the API keys the
#   --models providers need (mode 600; removed on the VM once read) and the seed offset, checks out <git-ref> (a
#   branch, tag or commit on GitHub; local-only commits cannot be fetched) and runs, detached on the VM,
#     python -m swarm_scaling.runner <runner args> --name <name>
#   (or, with --script, `python <path> <args>` with INSPECT_LOG_DIR=logs/<name>). When it ends, logs/<name>/ is
#   copied to the local logs/<name>/ (VM output in vm.log) and the VM is deleted, on failure too.
#   With `--checker remote` in the runner args the VM gets no seed offset; its env carries SCORER_URL and
#   SCORER_TOKEN from the running scorer (scorer.sh up), and its IP is allowed on the scorer port for the run.
#   Such a VM times nothing, so when <machine-type> has no capacity or quota anywhere it may get another x86
#   family with the same vCPUs, in any US zone (CREATE_FALLBACK, common.sh); cloud-status records what it got.
#   Every VM must hold the task image pulled by TASK_IMAGE_REF (task-image.env), as the scorer does; a run
#   stops before starting otherwise.
#   Progress is appended to logs/<name>/cloud-status as "<UTC time> <state> <detail>": created (cloud, zone,
#   machine), host (CPU model, task image), running, copied, deleted (delete-requested on a signal), exit.
#   The exit status is the run's (or 1-2 when the infrastructure failed first).
# Env: MAX_RUN_HOURS (default 8; the VM ends itself after this), POLL_S (default 60), SEED_OFFSET_FILE (default
#   data/.algotune_seed_offset), KEEP_VM=1 (debugging: do not delete), and GCP_ZONES / CREATE_ROUNDS /
#   CREATE_WAIT_S for stockouts (common.sh). Under bgjob, use --grace 30 or more.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/common.sh

script=-
if [ "${1:-}" = --script ]; then
  script=$2
  shift 2
fi
if [ $# -lt 4 ] || [ "$4" != -- ]; then
  sed -n '2,21p' "$0" >&2
  exit 2
fi
name=$1 mt=$2 ref=$3
shift 4
args=("$@")
cloud=$(cloud_of "$mt")
max_hours=${MAX_RUN_HOURS:-8}
poll_s=${POLL_S:-60}
seed_file=${SEED_OFFSET_FILE:-data/.algotune_seed_offset}
logdir=logs/$name
status=$logdir/cloud-status

remote=0 models=
for ((i = 0; i < ${#args[@]}; i++)); do
  case ${args[i]} in
    --models) models=${args[i + 1]:-} ;;
    --models=*) models=${args[i]#--models=} ;;
    --checker) [ "${args[i + 1]:-}" = remote ] && remote=1 ;;
    --checker=remote) remote=1 ;;
  esac
done
[ -n "${TASK_IMAGE_REF:-}" ] || { echo "no scripts/cloud/task-image.env (build_image.sh publish)" >&2; exit 2; }
if [ "$remote" = 0 ]; then
  [ -r "$seed_file" ] || { echo "no seed offset at $seed_file (set SEED_OFFSET_FILE)" >&2; exit 2; }
else
  export CREATE_FALLBACK=1
  [ "$(scorer_field task_image 2>/dev/null)" = "$TASK_IMAGE_REF" ] ||
    { echo "the scorer runs task image '$(scorer_field task_image 2>/dev/null)', not $TASK_IMAGE_REF" >&2; exit 2; }
  scorer_url=$(scorer_field url) && scorer_token=$(scorer_field token) || { echo "--checker remote but no scorer is up (scorer.sh up)" >&2; exit 2; }
  curl -sf --max-time 20 -H "Authorization: Bearer $scorer_token" "$scorer_url/health" >/dev/null ||
    { echo "scorer at $scorer_url does not answer /health" >&2; exit 2; }
fi
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
VM_ID= rc= copied=0 interrupted=0 allowed=
copy_back() {
  [ -n "$VM_ID" ] && [ "$copied" = 0 ] || return 0
  vm_exec "sudo tar -C $REMOTE_REPO/logs -czf /tmp/run-logs.tgz '$name' && sudo cp $REMOTE_STATE/run.log /tmp/run.log && sudo chmod 644 /tmp/run-logs.tgz /tmp/run.log" &&
    vm_get /tmp/run-logs.tgz /tmp/run.log "$tmp/" &&
    tar -C logs -xzf "$tmp/run-logs.tgz" && cp "$tmp/run.log" "$logdir/vm.log" || { note "copy-failed"; return 1; }
  copied=1
  note copied "$(grep -m1 '^commit=' "$logdir/vm.log")"
}
finish() {
  rc=${rc:-$?}
  if [ -n "$VM_ID" ] && [ "$interrupted" = 1 ]; then
    # Stopped (Ctrl-C, bgjob stop): no copy, and a detached delete (see vm_delete_detached).
    vm_delete_detached "$logdir/delete.log"
    note delete-requested "vm=$VM_ID (output in delete.log; check with scripts/cloud/fleet.py status)"
  elif [ -n "$VM_ID" ]; then
    copy_back
    if [ "${KEEP_VM:-0}" = 1 ]; then
      note kept "vm=$VM_ID zone=$ZONE"
    elif vm_delete; then
      note deleted "vm=$VM_ID"
    else
      note delete-failed "vm=$VM_ID zone=$ZONE (run scripts/cloud/fleet.py cleanup)"
    fi
  fi
  [ -n "$allowed" ] && scorer_revoke "$allowed"
  rm -rf "$tmp"
  if [ -e "$status" ]; then note exit "$rc"; else echo "[$name] no VM was created; exit $rc"; fi
}
trap finish EXIT
trap 'interrupted=1 rc=130; exit 130' INT TERM

# Keys: only the providers named in --models. Values come from scripts/env.sh and are never printed.
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
  if [ "$remote" = 1 ]; then
    printf 'export SCORER_URL=%q\nexport SCORER_TOKEN=%q\n' "$scorer_url" "$scorer_token"
  fi
) > "$tmp/env" || exit 2
uploads=("$tmp/env")
if [ "$remote" = 0 ]; then
  cp "$seed_file" "$tmp/seed_offset"
  uploads+=("$tmp/seed_offset")
fi

vm_create "$cloud" "$(vm_name "$name")" "$mt" "$max_hours" agent runner
created=$?
[ -n "$VM_ID" ] && note created "vm=$VM_ID cloud=$cloud zone=$ZONE machine=$MACHINE requested=$mt max_hours=$max_hours checker=$([ "$remote" = 1 ] && echo remote || echo local)"
[ "$created" = 0 ] || exit 1
if [ "$remote" = 1 ]; then
  scorer_allow "$VM_IP" "$VM" || exit 1
  allowed=$VM_IP
fi
vm_wait_ssh || exit 1
vm_task_image_ok || { note image-mismatch "$TASK_IMAGE_NAME on the VM is not $TASK_IMAGE_REF"; exit 1; }
note host "cpu=$(vm_cpu_model) task_image=$TASK_IMAGE_REF"
vm_exec "rm -rf swarm-upload && mkdir -m 700 swarm-upload" &&
  vm_put "${uploads[@]}" scripts/cloud/vm_run.sh swarm-upload/ &&
  vm_exec "sudo bash swarm-upload/vm_run.sh start $(printf '%q ' "$name" "$ref" "$script" "${args[@]}")" || exit 1
rm -f "$tmp/env" "$tmp/seed_offset"
note running "ref=$ref args=${args[*]}"

# Poll for the exit code. Tolerates SSH drops; gives up when the VM is gone (e.g. hit its max run duration).
deadline=$(( $(date +%s) + max_hours * 3600 + 600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep "$poll_s"
  if out=$(vm_exec "sudo cat $REMOTE_STATE/exit_code 2>/dev/null || echo pending" 2>/dev/null); then
    out=${out//[$'\r\n ']/}
    [ "$out" = pending ] || { rc=$out; break; }
  elif [ "$(vm_state)" = gone ]; then
    note vm-gone "the VM disappeared before the run finished"
    VM_ID=
    exit 1
  fi
done
[ -n "$rc" ] || { note timeout "no exit code after ${max_hours} h"; exit 1; }
exit "$rc"
