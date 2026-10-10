#!/usr/bin/env bash
# The remote scorer VM: one AWS instance running scripts/scorer_service.sh (branch remote-scorer), holding the
# seed offset; agent VMs reach it on SCORER_PORT with a bearer token (see scorer-api.md).
# Usage:
#   scorer.sh up --ref <git-ref> [--machine <AWS_FAMILY>.8xlarge: c7a in us-east-1, m7a in Sydney] [--hours 30] [-- <scorer_service.sh args, default --slots 2>]
#     Generates a fresh token, creates the VM from the runner AMI (checking it holds TASK_IMAGE_REF), uploads the seed offset and token (mode 600),
#     checks out <git-ref>, holds /dev/cpu_dma_latency at 0 (hold_cpu_dma_latency.py, log cpu_dma_latency.log),
#     starts the service detached, opens the port to this machine only, and waits for
#     /health. State (URL, token, instance) goes to $SCORER_STATE (mode 600); run_sample.sh reads it for
#     `--checker remote` runs and opens the port to each agent VM's IP for that run.
#     Slots: each takes 8 CPUs from --first-cpu (default 0) and the service needs 8 more, so e.g.
#     `--machine c7a.16xlarge -- --slots 4`. `fleet.py scorer up --plan <plan.tsv> --ref <ref>` picks both from a plan.
#   scorer.sh status      state, instance state and /health
#   scorer.sh down        copies the job log, queue samples (queue.jsonl), service and latency logs to logs/_scorer/<instance>/,
#                         terminates the VM (and any other instance tagged role=scorer), closes every rule on the scorer
#                         port, drops the state
# Env: SEED_OFFSET_FILE (default data/.algotune_seed_offset).
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/common.sh

health() {  # prints /health without the token
  curl -sf --max-time 20 -H "Authorization: Bearer $(scorer_field token)" "$(scorer_field url)/health"
}

load_state() {
  [ -r "$SCORER_STATE" ] || return 1
  CLOUD=aws VM=ss-scorer VM_ID=$(scorer_field instance_id) VM_IP=$(scorer_field ip) ZONE=$(scorer_field zone)
}

up() {
  local ref= mt=$AWS_FAMILY.8xlarge hours=30 service_args=(--slots 2) seed_file=${SEED_OFFSET_FILE:-data/.algotune_seed_offset}
  while [ $# -gt 0 ]; do
    case $1 in
      --ref) ref=$2; shift 2 ;;
      --machine) mt=$2; shift 2 ;;
      --hours) hours=$2; shift 2 ;;
      --) shift; service_args=("$@"); break ;;
      *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
  done
  [ -n "$ref" ] || { echo "--ref is required" >&2; exit 2; }
  [ -r "$seed_file" ] || { echo "no seed offset at $seed_file (set SEED_OFFSET_FILE)" >&2; exit 2; }
  if load_state && [ "$(vm_state)" != gone ]; then
    echo "a scorer is already up ($VM_ID); scorer.sh down first" >&2
    exit 2
  fi
  rm -f "$SCORER_STATE"  # stale: its VM is gone
  local ip token cpu
  tmp=$(mktemp -d)  # global: the EXIT trap removes it
  chmod 700 "$tmp"
  token=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
  printf 'export SCORER_TOKEN=%q\n' "$token" > "$tmp/env"
  cp "$seed_file" "$tmp/seed_offset"
  VM_ID=
  trap '[ -n "$VM_ID" ] && [ ! -e "$SCORER_STATE" ] && vm_delete && echo "deleted $VM_ID (scorer did not come up)"; rm -rf "$tmp"' EXIT
  trap 'exit 130' INT TERM

  vm_create aws ss-scorer "$mt" "$hours" scorer runner || exit 1
  echo "created $VM_ID ($mt) in $ZONE, $VM_IP"
  ip=$(my_ip) && scorer_allow "$ip" operator || exit 1
  vm_wait_ssh || exit 1
  vm_task_image_ok || { echo "$TASK_IMAGE_NAME on the scorer is not $TASK_IMAGE_REF" >&2; exit 1; }
  cpu=$(vm_cpu_model)
  vm_exec "rm -rf swarm-upload && mkdir -m 700 swarm-upload" && vm_put "$tmp/env" "$tmp/seed_offset" swarm-upload/ || exit 1
  # shellcheck disable=SC2016
  vm_exec "sudo bash -s -- $(printf '%q ' "$REMOTE_REPO" "$REMOTE_STATE" "$ref" "${service_args[@]}")" <<'EOF' || exit 1
set -euo pipefail
repo=$1 state=$2 ref=$3
shift 3
up=/home/$SUDO_USER/swarm-upload
mkdir -p "$state" && chmod 700 "$state"
cd "$repo"
git fetch --quiet origin
commit=$(git rev-parse --verify -q "origin/$ref^{commit}" || git rev-parse --verify "$ref^{commit}")
git checkout --quiet --detach "$commit"
echo "commit=$commit"
uv sync --frozen --quiet
# A volume made from an AMI fetches each block from S3 on first read (the AWS smoke's first dev_eval took 89 s
# against 12 s on GCP). Read the image, repo and caches once, so no timed job pays for that.
start=$(date +%s)
find /var/lib/containerd /var/lib/docker /opt /root/.cache /root/.local -type f -print0 2>/dev/null | xargs -0 -P 16 cat > /dev/null 2>&1 || true
echo "volume warm-up $(( $(date +%s) - start )) s"
install -m 600 "$up/seed_offset" data/.algotune_seed_offset
# Hold the CPU latency limit at 0 for the VM's life (no C2 exits with 800 us latency; scoring investigation).
setsid nohup python3 scripts/cloud/hold_cpu_dma_latency.py > "$state/cpu_dma_latency.log" 2>&1 < /dev/null &
sleep 1
cat "$state/cpu_dma_latency.log"
install -m 600 "$up/env" "$state/scorer.env"
rm -rf "$up"
# The service reads the token from its environment; the file is removed once the shell has loaded it.
setsid nohup bash -c 'set -a; source "$1"; set +a; rm -f "$1"; shift; exec bash scripts/scorer_service.sh "$@"' _ \
  "$state/scorer.env" "$@" > "$state/scorer.log" 2>&1 < /dev/null &
echo "service pid $!"
EOF
  (
    umask 077
    mkdir -p "$(dirname "$SCORER_STATE")"
    python3 -c 'import json, sys; json.dump(dict(zip(sys.argv[2::2], sys.argv[3::2])), open(sys.argv[1], "w"), indent=2)' "$SCORER_STATE" \
      cloud aws instance_id "$VM_ID" zone "$ZONE" ip "$VM_IP" url "http://$VM_IP:$SCORER_PORT" token "$token" \
      ref "$ref" machine "$mt" cpu "$cpu" task_image "$TASK_IMAGE_REF" started "$(date -u +%FT%TZ)"
  )
  local i
  for i in $(seq 1 60); do
    if out=$(health); then
      echo "scorer up at http://$VM_IP:$SCORER_PORT ($mt, $cpu, task image $TASK_IMAGE_REF): $out" | cut -c1-700
      return 0
    fi
    sleep 5
  done
  echo "scorer did not answer /health within 5 min; service log:" >&2
  vm_exec "sudo tail -n 30 $REMOTE_STATE/scorer.log" >&2
  down
  exit 1
}

status() {
  if ! load_state; then
    echo "no scorer state ($SCORER_STATE)"
  else
    echo "scorer $VM_ID ($(scorer_field machine), $(scorer_field cpu)) at $(scorer_field url), ref $(scorer_field ref), since $(scorer_field started): $(vm_state)"
    echo "task image $(scorer_field task_image)"
    health | cut -c1-600 || echo "/health did not answer"
  fi
  aw ec2 describe-instances --filters "Name=tag:tool,Values=$TOOL_LABEL" Name=tag:role,Values=scorer \
    Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]'
}

down() {
  local dir ids sg rules
  if load_state && [ "$(vm_state)" != gone ]; then
    dir=logs/_scorer/$VM_ID
    mkdir -p "$dir"
    vm_exec "sudo tar -C / -czf /tmp/scorer-logs.tgz --ignore-failed-read --transform 's,.*/,,' ${REMOTE_REPO#/}/data/.scorer/jobs.jsonl ${REMOTE_REPO#/}/data/.scorer/queue.jsonl ${REMOTE_STATE#/}/scorer.log ${REMOTE_STATE#/}/cpu_dma_latency.log; sudo chmod 644 /tmp/scorer-logs.tgz" &&
      vm_get /tmp/scorer-logs.tgz "$dir/" && tar -C "$dir" -xzf "$dir/scorer-logs.tgz" && rm "$dir/scorer-logs.tgz" &&
      echo "scorer logs in $dir" || echo "could not copy the scorer logs" >&2
  fi
  ids=$(aw ec2 describe-instances --filters "Name=tag:tool,Values=$TOOL_LABEL" Name=tag:role,Values=scorer \
    Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId')
  if [ -n "$ids" ]; then
    aw ec2 terminate-instances --instance-ids $ids >/dev/null && aw ec2 wait instance-terminated --instance-ids $ids &&
      echo "terminated $ids" || { echo "terminate $ids failed" >&2; return 1; }
  fi
  sg=$(aws_scorer_sg) || return 1
  rules=$(aw ec2 describe-security-group-rules --filters "Name=group-id,Values=$sg" --query 'SecurityGroupRules[?!IsEgress].SecurityGroupRuleId')
  [ -z "$rules" ] || aw ec2 revoke-security-group-ingress --group-id "$sg" --security-group-rule-ids $rules >/dev/null || return 1
  rm -f "$SCORER_STATE"
  echo "scorer down"
}

cmd=${1:-}
shift || true
case $cmd in
  up) up "$@" ;;
  status) status ;;
  down) down ;;
  *) sed -n '2,17p' "$0" >&2; exit 2 ;;
esac
