# Shared settings and helpers for the GCP runner scripts. Sourced, not run.
# Every gcloud call goes through gc(), which pins the account and project: the default gcloud account belongs to
# a different org and must never be used here.

GCP_ACCOUNT=jacktpayne51@gmail.com
GCP_PROJECT=swarm-scaling-jp
# Zones tried in order when a create fails (T2D stockouts are per zone).
GCP_ZONES=${GCP_ZONES:-"us-central1-a us-central1-b us-central1-c us-central1-f"}
IMAGE_FAMILY=swarm-scaling-runner
TOOL_LABEL=swarm-gcp-runner  # label tool=<this> on every VM and image the scripts create; cleanup keys on it
REPO_URL=https://github.com/JackPayne123/swarm-scaling
REMOTE_REPO=/opt/swarm-scaling
REMOTE_STATE=/var/lib/swarm-run
DISK_GB=60

gc() { gcloud --account="$GCP_ACCOUNT" --project="$GCP_PROJECT" --quiet "$@"; }

# Own key only: ~/.ssh/config routes every host through the 1Password agent, which would prompt per connection.
SSH_OPTS=(-oIdentitiesOnly=yes -oIdentityAgent=none -oConnectTimeout=20 -oServerAliveInterval=30 -oLogLevel=ERROR)

vm_ssh() {  # vm_ssh <vm> <zone> <command>
  gc compute ssh "$1" --zone "$2" --strict-host-key-checking=no "${SSH_OPTS[@]/#/--ssh-flag=}" --command "$3"
}

vm_scp() {  # vm_scp <zone> <src...> <dest>; remote paths are <vm>:<path>
  local zone=$1
  shift
  gc compute scp --zone "$zone" --strict-host-key-checking=no "${SSH_OPTS[@]/#/--scp-flag=}" "$@"
}

wait_ssh() {  # wait_ssh <vm> <zone>: up to ~5 min for sshd and the guest agent's key install
  local i
  for i in $(seq 1 30); do
    vm_ssh "$1" "$2" true 2>/dev/null && return 0
    sleep 10
  done
  echo "ssh to $1 did not come up" >&2
  return 1
}

# create_vm <vm> <machine-type> <max-run-hours> <labels> <image flags...>
# Tries each zone in GCP_ZONES; when every zone is out of that machine type, waits CREATE_WAIT_S (default 120) and
# tries again, up to CREATE_ROUNDS (default 5) rounds. Sets ZONE to the zone the VM exists in, also when gcloud
# reported a failure but the VM was created anyway (so the caller can still delete it). The VM deletes itself
# after <max-run-hours> (with its boot disk) even if nothing else does.
create_vm() {
  local vm=$1 mt=$2 hours=$3 labels=$4 zone out round
  shift 4
  ZONE=
  for round in $(seq 1 "${CREATE_ROUNDS:-5}"); do
    for zone in $GCP_ZONES; do
      if out=$(gc compute instances create "$vm" --zone "$zone" --machine-type "$mt" "$@" \
        --boot-disk-type pd-balanced --boot-disk-size "${DISK_GB}GB" \
        --labels "tool=$TOOL_LABEL,$labels" --no-service-account --no-scopes \
        --max-run-duration "${hours}h" --instance-termination-action DELETE 2>&1); then
        ZONE=$zone
        return 0
      fi
      if gc compute instances describe "$vm" --zone "$zone" --format="value(name)" >/dev/null 2>&1; then
        echo "create $vm in $zone failed but the VM exists: $out" >&2
        ZONE=$zone
        return 1
      fi
      case $out in
        *RESOURCE_POOL_EXHAUSTED* | *ZONE_RESOURCE* | *"does not have enough resources"*)
          echo "$zone has no $mt available" >&2 ;;
        *) echo "create $vm in $zone failed: $out" >&2; return 1 ;;
      esac
    done
    [ "$round" -lt "${CREATE_ROUNDS:-5}" ] && sleep "${CREATE_WAIT_S:-120}"
  done
  return 1
}

delete_vm() {  # delete_vm <vm> <zone>; succeeds once the VM is gone (also if it already was)
  local out
  out=$(gc compute instances delete "$1" --zone "$2" --delete-disks=all 2>&1)
  if gc compute instances describe "$1" --zone "$2" --format="value(name)" >/dev/null 2>&1; then
    echo "delete $1 failed: $out" >&2
    return 1
  fi
}

vm_name() {  # GCE name for a run: lowercase [a-z0-9-], starts with a letter, at most 63 chars
  local n
  n=$(printf 'ss-%s' "$1" | tr 'A-Z_./' 'a-z---' | tr -cd 'a-z0-9-')
  if [ ${#n} -gt 63 ]; then
    n="${n:0:54}-$(printf '%s' "$1" | shasum | cut -c1-8)"
  fi
  printf '%s' "${n%-}"
}
