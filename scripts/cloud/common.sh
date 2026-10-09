# Shared settings and helpers for the cloud runner scripts (GCP and AWS). Sourced, not run.
# Every gcloud call goes through gc(), which pins the account and project: the default gcloud account belongs to
# a different org and must never be used here. Every AWS call goes through aw(), which pins the profile and
# region (`command aws`: on this Mac `aws` is a shell function wrapper).
#
# One VM at a time per shell: vm_create sets CLOUD, VM, VM_ID, ZONE and VM_IP, and the other vm_* helpers act on
# that VM. A GCP VM is addressed by name and zone, an AWS one by instance id and public IP.

TOOL_LABEL=swarm-runner  # label/tag tool=<this> on every VM and image the scripts create; cleanup keys on it
REPO_URL=https://github.com/JackPayne123/swarm-scaling
REMOTE_REPO=/opt/swarm-scaling
REMOTE_STATE=/var/lib/swarm-run
DISK_GB=60
SSH_KEY=$HOME/.ssh/google_compute_engine  # no passphrase; gcloud's key, imported to AWS as key pair swarm-runner
SCORER_STATE=$HOME/.cache/swarm-scaling/scorer.json  # mode 600; holds the scorer's URL and token
SCORER_PORT=8770

GCP_ACCOUNT=jacktpayne51@gmail.com
GCP_PROJECT=swarm-scaling-jp
# Zones tried in order when a create fails (T2D stockouts are per zone).
GCP_ZONES=${GCP_ZONES:-"us-central1-a us-central1-b us-central1-c us-central1-f"}
IMAGE_FAMILY=swarm-scaling-runner
REGISTRY=us-central1-docker.pkg.dev/swarm-scaling-jp/swarm-runner  # Artifact Registry; build VMs pull with a 1 h token
# TASK_IMAGE_NAME (the hb__ name the harness uses) and TASK_IMAGE_REF (registry@sha256 digest): build_image.sh publish
[ -r "$(dirname "${BASH_SOURCE[0]}")/task-image.env" ] && source "$(dirname "${BASH_SOURCE[0]}")/task-image.env"

# An org SCP denies EC2 in every region tried except ap-southeast-2 (checked 2026-10-09). c7a is not offered
# there; m7a is the same EPYC Genoa generation with 1 vCPU per physical core.
AWS_PROFILE_NAME=personal
AWS_REGION=${AWS_REGION:-ap-southeast-2}
AWS_KEY_NAME=swarm-runner
AWS_SG_SSH=swarm-runner-ssh
AWS_SG_SCORER=swarm-scorer
UBUNTU_AMI_PARAM=/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id

gc() { gcloud --account="$GCP_ACCOUNT" --project="$GCP_PROJECT" --quiet "$@"; }
aw() { command aws --profile "$AWS_PROFILE_NAME" --region "$AWS_REGION" --output text "$@"; }

cloud_of() {  # cloud_of <machine-type>: AWS types have a dot (m7a.4xlarge), GCP types do not (t2d-standard-16)
  case $1 in *.*) echo aws ;; *) echo gcp ;; esac
}

my_ip() { curl -sf --max-time 10 https://checkip.amazonaws.com; }

# Own key only: ~/.ssh/config routes every host through the 1Password agent, which would prompt per connection.
SSH_OPTS=(-oIdentitiesOnly=yes -oIdentityAgent=none -oConnectTimeout=20 -oServerAliveInterval=30 -oLogLevel=ERROR)
# AWS public IPs are reused across instances, so host keys are not remembered.
AWS_SSH_OPTS=("${SSH_OPTS[@]}" -i "$SSH_KEY" -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null)

vm_exec() {  # vm_exec <command>
  if [ "$CLOUD" = gcp ]; then
    gc compute ssh "$VM_ID" --zone "$ZONE" --strict-host-key-checking=no "${SSH_OPTS[@]/#/--ssh-flag=}" --command "$1"
  else
    ssh "${AWS_SSH_OPTS[@]}" "ubuntu@$VM_IP" "$1"
  fi
}

vm_put() {  # vm_put <local file...> <remote dir/>: into the ssh user's home when relative
  local dest=${*: -1}
  if [ "$CLOUD" = gcp ]; then
    gc compute scp --zone "$ZONE" --strict-host-key-checking=no "${SSH_OPTS[@]/#/--scp-flag=}" "${@:1:$#-1}" "$VM_ID:$dest"
  else
    scp "${AWS_SSH_OPTS[@]}" "${@:1:$#-1}" "ubuntu@$VM_IP:$dest"
  fi
}

vm_get() {  # vm_get <remote file...> <local dir>
  local dest=${*: -1} f srcs=()
  for f in "${@:1:$#-1}"; do
    if [ "$CLOUD" = gcp ]; then srcs+=("$VM_ID:$f"); else srcs+=("ubuntu@$VM_IP:$f"); fi
  done
  if [ "$CLOUD" = gcp ]; then
    gc compute scp --zone "$ZONE" --strict-host-key-checking=no "${SSH_OPTS[@]/#/--scp-flag=}" "${srcs[@]}" "$dest"
  else
    scp "${AWS_SSH_OPTS[@]}" "${srcs[@]}" "$dest"
  fi
}

vm_wait_ssh() {  # up to ~5 min for sshd and (GCP) the guest agent's key install
  local i
  for i in $(seq 1 30); do
    vm_exec true 2>/dev/null && return 0
    sleep 10
  done
  echo "ssh to $VM did not come up" >&2
  return 1
}

vm_task_image_ok() {  # does the VM's hb__ image come from TASK_IMAGE_REF (pulled by that digest)?
  vm_exec "sudo docker image inspect '$TASK_IMAGE_NAME' --format '{{json .RepoDigests}}'" 2>/dev/null | grep -qF "\"$TASK_IMAGE_REF\""
}

vm_cpu_model() { vm_exec "lscpu | sed -n 's/^Model name: *//p'" 2>/dev/null | head -n 1; }

vm_state() {  # prints the VM's state (GCP status or EC2 state), or "gone"
  local out
  if [ "$CLOUD" = gcp ]; then
    out=$(gc compute instances describe "$VM_ID" --zone "$ZONE" --format="value(status)" 2>&1) ||
      { case $out in *"was not found"*) echo gone ;; *) echo unknown ;; esac; return; }
  else
    out=$(aw ec2 describe-instances --instance-ids "$VM_ID" --query 'Reservations[].Instances[].State.Name' 2>&1) ||
      { case $out in *InvalidInstanceID.NotFound*) echo gone ;; *) echo unknown ;; esac; return; }
    case $out in terminated | shutting-down) out=gone ;; esac
  fi
  echo "$out"
}

vm_delete() {  # deletes the VM and waits; succeeds once it is gone (also if it already was)
  local out
  if [ "$CLOUD" = gcp ]; then
    out=$(gc compute instances delete "$VM_ID" --zone "$ZONE" --delete-disks=all 2>&1)
  else
    out=$(aw ec2 terminate-instances --instance-ids "$VM_ID" 2>&1)
  fi
  [ "$(vm_state)" = gone ] || { echo "delete $VM failed: $out" >&2; return 1; }
}

vm_delete_detached() {  # vm_delete_detached <log file>: starts the delete in its own session and returns
  # Used from signal handlers: a KILL that follows the TERM to this process group does not stop the delete.
  local cmd
  if [ "$CLOUD" = gcp ]; then
    cmd=(gcloud --account="$GCP_ACCOUNT" --project="$GCP_PROJECT" --quiet compute instances delete "$VM_ID" --zone "$ZONE" --delete-disks=all)
  else
    cmd=(aws --profile "$AWS_PROFILE_NAME" --region "$AWS_REGION" ec2 terminate-instances --instance-ids "$VM_ID")
  fi
  python3 -c 'import subprocess, sys; subprocess.Popen(sys.argv[2:], start_new_session=True, stdin=subprocess.DEVNULL,
    stdout=open(sys.argv[1], "a"), stderr=subprocess.STDOUT)' "$1" "${cmd[@]}"
}

# vm_create <cloud> <name> <machine-type> <max-run-hours> <role> <runner|base>
# runner = the newest VM image this tooling built around TASK_IMAGE_REF; base = stock Ubuntu 24.04 LTS x86_64.
# Tries every zone (GCP_ZONES on GCP, every AZ offering the type on AWS). With CREATE_FALLBACK=1 (agent VMs of
# --checker remote runs, which time nothing) it then tries the other x86 families of the same vCPU count
# (fallback_types) and, on GCP, every US zone offering them, treating quota errors like stockouts. When nothing
# is free it waits CREATE_WAIT_S (default 120) and tries again, up to CREATE_ROUNDS (default 5) rounds. Sets
# VM_ID/ZONE/MACHINE when the VM exists, also when the create reported a failure (so the caller can still
# delete it). The VM ends itself after <max-run-hours>, with its disk.
vm_create() {
  CLOUD=$1 VM=$2 VM_ID= ZONE= VM_IP= MACHINE=
  local mt=$3 hours=$4 role=$5 image=$6 round t types
  types=("$mt")
  [ "${CREATE_FALLBACK:-0}" = 1 ] && types=($(fallback_types "$mt"))
  for round in $(seq 1 "${CREATE_ROUNDS:-5}"); do
    for t in "${types[@]}"; do
      if [ "$CLOUD" = gcp ]; then _gcp_create "$t" "$hours" "$role" "$image"; else _aws_create "$t" "$hours" "$role" "$image"; fi
      case $? in
        0) MACHINE=$t; return 0 ;;
        2) ;;  # no capacity (or, with fallback, no quota) for this type anywhere tried
        *) MACHINE=$t; return 1 ;;
      esac
    done
    [ "$round" -lt "${CREATE_ROUNDS:-5}" ] && sleep "${CREATE_WAIT_S:-120}"
  done
  return 1
}

fallback_types() {  # fallback_types <machine-type>: it, then other x86 families with the same vCPU count
  local mt=$1 f
  echo "$mt"
  if [ "$(cloud_of "$mt")" = gcp ]; then
    for f in t2d n2d c2d e2; do [ "$f-standard-${mt##*-}" = "$mt" ] || echo "$f-standard-${mt##*-}"; done
  else
    for f in m7a c7a m6a c6a m7i c7i; do [ "$f.${mt#*.}" = "$mt" ] || echo "$f.${mt#*.}"; done
  fi
}

_no_capacity() {  # does this create error mean "nothing free here" (with fallback: also "no quota here")?
  case $1 in
    *RESOURCE_POOL_EXHAUSTED* | *ZONE_RESOURCE* | *"does not have enough resources"* | *InsufficientInstanceCapacity*) return 0 ;;
    *QUOTA_EXCEEDED* | *"Quota '"* | *VcpuLimitExceeded* | *"is not supported in your requested Availability Zone"*)
      [ "${CREATE_FALLBACK:-0}" = 1 ] && return 0 ;;
  esac
  return 1
}

_gcp_create() {  # one pass over the zones for <mt>; returns 0 created, 1 failed, 2 nothing free
  local mt=$1 hours=$2 role=$3 image=$4 zone zones out flags dtag
  if [ "$image" = runner ]; then
    dtag=${TASK_IMAGE_REF##*sha256:}
    flags=(--image "$(gc compute images list --filter="family=$IMAGE_FAMILY AND labels.task-image=${dtag:0:12}" \
      --sort-by=~creationTimestamp --limit 1 --format="value(name)")" --image-project "$GCP_PROJECT")
    [ -n "${flags[1]}" ] || { echo "no GCE image built around $TASK_IMAGE_REF (build_image.sh gcp)" >&2; return 1; }
  else
    flags=(--image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud)
  fi
  zones=$GCP_ZONES
  if [ "${CREATE_FALLBACK:-0}" = 1 ]; then  # every US zone offering the type, us-central1 first
    zones=$(gc compute machine-types list --filter="name=$mt AND zone:us-" --format="value(zone)" | sort | awk '/^us-central1/ {print; next} {rest = rest $0 "\n"} END {printf "%s", rest}')
  fi
  local full=  # a region whose quota for this family is used up: its other zones are skipped
  for zone in $zones; do
    [ "${zone%-*}" = "$full" ] && continue
    if out=$(gc compute instances create "$VM" --zone "$zone" --machine-type "$mt" "${flags[@]}" \
      --boot-disk-type pd-balanced --boot-disk-size "${DISK_GB}GB" \
      --labels "tool=$TOOL_LABEL,run=$VM,role=$role" --no-service-account --no-scopes \
      --max-run-duration "${hours}h" --instance-termination-action DELETE 2>&1); then
      VM_ID=$VM ZONE=$zone
      VM_IP=$(gc compute instances describe "$VM" --zone "$zone" --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
      return 0
    fi
    if gc compute instances describe "$VM" --zone "$zone" --format="value(name)" >/dev/null 2>&1; then
      echo "create $VM in $zone failed but the VM exists: $out" >&2
      VM_ID=$VM ZONE=$zone
      return 1
    fi
    _no_capacity "$out" || { echo "create $VM ($mt) in $zone failed: $out" >&2; return 1; }
    case $out in
      *QUOTA_EXCEEDED* | *"Quota '"*) full=${zone%-*}; echo "${zone%-*}: no quota for $mt" >&2 ;;
      *) echo "$zone: no $mt available" >&2 ;;
    esac
  done
  return 2
}

aws_ensure_access() {  # key pair and SSH security group (port 22 from this machine only); prints the group id
  local vpc sg ip
  aw ec2 describe-key-pairs --key-names "$AWS_KEY_NAME" >/dev/null 2>&1 ||
    aw ec2 import-key-pair --key-name "$AWS_KEY_NAME" --public-key-material "fileb://$SSH_KEY.pub" \
      --tag-specifications "ResourceType=key-pair,Tags=[{Key=tool,Value=$TOOL_LABEL}]" >/dev/null || return 1
  vpc=$(aw ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId') || return 1
  sg=$(aw ec2 describe-security-groups --filters "Name=group-name,Values=$AWS_SG_SSH" "Name=vpc-id,Values=$vpc" --query 'SecurityGroups[0].GroupId')
  if [ -z "$sg" ] || [ "$sg" = None ]; then
    sg=$(aw ec2 create-security-group --group-name "$AWS_SG_SSH" --vpc-id "$vpc" --description "swarm-runner: SSH from the operator only" \
      --tag-specifications "ResourceType=security-group,Tags=[{Key=tool,Value=$TOOL_LABEL}]" --query GroupId) || return 1
  fi
  ip=$(my_ip) || { echo "could not read this machine's public IP" >&2; return 1; }
  aw ec2 describe-security-group-rules --filters "Name=group-id,Values=$sg" \
    --query "SecurityGroupRules[?FromPort==\`22\` && CidrIpv4=='$ip/32'].SecurityGroupRuleId" | grep -q . ||
    aw ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 --cidr "$ip/32" >/dev/null || return 1
  echo "$sg"
}

_aws_create() {  # one round over the default subnets of zones offering <mt>; returns 0 / 1 / 2 like _gcp_create
  local mt=$1 hours=$2 role=$3 image=$4 ami azs sgs subnet az out userdata tags init dtag
  sgs=$(aws_ensure_access) || return 1
  [ "$role" = scorer ] && sgs="$sgs $(aws_scorer_sg)"
  if [ "$image" = runner ]; then
    dtag=${TASK_IMAGE_REF##*sha256:}
    ami=$(aw ec2 describe-images --owners self --filters "Name=tag:tool,Values=$TOOL_LABEL" "Name=tag:task-image,Values=${dtag:0:12}" \
      Name=state,Values=available --query 'sort_by(Images, &CreationDate)[-1].ImageId')
  else
    ami=$(aw ssm get-parameter --name "$UBUNTU_AMI_PARAM" --query Parameter.Value)
  fi
  case $ami in ami-*) ;; *) echo "no $image AMI found ($ami)" >&2; return 1 ;; esac
  azs=$(aw ec2 describe-instance-type-offerings --location-type availability-zone --filters "Name=instance-type,Values=$mt" \
    --query 'InstanceTypeOfferings[].Location' | tr '\t' ',')
  if [ -z "$azs" ]; then
    echo "$mt is not offered in $AWS_REGION" >&2
    [ "${CREATE_FALLBACK:-0}" = 1 ] && return 2 || return 1
  fi
  # The backstop: shut down after <hours>; instance-initiated shutdown terminates the instance.
  userdata=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%d\n' $((hours * 60)) > "$userdata"
  # A volume made from an AMI fetches each block from S3 on first read: reading the 12 GB task image took 810 s on
  # the first scorer (2026-10-09). A provisioned initialization rate (100-300 MiB/s) loads it at a fixed rate.
  if [ "$image" = runner ]; then init=",VolumeInitializationRate=300"; else init=; fi
  tags="{Key=tool,Value=$TOOL_LABEL},{Key=run,Value=$VM},{Key=role,Value=$role},{Key=Name,Value=$VM}"
  while read -r subnet az; do
    if VM_ID=$(aw ec2 run-instances --image-id "$ami" --instance-type "$mt" --key-name "$AWS_KEY_NAME" \
      --subnet-id "$subnet" --security-group-ids $sgs --associate-public-ip-address \
      --instance-initiated-shutdown-behavior terminate --user-data "file://$userdata" \
      --metadata-options HttpTokens=required \
      --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3,DeleteOnTermination=true$init}" \
      --tag-specifications "ResourceType=instance,Tags=[$tags]" "ResourceType=volume,Tags=[$tags]" \
      --query 'Instances[0].InstanceId' 2>&1); then
      ZONE=$az
      rm -f "$userdata"
      aw ec2 wait instance-running --instance-ids "$VM_ID" || return 1
      VM_IP=$(aw ec2 describe-instances --instance-ids "$VM_ID" --query 'Reservations[0].Instances[0].PublicIpAddress')
      return 0
    fi
    out=$VM_ID VM_ID=
    _no_capacity "$out" || { echo "create $VM ($mt) in $az failed: $out" >&2; rm -f "$userdata"; return 1; }
    echo "$az: no $mt available" >&2
  done < <(aw ec2 describe-subnets --filters Name=default-for-az,Values=true "Name=availability-zone,Values=$azs" \
    --query 'Subnets[].[SubnetId,AvailabilityZone]')
  rm -f "$userdata"
  return 2
}

aws_scorer_sg() {  # the scorer's security group (port SCORER_PORT, rules added per caller); prints its id
  local vpc sg
  vpc=$(aw ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId') || return 1
  sg=$(aw ec2 describe-security-groups --filters "Name=group-name,Values=$AWS_SG_SCORER" "Name=vpc-id,Values=$vpc" --query 'SecurityGroups[0].GroupId')
  if [ -z "$sg" ] || [ "$sg" = None ]; then
    sg=$(aw ec2 create-security-group --group-name "$AWS_SG_SCORER" --vpc-id "$vpc" --description "swarm-runner: scorer port from agent VMs and the operator" \
      --tag-specifications "ResourceType=security-group,Tags=[{Key=tool,Value=$TOOL_LABEL}]" --query GroupId) || return 1
  fi
  echo "$sg"
}

scorer_field() {  # scorer_field <key>: a field of the running scorer's state file (fails when no scorer is up)
  [ -r "$SCORER_STATE" ] && python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$SCORER_STATE" "$1"
}

scorer_allow() {  # scorer_allow <ip> <label>: open the scorer port to <ip>
  local sg out
  sg=$(aws_scorer_sg) || return 1
  out=$(aw ec2 authorize-security-group-ingress --group-id "$sg" \
    --ip-permissions "IpProtocol=tcp,FromPort=$SCORER_PORT,ToPort=$SCORER_PORT,IpRanges=[{CidrIp=$1/32,Description=$2}]" 2>&1) ||
    case $out in *InvalidPermission.Duplicate*) ;; *) echo "scorer_allow $1: $out" >&2; return 1 ;; esac
}

scorer_revoke() {  # scorer_revoke <ip>
  local sg
  sg=$(aws_scorer_sg) || return 1
  aw ec2 revoke-security-group-ingress --group-id "$sg" --protocol tcp --port "$SCORER_PORT" --cidr "$1/32" >/dev/null
}

vm_name() {  # VM name for a run: lowercase [a-z0-9-], starts with a letter, at most 63 chars
  local n
  n=$(printf 'ss-%s' "$1" | tr 'A-Z_./' 'a-z---' | tr -cd 'a-z0-9-')
  if [ ${#n} -gt 63 ]; then
    n="${n:0:54}-$(printf '%s' "$1" | shasum | cut -c1-8)"
  fi
  printf '%s' "${n%-}"
}
