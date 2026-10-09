#!/usr/bin/env bash
# Builds the reusable VM image that run_sample.sh boots: a GCE disk image (family swarm-scaling-runner) or an AWS
# AMI (tag tool=swarm-runner). Both hold Ubuntu 24.04 LTS x86_64, Docker, uv, the repo at <git-ref> under
# /opt/swarm-scaling with `uv sync` done, the pinned AlgoTune dataset in the Harbor cache and the hb__ task image.
# No seed offset or key is ever on the build VM.
# Skips the build when that cloud already has an image from the same commit (FORCE=1 rebuilds). The build VM is
# deleted on exit, success or not, and ends itself after 3 h regardless.
# Usage: bgjob run --name image-gcp -- bash scripts/cloud/build_image.sh <gcp|aws> [git-ref, default main]
#   BUILD_MACHINE_TYPE (default e2-standard-8 on GCP, m7a.2xlarge on AWS)
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/common.sh

cloud=${1:?usage: build_image.sh <gcp|aws> [git-ref]}
ref=${2:-main}
commit=$(git ls-remote "$REPO_URL" "refs/heads/$ref" | cut -f1)
commit=${commit:-$ref}  # a commit id
tag=${commit:0:12}
if [ "$cloud" = gcp ]; then
  mt=${BUILD_MACHINE_TYPE:-e2-standard-8}
  existing=$(gc compute images list --filter="family=$IMAGE_FAMILY AND labels.commit=$tag" --format="value(name)" 2>/dev/null)
else
  mt=${BUILD_MACHINE_TYPE:-m7a.2xlarge}
  existing=$(aw ec2 describe-images --owners self --filters "Name=tag:tool,Values=$TOOL_LABEL" "Name=tag:commit,Values=$tag" --query 'Images[].Name')
fi
if [ "${FORCE:-0}" != 1 ] && [ -n "$existing" ]; then
  echo "$cloud image $existing already built from $tag; FORCE=1 to rebuild"
  exit 0
fi

VM_ID=
cleanup() {
  [ -n "$VM_ID" ] && vm_delete && echo "deleted $VM"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

mkdir -p logs
start=$(date +%s)
vm_create "$cloud" ss-image-build "$mt" 3 image-build base || exit 1
echo "created $VM ($VM_ID) in $ZONE"
vm_wait_ssh || exit 1
vm_put scripts/cloud/vm_setup.sh . || exit 1
vm_exec "sudo bash vm_setup.sh '$REPO_URL' '$REMOTE_REPO' '$ref' && rm vm_setup.sh" || { echo "setup failed"; exit 1; }
vm_exec "cat /opt/swarm-image-pip-freeze.txt" > "logs/image-pip-freeze-$cloud-$tag.txt"

image="swarm-runner-$(date -u +%Y%m%d-%H%M)-$tag"
if [ "$cloud" = gcp ]; then
  gc compute instances stop "$VM_ID" --zone "$ZONE" || exit 1
  gc compute images create "$image" --source-disk "$VM_ID" --source-disk-zone "$ZONE" --family "$IMAGE_FAMILY" \
    --labels "tool=$TOOL_LABEL,commit=$tag" --description "swarm-scaling runner base, repo $ref ($commit)" || exit 1
else
  aw ec2 stop-instances --instance-ids "$VM_ID" >/dev/null && aw ec2 wait instance-stopped --instance-ids "$VM_ID" || exit 1
  ami=$(aw ec2 create-image --instance-id "$VM_ID" --name "$image" --description "swarm-scaling runner base, repo $ref ($commit)" \
    --tag-specifications "ResourceType=image,Tags=[{Key=tool,Value=$TOOL_LABEL},{Key=commit,Value=$tag}]" \
    "ResourceType=snapshot,Tags=[{Key=tool,Value=$TOOL_LABEL},{Key=commit,Value=$tag}]" --query ImageId) || exit 1
  for i in $(seq 1 120); do  # the stock waiter gives up after 10 min; a 60 GB image can take longer
    state=$(aw ec2 describe-images --image-ids "$ami" --query 'Images[0].State')
    [ "$state" = available ] && break
    [ "$state" = failed ] && { echo "AMI $ami failed"; exit 1; }
    sleep 15
  done
  [ "$state" = available ] || { echo "AMI $ami still $state after 30 min"; exit 1; }
  image="$image ($ami)"
fi
echo "$cloud image $image created in $(( ($(date +%s) - start) / 60 )) min; pip freeze in logs/image-pip-freeze-$cloud-$tag.txt"
