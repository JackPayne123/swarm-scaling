#!/usr/bin/env bash
# Publishes the task image once, then builds each cloud's VM image around that exact image.
# Usage: bgjob run --name image -- bash scripts/cloud/build_image.sh <publish|gcp|aws> [git-ref, default main]
#   publish  builds the hb__ task image from its Dockerfile on a GCP VM, pushes it to Artifact Registry and writes
#            its name and digest to scripts/cloud/task-image.env (commit that file). The Dockerfile pins nothing,
#            so this build is the only one: every VM runs the image by this digest.
#   gcp|aws  builds the VM image run_sample.sh and scorer.sh boot: a GCE disk image (family swarm-scaling-runner) or
#            an AWS AMI (tag tool=swarm-runner), holding Ubuntu 24.04 LTS x86_64, Docker, uv, the repo at <git-ref>
#            under /opt/swarm-scaling with `uv sync` done, the pinned AlgoTune dataset in the Harbor cache, and the
#            task image pulled by digest and tagged with its hb__ name. Skipped when that cloud already has an image
#            for the same commit and digest (FORCE=1 rebuilds).
# No seed offset or API key is ever on a build VM; the registry token is short-lived (1 h) and deleted after use.
# The build VM is deleted on exit, success or not, and ends itself after 3 h regardless.
#   BUILD_MACHINE_TYPE (default e2-standard-8 on GCP, <AWS_FAMILY>.2xlarge on AWS). To use the AWS image in another
#   region, copy it (aws ec2 copy-image --copy-image-tags) instead of rebuilding: a rebuild would pull the same digest.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/common.sh

what=${1:?usage: build_image.sh <publish|gcp|aws> [git-ref]}
ref=${2:-main}
commit=$(git ls-remote "$REPO_URL" "refs/heads/$ref" | cut -f1)
commit=${commit:-$ref}  # a commit id
tag=${commit:0:12}
cloud=$what
[ "$what" = publish ] && cloud=gcp
if [ "$what" != publish ]; then
  [ -n "${TASK_IMAGE_REF:-}" ] || { echo "no scripts/cloud/task-image.env: run build_image.sh publish first" >&2; exit 2; }
  dtag=${TASK_IMAGE_REF##*sha256:}
  dtag=${dtag:0:12}
  if [ "$cloud" = gcp ]; then
    existing=$(gc compute images list --filter="family=$IMAGE_FAMILY AND labels.commit=$tag AND labels.task-image=$dtag" --format="value(name)" 2>/dev/null)
  else
    existing=$(aw ec2 describe-images --owners self --filters "Name=tag:tool,Values=$TOOL_LABEL" "Name=tag:commit,Values=$tag" \
      "Name=tag:task-image,Values=$dtag" --query 'Images[].Name')
  fi
  if [ "${FORCE:-0}" != 1 ] && [ -n "$existing" ]; then
    echo "$cloud image $existing already built from $tag with task image $dtag; FORCE=1 to rebuild"
    exit 0
  fi
fi
[ "$cloud" = gcp ] && mt=${BUILD_MACHINE_TYPE:-e2-standard-8} || mt=${BUILD_MACHINE_TYPE:-$AWS_FAMILY.2xlarge}

VM_ID=
tmp=$(mktemp -d)
chmod 700 "$tmp"
cleanup() {
  rm -rf "$tmp"
  [ -n "$VM_ID" ] && vm_delete && echo "deleted $VM"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

registry_token() {  # a 1-hour Artifact Registry token, uploaded to the VM as a file (never on a command line)
  gcloud auth print-access-token --account="$GCP_ACCOUNT" > "$tmp/registry_token" &&
    vm_exec "rm -rf swarm-upload && mkdir -m 700 swarm-upload" && vm_put "$tmp/registry_token" swarm-upload/ &&
    rm -f "$tmp/registry_token"
}

mkdir -p logs
start=$(date +%s)
vm_create "$cloud" "ss-image-$what" "$mt" 3 image-build base || exit 1
echo "created $VM ($VM_ID) in $ZONE"
vm_wait_ssh || exit 1
vm_put scripts/cloud/vm_setup.sh . || exit 1

if [ "$what" = publish ]; then
  vm_exec "sudo bash vm_setup.sh '$REPO_URL' '$REMOTE_REPO' '$ref' build && rm vm_setup.sh" || { echo "setup failed"; exit 1; }
  name=$(vm_exec "sudo docker image ls --format '{{.Repository}}' --filter 'reference=hb__*'" | head -n 1)
  [ -n "$name" ] || { echo "no hb__ image on the VM"; exit 1; }
  registry_token || exit 1
  vm_exec "sudo bash -c 'set -e; docker login -u oauth2accesstoken --password-stdin https://${REGISTRY%%/*} < swarm-upload/registry_token;
    rm -rf swarm-upload; docker tag $name $REGISTRY/algotune:$name; docker push --quiet $REGISTRY/algotune:$name;
    docker logout https://${REGISTRY%%/*}'" || { echo "push failed"; exit 1; }
  digest=$(gc artifacts docker images describe "$REGISTRY/algotune:$name" --format='value(image_summary.digest)') || exit 1
  vm_exec "cat /opt/swarm-image-pip-freeze.txt" > "logs/image-pip-freeze-${digest#sha256:}.txt"
  cat > scripts/cloud/task-image.env <<EOF
# The one task image every VM runs (agent containers and scorer jobs), by digest. Written by build_image.sh publish
# on $(date -u +%F) (repo $ref); package list in logs/image-pip-freeze-${digest#sha256:}.txt.
TASK_IMAGE_NAME=$name
TASK_IMAGE_REF=$REGISTRY/algotune@$digest
EOF
  echo "published $REGISTRY/algotune@$digest as $name in $(( ($(date +%s) - start) / 60 )) min; commit scripts/cloud/task-image.env"
  exit 0
fi

registry_token || exit 1
vm_exec "sudo bash vm_setup.sh '$REPO_URL' '$REMOTE_REPO' '$ref' '$TASK_IMAGE_NAME' '$TASK_IMAGE_REF' && rm vm_setup.sh" || { echo "setup failed"; exit 1; }
image="swarm-runner-$(date -u +%Y%m%d-%H%M)-$tag"
if [ "$cloud" = gcp ]; then
  gc compute instances stop "$VM_ID" --zone "$ZONE" || exit 1
  gc compute images create "$image" --source-disk "$VM_ID" --source-disk-zone "$ZONE" --family "$IMAGE_FAMILY" \
    --labels "tool=$TOOL_LABEL,commit=$tag,task-image=$dtag" --description "swarm-scaling runner base, repo $ref ($commit), $TASK_IMAGE_REF" || exit 1
else
  aw ec2 stop-instances --instance-ids "$VM_ID" >/dev/null && aw ec2 wait instance-stopped --instance-ids "$VM_ID" || exit 1
  t="{Key=tool,Value=$TOOL_LABEL},{Key=commit,Value=$tag},{Key=task-image,Value=$dtag}"
  ami=$(aw ec2 create-image --instance-id "$VM_ID" --name "$image" --description "swarm-scaling runner base, repo $ref ($commit), $TASK_IMAGE_REF" \
    --tag-specifications "ResourceType=image,Tags=[$t]" "ResourceType=snapshot,Tags=[$t]" --query ImageId) || exit 1
  for i in $(seq 1 120); do  # the stock waiter gives up after 10 min; a 60 GB image can take longer
    state=$(aw ec2 describe-images --image-ids "$ami" --query 'Images[0].State')
    [ "$state" = available ] && break
    [ "$state" = failed ] && { echo "AMI $ami failed"; exit 1; }
    sleep 15
  done
  [ "$state" = available ] || { echo "AMI $ami still $state after 30 min"; exit 1; }
  image="$image ($ami)"
fi
echo "$cloud image $image created in $(( ($(date +%s) - start) / 60 )) min"
