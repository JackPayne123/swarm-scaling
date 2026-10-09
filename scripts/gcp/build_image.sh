#!/usr/bin/env bash
# Builds the reusable GCE disk image (family swarm-scaling-runner) that run_sample.sh boots: Ubuntu 24.04 LTS
# x86_64, Docker, uv, the repo at <git-ref> under /opt/swarm-scaling with `uv sync` done, the pinned AlgoTune
# dataset in the Harbor cache and the hb__ task image built. No seed offset or key is ever on this VM.
# Skips the build when the family already has an image from the same commit (FORCE=1 rebuilds). The build VM is
# deleted on exit, success or not, and deletes itself after 3 h regardless.
# Usage: bgjob run --name gcp-image -- bash scripts/gcp/build_image.sh [git-ref, default main]
#   BUILD_MACHINE_TYPE (default e2-standard-8, so a rebuild does not use T2D quota)
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/gcp/common.sh

ref=${1:-main}
mt=${BUILD_MACHINE_TYPE:-e2-standard-8}
vm=ss-image-build
commit=$(git ls-remote "$REPO_URL" "refs/heads/$ref" | cut -f1)
commit=${commit:-$ref}  # a commit id
if [ "${FORCE:-0}" != 1 ] && existing=$(gc compute images list --filter="family=$IMAGE_FAMILY AND labels.commit=${commit:0:12}" --format="value(name)") && [ -n "$existing" ]; then
  echo "image $existing already built from ${commit:0:12}; FORCE=1 to rebuild"
  exit 0
fi

ZONE=
cleanup() {
  [ -n "$ZONE" ] && delete_vm "$vm" "$ZONE" && echo "deleted $vm"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

mkdir -p logs
start=$(date +%s)
create_vm "$vm" "$mt" 3 "role=image-build" --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud || exit 1
zone=$ZONE
echo "created $vm in $zone"
wait_ssh "$vm" "$zone" || exit 1
vm_scp "$zone" scripts/gcp/vm_setup.sh "$vm:" || exit 1
vm_ssh "$vm" "$zone" "sudo bash vm_setup.sh '$REPO_URL' '$REMOTE_REPO' '$ref' && rm vm_setup.sh" || { echo "setup failed"; exit 1; }
vm_ssh "$vm" "$zone" "cat /opt/swarm-image-pip-freeze.txt" > "logs/gcp-image-pip-freeze-${commit:0:12}.txt"

gc compute instances stop "$vm" --zone "$zone" || exit 1
image="swarm-runner-$(date -u +%Y%m%d-%H%M)-${commit:0:12}"
gc compute images create "$image" --source-disk "$vm" --source-disk-zone "$zone" --family "$IMAGE_FAMILY" \
  --labels "tool=$TOOL_LABEL,commit=${commit:0:12}" --description "swarm-scaling runner base, repo $ref ($commit)" || exit 1
echo "image $image created in $(( ($(date +%s) - start) / 60 )) min; pip freeze in logs/gcp-image-pip-freeze-${commit:0:12}.txt"
