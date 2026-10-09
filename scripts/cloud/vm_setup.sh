#!/bin/bash
# Provisions the base-image VM (runs there as root; build_image.sh copies and runs it). Each step is skipped
# when already done, so a rerun on the same VM resumes.
# Usage: vm_setup.sh <repo-url> <remote-repo> <git-ref>
set -euo pipefail
url=$1 repo=$2 ref=$3
export DEBIAN_FRONTEND=noninteractive

command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh
systemctl enable --now docker
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh

[ -d "$repo/.git" ] || git clone --quiet "$url" "$repo"
cd "$repo"
git fetch --quiet origin
commit=$(git rev-parse --verify -q "origin/$ref^{commit}" || git rev-parse --verify "$ref^{commit}")
git checkout --quiet --detach "$commit"
echo "commit=$commit"
uv sync --frozen --quiet

# Fetch the pinned AlgoTune dataset into the Harbor cache (as tasks.algotune_task does, without creating a seed
# offset) and print the one image every task uses with its build context.
built=$(uv run python -c '
import json
from inspect_harbor import algotune
from swarm_scaling.tasks import ALGOTUNE_REF, SPLIT_PATH, _harbor_name
split = json.loads(SPLIT_PATH.read_text())
task = algotune(ref=ALGOTUNE_REF, dataset_task_names=[_harbor_name(n) for k in ("pilot", "heldout") for n in split[k]])
builds = {(s.sandbox.config.services["default"].image, s.sandbox.config.services["default"].build.context) for s in task.dataset}
images = {i for i, _ in builds}
assert len(images) == 1, images
print(*sorted(builds)[0])
' | tail -n 1)
read -r image context <<< "$built"
echo "image=$image context=$context"
docker image inspect "$image" >/dev/null 2>&1 || docker build --quiet -t "$image" "$context"
docker builder prune -af >/dev/null
# The Dockerfile pins no versions, so record what this build installed.
docker run --rm --network none "$image" pip freeze > /opt/swarm-image-pip-freeze.txt
docker image ls "$image" --format 'built {{.Repository}} {{.Size}}'
apt-get clean
