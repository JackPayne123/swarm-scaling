#!/bin/bash
# Runs ablate_m.py for the 11 provisional-M tasks, at most 4 at a time, 25 instances each,
# instance loop capped at 1200 s per task (the script stops after the instance in progress).
# The randomized_svd denominator (pub_randomized_svd_patched.json) must exist first; it came from:
#   runpub.py randomized_svd_patched randomized_svd pub_randomized_svd_patched.json 400 40 1200 <top 3 solvers in pub_plan.tsv>
S=/private/tmp/claude-501/-Users-jack-projects-pa/cb767837-0539-44e7-8d8b-4bf3c9abe4f2/scratchpad
O=/Users/jack/projects/swarm-scaling/analysis/v7-prep/ablations
mkdir -p "$O/logs" "$S/abl_cwd"
cd "$S/abl_cwd" || exit 1
export S O
# Fresh numba cache dir per task run: a cache=True published solver loaded through
# spec_from_file_location fails to reload its own cache ("No module named '<dynamic>'").
TASKS="${*:-chebyshev_center generalized_eigenvectors_complex ode_lorenz96_nonchaotic graph_global_efficiency pagerank minimum_spanning_tree shortest_path_dijkstra l0_pruning l1_pruning pde_heat1d randomized_svd}"
printf '%s\n' $TASKS |
  xargs -P 4 -I{} sh -c 'NUMBA_CACHE_DIR=$(mktemp -d "$S/abl_cwd/nc.XXXXXX") timeout 1800 "$S/venv/bin/python" -I "$O/ablate_m.py" "$S" {} 25 1200 "$O" > "$O/logs/{}.log" 2>&1; echo "{} exit $?" >> "$O/logs/_progress.txt"'
echo ALLDONE >> "$O/logs/_progress.txt"
