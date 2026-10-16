#!/bin/bash
# Robustness replicate (same method, same seeds) for the tasks whose primary share fell within 0.05 of 0.75.
# Reported in ablations.md notes only; the stratum verdict uses the primary run.
S=/private/tmp/claude-501/-Users-jack-projects-pa/cb767837-0539-44e7-8d8b-4bf3c9abe4f2/scratchpad
O=/Users/jack/projects/swarm-scaling/analysis/v7-prep/ablations
cd "$S/abl_cwd" || exit 1
for t in ${*:-l0_pruning generalized_eigenvectors_complex l1_pruning}; do
  ( NUMBA_CACHE_DIR=$(mktemp -d "$S/abl_cwd/nc.XXXXXX") timeout 1800 "$S/venv/bin/python" -I "$O/ablate_m.py" "$S" "$t" 25 1200 "$O/replicate" > "$O/replicate/$t.log" 2>&1
    echo "$t exit $?" >> "$O/replicate/_progress.txt" ) &
done
wait
echo ALLDONE >> "$O/replicate/_progress.txt"
