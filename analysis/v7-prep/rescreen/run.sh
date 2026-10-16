#!/bin/bash
# Rescreen of the v7-patched tasks (pca, randomized_svd, sparse_lowest_eigenvalues_posdef) with v7-tasks.md's scripts.
S=/private/tmp/claude-501/-Users-jack-projects-pa/cb767837-0539-44e7-8d8b-4bf3c9abe4f2/scratchpad
O=/Users/jack/projects/swarm-scaling/analysis/v7-prep/rescreen
PY="$S/venv/bin/python -I"
export NUMBA_CACHE_DIR=$O/numba_cache
dir() { ls -d $S/patched/hub/algotune/$1/*/ | head -1; }
pca=$(dir pca) rsvd=$(dir randomized-svd) spl=$(dir sparse-lowest-eigenvalues-posdef)
R=$S/ext/algotune/results
pcasolvers=()
for m in "GPT-5.4" "Gemini 3.1 Pro Preview" "Gemini 3 Pro Preview" "Claude Opus 4.5" "R1" "GPT-OSS-120b" "GPT-5" "o4-mini" "GPT-5-mini" "GPT-5 Pro (medium)" "Claude Opus 4.6" "GLM-4.5" "Claude Opus 4.1" "Gemini 2.5 Pro" "Qwen3 Coder" "GPT-5.2"; do
  pcasolvers+=("$R/$m/pca/solver.py")
done
splsolvers=("$R/Qwen3 Coder/sparse_lowest_eigenvalues_posdef/solver.py" "$R/Gemini 3.1 Pro Preview/sparse_lowest_eigenvalues_posdef/solver.py" "$R/GPT-OSS-120b/sparse_lowest_eigenvalues_posdef/solver.py")
(
  for t in pca:$pca randomized_svd:$rsvd sparse_lowest_eigenvalues_posdef:$spl; do
    timeout 1200 $PY $S/screen.py ${t#*:} ${t%%:*} $O/screen_${t%%:*}.json 100 1 900
  done
) > $O/screen.log 2>&1 &
(
  for t in pca:$pca randomized_svd:$rsvd sparse_lowest_eigenvalues_posdef:$spl; do
    timeout 900 $PY $S/gencheck.py ${t#*:} ${t%%:*} $O/gen_${t%%:*}.json 1000 600
  done
) > $O/gen.log 2>&1 &
(cd $O && timeout 1800 $PY $S/runpub.py $pca pca $O/pub_pca.json 400 40 1500 "${pcasolvers[@]}") > $O/pub_pca.log 2>&1 &
(
  timeout 1800 $PY $S/runpub.py $spl sparse_lowest_eigenvalues_posdef $O/pub_spl.json 400 40 1500 "${splsolvers[@]}"
  timeout 1200 $PY $O/pca_checks.py $pca $O/pca_checks.json
) > $O/pub_spl_pcachecks.log 2>&1 &
wait
echo ALLDONE
