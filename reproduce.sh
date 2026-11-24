#!/usr/bin/env bash
# Rerun the experiments on the included datasets and rewrite results/.
# Takes about 45 minutes on a laptop. Not rerun: the NS-3 simulations, the
# timing benchmark and the language-model fine-tune.
#
#   bash reproduce.sh
#   PY=/path/to/python3 bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/models"
PY="${PY:-python3}"

echo "== [1/9] check of the out-of-fold HRM fit =="
"$PY" test_hrm_oof.py

echo "== [2/9] Table II: bootstrap dataset, seeds 0-2 =="
"$PY" run_experiments.py

echo "== [3/9] ten splits, transfer, noise, pulsed attacker, attention =="
"$PY" extra_experiments.py all

echo "== [4/9] Table III: NS-3 main grid, seeds 0-2 =="
"$PY" ns3_experiments.py

echo "== [5/9] NS-3 results by attack intensity =="
"$PY" ns3_perdrop.py

echo "== [6/9] comparison with prior work (legacy-mirror scenarios) =="
"$PY" legacy_mirror_eval.py

echo "== [7/9] per-split metrics and split ids =="
"$PY" per_split_metrics.py
"$PY" per_split_metrics.py --ns3-ten-extra

echo "== [8/9] paired comparisons =="
"$PY" split_statistics.py

echo "== [9/9] figures =="
"$PY" make_figures.py

echo "done: see results/"
