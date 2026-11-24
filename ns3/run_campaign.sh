#!/usr/bin/env bash
# run_campaign.sh: sweep the ns-3 gray-hole scenario across the experimental
# grid. Runs are independent and run in parallel with xargs (adjust JOBS).
# Usage:
#   NS3_DIR=~/ns-3-dev ./run_campaign.sh          # full grid
#   NS3_DIR=~/ns-3-dev JOBS=8 ./run_campaign.sh
set -euo pipefail

NS3_DIR="${NS3_DIR:?set NS3_DIR to your ns-3 directory}"
OUTDIR="${OUTDIR:-$(pwd)/ns3_runs}"
JOBS="${JOBS:-4}"
mkdir -p "$OUTDIR"

# rw2d is a bounded 2D random walk. nFlows is left at the binary's default,
# max(8, N/4), and the resolved value is recorded in each run's .meta.json.
MOBILITIES=("rwp" "gm" "rw2d")
DROPS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)
SIZES=(20 40 60 80 100)
SEEDS=(1 2 3 4 5)

# build once
( cd "$NS3_DIR" && ./ns3 build )

JOBFILE="$(mktemp)"
for mob in "${MOBILITIES[@]}"; do
  for drop in "${DROPS[@]}"; do
    for size in "${SIZES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        tag="${mob}_n${size}_d${drop//./}_s${seed}"
        echo "cd '$NS3_DIR' && ./ns3 run --no-build \"grayhole-manet \
--nNodes=$size --mobility=$mob --dropRatio=$drop --grayFrac=0.2 \
--seed=$seed --out=$OUTDIR/$tag.csv\" > $OUTDIR/$tag.log 2>&1"
      done
    done
  done
done > "$JOBFILE"

echo "launching $(wc -l < "$JOBFILE") runs on $JOBS parallel workers..."
cat "$JOBFILE" | xargs -P "$JOBS" -I{} bash -c '{}'
rm -f "$JOBFILE"
echo "campaign complete -> $OUTDIR"
echo "next: python3 validate_telemetry.py $OUTDIR   # verify counters first"
echo "then: python3 parse_ns3_to_dataset.py $OUTDIR ../data"
