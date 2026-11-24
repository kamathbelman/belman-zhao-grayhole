#!/usr/bin/env bash
# run_full_campaign_macos.sh: the full ns-3 campaign on macOS in one command.
#   bash ns3/run_full_campaign_macos.sh
# Steps (each is skipped if already done, and finished runs are not rerun):
#   1. check Xcode command-line tools, install cmake and ninja via pip if absent
#   2. download the ns-3.40 source to ~/ns3-grayhole (about 45 MB)
#   3. configure and build the needed ns-3 modules (release profile)
#   4. copy grayhole-manet.cc into scratch/ and build it
#   5. run the campaign grid in parallel into ns3/ns3_runs_full/: 3 mobility
#      models (rwp, gm, rw2d) x 7 drop ratios (0.1-0.7) x 5 sizes (20-100) x
#      5 seeds = 525 runs of 300 s, plus a legacy mirror approximating the
#      WIDECOM-2020 setup (30 nodes, 2000 m field, 600 s, random waypoint,
#      same 250 m disk propagation, so not an exact PHY reproduction)
#   6. parse everything into data_ns3_full/ (bootstrap-compatible schema)
#   7. train and evaluate all four models, writing results/ns3_full_results.json
# Recommended: run bash ns3/smoke_test.sh first (5 short runs and validation).
# Disk: about 2 GB under ~/ns3-grayhole. Override the location with NS3_HOME.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
NS3_HOME="${NS3_HOME:-$HOME/ns3-grayhole}"
NS3_DIR="$NS3_HOME/ns3"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"
OUT="$REPO/ns3/ns3_runs_full"
mkdir -p "$OUT" "$NS3_HOME"

# Schema guard: a run is current only if its meta.json declares schema
# revision 3. Outputs from an earlier schema must not satisfy the resumable
# skip-check, so they are moved to $PRE and regenerated.
PRE="$OUT/old-schema"
QN=0
for f in "$OUT"/run_*.csv; do
  [ -e "$f" ] || continue
  case "$f" in *.adj.csv) continue;; esac
  if ! grep -q '"schemaRev": 3' "$f.meta.json" 2>/dev/null; then
    mkdir -p "$PRE"
    mv "$f" "$PRE/"
    [ -f "$f.adj.csv" ] && mv "$f.adj.csv" "$PRE/"
    [ -f "$f.meta.json" ] && mv "$f.meta.json" "$PRE/"
    QN=$((QN+1))
  fi
done
[ "$QN" -gt 0 ] && echo "moved $QN old-schema runs to $PRE (will be regenerated)"

echo "== [1/7] toolchain =="
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode command-line tools missing. Run:  xcode-select --install"
  echo "then re-run this script."; exit 1
fi
export PATH="$PATH:$HOME/.local/bin:$HOME/Library/Python/3.9/bin:$HOME/Library/Python/3.11/bin:$HOME/Library/Python/3.12/bin:$HOME/Library/Python/3.13/bin"
if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null; then
  echo "installing cmake + ninja via pip..."
  pip3 install --user -q cmake ninja
fi
command -v cmake >/dev/null || { echo "cmake still not on PATH; open a new terminal or 'brew install cmake ninja'"; exit 1; }

echo "== [2/7] ns-3.40 source =="
if [ ! -f "$NS3_DIR/CMakeLists.txt" ]; then
  curl -L --progress-bar -o "$NS3_HOME/ns3.tar.gz" \
    "https://gitlab.com/nsnam/ns-3-dev/-/archive/ns-3.40/ns-3-dev-ns-3.40.tar.gz"
  tar xzf "$NS3_HOME/ns3.tar.gz" -C "$NS3_HOME"
  mv "$NS3_HOME/ns-3-dev-ns-3.40" "$NS3_DIR"
  rm "$NS3_HOME/ns3.tar.gz"
fi

echo "== [3/7] configure =="
cd "$NS3_DIR"
if [ ! -d cmake-cache ]; then
  ./ns3 configure -d release \
    --enable-modules="aodv;wifi;applications;flow-monitor;mobility" \
    --disable-python --disable-examples --disable-tests --disable-werror
fi

echo "== [4/7] build scenario =="
cp "$REPO/ns3/grayhole-manet.cc" scratch/
./ns3 build
BIN="$NS3_DIR/build/scratch/ns3.40-grayhole-manet"
[ -x "$BIN" ] || BIN="$(ls "$NS3_DIR"/build/scratch/*grayhole-manet 2>/dev/null | head -1)"
export DYLD_LIBRARY_PATH="$NS3_DIR/build/lib:${DYLD_LIBRARY_PATH:-}"
# macOS SIP strips DYLD_* variables when passing through system binaries
# (xargs, bash), so carry the lib path in NS3LIB and re-export it in the worker.
export NS3LIB="$NS3_DIR/build/lib"
export BIN OUT
echo "binary: $BIN"

echo "== [5/7] campaign (525 runs + legacy mirror, parallel x$JOBS, resumable) =="
# Job list: one line "mob drop n seed dur field" per pending run, read by an
# inline worker (xargs -n 6), which avoids the 255-char -I{} limit of BSD xargs.
JOBFILE="$(mktemp)"
# main grid
for mob in rwp gm rw2d; do
  for drop in 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
    for n in 20 40 60 80 100; do
      for seed in 1 2 3 4 5; do
        tag="${mob}_n${n}_d${drop//./}_s${seed}"
        [ -s "$OUT/run_${tag}.csv" ] && continue
        echo "$mob $drop $n $seed 300 1000" >> "$JOBFILE"
      done
    done
  done
done
# legacy WIDECOM-2020 mirror: 30 nodes, sparse 2000 m square field, 600 s
for drop in 0.2 0.4 0.6; do
  for seed in 1 2 3 4 5; do
    tag="legacy_n30_d${drop//./}_s${seed}"
    [ -s "$OUT/run_${tag}.csv" ] && continue
    echo "legacy $drop 30 $seed 600 2000" >> "$JOBFILE"
  done
done
TOTAL=$(wc -l < "$JOBFILE" 2>/dev/null | tr -d ' ' || echo 0)
echo "pending runs: ${TOTAL:-0}"
if [ "${TOTAL:-0}" -gt 0 ]; then
  xargs -P "$JOBS" -n 6 bash -c '
    export DYLD_LIBRARY_PATH="$NS3LIB:${DYLD_LIBRARY_PATH:-}"
    mob="$0"; drop="$1"; n="$2"; seed="$3"; dur="$4"; field="$5"
    m="$mob"; [ "$mob" = "legacy" ] && m="rwp"
    tag="${mob}_n${n}_d${drop//./}_s${seed}"
    # Work inside $OUT and pass a plain relative filename: paths with spaces
    # break the ns-3 command line (--out= is truncated at a space).
    cd "$OUT" || { echo "FAILED $tag (cannot cd to OUT)"; exit 0; }
    out="run_${tag}.csv"
    "$BIN" --nNodes="$n" --mobility="$m" --dropRatio="$drop" \
      --duration="$dur" --area="$field" --seed="$seed" --out="$out" \
      >/dev/null 2>".err_${tag}.log"
    # Trust the file, not the exit code: verify a non-empty CSV appeared.
    if [ -s "$out" ]; then
      rm -f ".err_${tag}.log"; echo "done $tag"
    else
      rm -f "$out"; echo "FAILED $tag (see $OUT/.err_${tag}.log)"
    fi
  ' < "$JOBFILE"
fi
rm -f "$JOBFILE"
DONE=$(ls "$OUT"/run_*.csv 2>/dev/null | grep -vc adj || true)
echo "completed run files: $DONE"

echo "== [6/7] validate telemetry, then parse to datasets =="
cd "$REPO"
python3 ns3/validate_telemetry.py "$OUT" --expect paper || {
  echo "TELEMETRY VALIDATION FAILED: fix before building any dataset"; exit 1; }
# data_ns3_full keeps everything (used only by the legacy-mirror comparison).
# data_ns3_grid excludes the 15 legacy-mirror scenarios (main results).
python3 ns3/parse_ns3_to_dataset.py "$OUT" data_ns3_full
python3 ns3/parse_ns3_to_dataset.py "$OUT" data_ns3_grid --exclude-legacy

echo "== [7/7] train + evaluate on the NS-3 MAIN GRID (525 scenarios) =="
( cd models && python3 ns3_experiments.py )

echo ""
echo "=========================================================="
echo " DONE. Key outputs:"
echo "   $REPO/results/ns3_full_results.json"
echo "   $REPO/data_ns3_full/  (node_features.csv, sequences.npz)"
echo "=========================================================="
