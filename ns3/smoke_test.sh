#!/usr/bin/env bash
# smoke_test.sh: small smoke test grid, to run before a full campaign.
# validate_telemetry.py then checks four conditions:
#   1. all forwarding/drop ratios in [0,1]
#   2. every transit data packet has one consistent final outcome
#   3. benign and gray nodes use identical queueing structures
#   4. recorded scenario parameters match the expected configuration
# and parse_ns3_to_dataset.py must produce the 10-feature schema.
# Usage: NS3_DIR=~/ns3-grayhole/ns3 bash smoke_test.sh
#   (default: $HOME/ns3-grayhole/ns3, set up by run_full_campaign_macos.sh)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
NS3_DIR="${NS3_DIR:-$HOME/ns3-grayhole/ns3}"
OUT="${OUT:-$HERE/ns3_smoke}"
DUR="${DUR:-100}"
# pip-installed cmake/ninja live here on macOS
export PATH="$PATH:$HOME/.local/bin:$HOME/Library/Python/3.9/bin:$HOME/Library/Python/3.11/bin:$HOME/Library/Python/3.12/bin:$HOME/Library/Python/3.13/bin"

[ -f "$NS3_DIR/CMakeLists.txt" ] || {
  echo "NS3_DIR=$NS3_DIR is not an ns-3 tree (run run_full_campaign_macos.sh "
  echo "steps 1-4 first, or set NS3_DIR)"; exit 1; }
command -v cmake >/dev/null || {
  echo "cmake not on PATH (pip3 install --user cmake ninja, or brew install"
  echo "cmake ninja), then re-run"; exit 1; }

echo "== build scenario (ns-3 tree: $NS3_DIR) =="
cp "$HERE/grayhole-manet.cc" "$NS3_DIR/scratch/"
BUILD_LOG="$(mktemp -t ns3build)"
if ! ( cd "$NS3_DIR" && ./ns3 build ) > "$BUILD_LOG" 2>&1; then
  echo "BUILD FAILED, last 30 lines of $BUILD_LOG:"
  tail -30 "$BUILD_LOG"
  exit 1
fi
BIN="$(ls "$NS3_DIR"/build/scratch/*grayhole-manet 2>/dev/null | head -1)"
[ -x "$BIN" ] || { echo "build ok but no grayhole-manet binary found in "
  echo "$NS3_DIR/build/scratch contents:"; ls "$NS3_DIR/build/scratch"; exit 1; }
echo "binary: $BIN"
export LD_LIBRARY_PATH="$NS3_DIR/build/lib:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$NS3_DIR/build/lib:${DYLD_LIBRARY_PATH:-}"

mkdir -p "$OUT"; cd "$OUT"
echo "== smoke grid (5 runs, ${DUR}s each) =="
run() { # mob n drop seed
  local tag="$1_n$2_d${3//./}_s$4"
  echo "  run_$tag.csv"
  "$BIN" --nNodes="$2" --mobility="$1" --dropRatio="$3" --seed="$4" \
         --duration="$DUR" --out="run_$tag.csv" >"err_$tag.log" 2>&1 || true
  # trust the file, not the exit code
  if [ -s "run_$tag.csv" ]; then rm -f "err_$tag.log"; else
    echo "  RUN FAILED, err_$tag.log:"; tail -20 "err_$tag.log"; exit 1
  fi
}
run rwp  20 0.4 1     # all three mobilities at N=20
run gm   20 0.4 1
run rw2d 20 0.4 1
run rw2d 40 0.2 1     # second size: flows rule must give max(8,40/4)=10
run rwp  20 0.0 2     # edge case: attack off (dropRatio=0)

echo "== validate telemetry (checks 1-4) =="
python3 "$HERE/validate_telemetry.py" "$OUT" --expect smoke

echo "== parse to dataset (must be 10 features) =="
TMPD="$(mktemp -d)"
python3 "$HERE/parse_ns3_to_dataset.py" "$OUT" "$TMPD"
python3 - "$TMPD" <<'EOF'
import json, sys, numpy as np
meta = json.load(open(f"{sys.argv[1]}/meta.json"))
assert meta["n_features"] == 10, meta["n_features"]
assert "routes_advertised" not in meta["seq_keys"]
assert "manhattan" not in str(meta).lower()
z = np.load(f"{sys.argv[1]}/sequences.npz", allow_pickle=True)
assert z["seq"].shape[-1] == 10
print(f"dataset OK: F=10, keys={meta['seq_keys']}")
EOF
rm -rf "$TMPD"
echo ""
echo "SMOKE TEST PASSED: safe to launch the full campaign."
