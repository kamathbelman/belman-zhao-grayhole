"""Validate NS-3 per-node telemetry (schema revision 3) before building datasets.

Counts cover unicast transit data packets. Each run is checked for:
  1. Forwarding and drop ratios in [0,1].
  2. Conservation: cumulative rx == forwarded + dropped + in_flight per window.
  3. The same queueing structure on benign and gray-hole nodes.
  4. Recorded scenario parameters that match the expected configuration.
Usage: python3 validate_telemetry.py <runs_dir> [--expect paper|smoke]
Exit code 0 if all runs pass, 1 otherwise.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

EXPECTED_COLS = ["node", "is_gray", "window", "data_rx_forward",
                 "data_forwarded", "data_dropped", "rreq_forwarded",
                 "rrep_sent", "data_originated", "avg_neighbors",
                 "in_flight", "drop_selective", "drop_overflow",
                 "drop_no_route", "drop_route_error", "drop_ttl_expired",
                 "drop_arp_fail", "drop_ip_other", "rerr_sent",
                 "rreq_originated"]
SCHEMA_REV = 3
CAUSES = ["drop_selective", "drop_overflow", "drop_no_route",
          "drop_route_error", "drop_ttl_expired", "drop_arp_fail",
          "drop_ip_other"]

# Expected scenario configuration.
PAPER = {"area_m": 1000.0, "duration_s": 300.0, "window_s": 20.0,
         "commRange_m": 250.0, "pktSizeBytes": 512, "dataRate": "8kbps",
         "grayFrac": 0.2, "wifiStandard": "80211b", "routing": "AODV"}
# The WIDECOM 2020 mirror scenarios (run_legacy_*) differ: 30 nodes, a sparse
# 2000 m square field, 600 s, random waypoint.
LEGACY_OVERRIDES = {"area_m": 2000.0, "duration_s": 600.0}
ALLOWED_MOBILITY = {"rwp", "gm", "rw2d"}


class Failures(list):
    def add(self, msg):
        self.append(msg)


def check_run(csv_path, expect):
    fails = Failures()
    meta_path = csv_path + ".meta.json"
    adj_path = csv_path + ".adj.csv"
    if not os.path.exists(meta_path):
        fails.add("missing .meta.json (rerun with the current grayhole-manet.cc)")
        return fails
    if not os.path.exists(adj_path):
        fails.add("missing .adj.csv")
    meta = json.load(open(meta_path))
    df = pd.read_csv(csv_path)

    if meta.get("schemaRev") != SCHEMA_REV:
        fails.add(f"schemaRev {meta.get('schemaRev')} != {SCHEMA_REV} "
                  f"(older schema; regenerate)")
        return fails
    if list(df.columns) != EXPECTED_COLS:
        fails.add(f"schema mismatch: {list(df.columns)}")
        return fails
    if (df[EXPECTED_COLS[:9] + EXPECTED_COLS[10:]] < 0).any().any():
        fails.add("negative counter value")

    sc = meta["scenario"]
    qd = meta["queueDisc"]
    nodes = {n["id"]: n for n in meta["nodes"]}
    N = sc["nNodes"]

    # Check 1: ratios over resolved outcomes (forwarded + dropped) lie in [0,1].
    den = (df["data_forwarded"] + df["data_dropped"]).astype(float)
    fwd_ratio = np.where(den > 0, df["data_forwarded"] / np.maximum(den, 1), 1.0)
    drop_ratio = np.where(den > 0, df["data_dropped"] / np.maximum(den, 1), 0.0)
    if ((fwd_ratio < 0) | (fwd_ratio > 1)).any():
        fails.add("fwd_ratio outside [0,1]")
    if ((drop_ratio < 0) | (drop_ratio > 1)).any():
        fails.add("drop_ratio outside [0,1]")

    # Check 2a: the per-cause drop counters sum to data_dropped.
    if not (df[CAUSES].sum(axis=1) == df["data_dropped"]).all():
        fails.add("sum(drop causes) != data_dropped in some window")

    # Check 2b: per-window conservation, so that each transit packet has one
    # outcome: cum(rx) - cum(forwarded) - cum(dropped) == in_flight.
    for node, g in df.sort_values("window").groupby("node"):
        cum = (g["data_rx_forward"].cumsum()
               - g["data_forwarded"].cumsum()
               - g["data_dropped"].cumsum())
        if not (cum.values == g["in_flight"].values).all():
            w = int(g["window"].values[np.argmax(
                cum.values != g["in_flight"].values)])
            fails.add(f"node {node}: cum(rx-fwd-drop) != in_flight "
                      f"(first bad window {w}): outcome not unique")
            break

    # Check 2c: whole-run identity, rx == forwarded + dropped + pending_end.
    for i, nd in nodes.items():
        if nd["rx"] != nd["forwarded"] + nd["dropped"] + nd["pending_end"]:
            fails.add(f"node {i}: meta identity rx != fwd+drop+pending "
                      f"({nd['rx']} != {nd['forwarded']}+{nd['dropped']}"
                      f"+{nd['pending_end']})")
            break
        if sum(nd["causes"].values()) != nd["dropped"]:
            fails.add(f"node {i}: meta causes sum != dropped")
            break
    # window sums cannot exceed the whole-run totals (tail after the last flush)
    csv_rx = df.groupby("node")["data_rx_forward"].sum()
    for i in csv_rx.index:
        if csv_rx[i] > nodes[int(i)]["rx"]:
            fails.add(f"node {i}: windowed rx exceeds whole-run total")
            break

    # Check 3: the same queue disc on every node, with drop probability 0 on
    # benign nodes and dropRatio on gray nodes.
    if not qd.get("installedOnAllNodes", False):
        fails.add("queue disc not installed on all nodes")
    if qd.get("type") != "SelectiveDropQueueDisc" or qd.get("maxSize") != "500p":
        fails.add(f"unexpected queue disc config: {qd}")
    if qd.get("benignDropProbability", -1) != 0.0:
        fails.add("benign drop probability != 0")
    n_gray_meta = 0
    for i, nd in nodes.items():
        want = sc["dropRatio"] if nd["isGray"] else 0.0
        if abs(nd["dropProbability"] - want) > 1e-9:
            fails.add(f"node {i}: dropProbability {nd['dropProbability']} != "
                      f"{want} (isGray={nd['isGray']})")
            break
        n_gray_meta += nd["isGray"]
    want_gray = max(1, round(sc["grayFrac"] * N))
    if n_gray_meta != want_gray or n_gray_meta != sc["nGray"]:
        fails.add(f"gray count mismatch: meta nodes {n_gray_meta}, "
                  f"scenario.nGray {sc['nGray']}, expected {want_gray}")
    # CSV labels must match meta
    lab = df.groupby("node")["is_gray"].first()
    for i in lab.index:
        if int(lab[i]) != nodes[int(i)]["isGray"]:
            fails.add(f"node {i}: CSV is_gray != meta isGray")
            break
    # The attacker selects packets by size, so every selective drop must fall
    # inside the counted population (unicast transit data packets).
    if qd.get("selectiveDropsOutsideP", -1) != 0:
        fails.add(f"selective drops outside population P: "
                  f"{qd.get('selectiveDropsOutsideP')} (must be 0)")
    # the queue-disc configuration is verified at run time on every node
    if qd.get("runtimeVerifiedNodes") != N:
        fails.add(f"runtime disc verification covered "
                  f"{qd.get('runtimeVerifiedNodes')} of {N} nodes")
    # selective drops cannot occur on benign nodes
    ben = df[df["is_gray"] == 0]
    if ben["drop_selective"].sum() != 0:
        fails.add("benign node recorded selective drops (impossible if "
                  "DropProbability=0)")
    # and are expected on gray nodes when the attack is on and traffic flowed
    gray = df[df["is_gray"] == 1]
    if sc["dropRatio"] > 0 and gray["data_rx_forward"].sum() > 50 \
            and gray["drop_selective"].sum() == 0:
        fails.add("no selective drops on gray nodes despite traffic "
                  "(attack inactive?)")

    # Check 4: recorded parameters match the expected configuration.
    is_legacy = os.path.basename(csv_path).startswith("run_legacy_")
    prof = dict(PAPER)
    if is_legacy:
        prof.update(LEGACY_OVERRIDES)
    if expect == "smoke":  # short runs: duration and area as declared
        prof.pop("duration_s"); prof.pop("area_m")
    for k, want in prof.items():
        got = sc.get(k)
        ok = (abs(got - want) < 1e-9) if isinstance(want, float) else got == want
        if not ok:
            fails.add(f"param {k}: recorded {got}, paper says {want}")
    if is_legacy:
        if N != 30:
            fails.add(f"legacy mirror must have 30 nodes, got {N}")
        if sc["mobility"] != "rwp":
            fails.add(f"legacy mirror must be rwp, got {sc['mobility']}")
    if sc["mobility"] not in ALLOWED_MOBILITY:
        fails.add(f"mobility label '{sc['mobility']}' not in {ALLOWED_MOBILITY}"
                  " (the 2D random walk is labeled rw2d)")
    want_flows = max(8, N // 4)
    if sc["nFlows"] != want_flows:
        fails.add(f"nFlows {sc['nFlows']} != max(8, N/4) = {want_flows}")
    want_windows = int(sc["duration_s"] // sc["window_s"])
    n_windows = df["window"].nunique()
    if n_windows != want_windows:
        fails.add(f"{n_windows} windows recorded, expected {want_windows}")
    if df["node"].nunique() != N:
        fails.add(f"{df['node'].nunique()} nodes in CSV, expected {N}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--expect", choices=("paper", "smoke"), default="paper")
    args = ap.parse_args()

    runs = sorted(f for f in glob.glob(os.path.join(args.runs_dir, "*.csv"))
                  if not f.endswith(".adj.csv"))
    if not runs:
        print(f"no run CSVs found in {args.runs_dir}"); sys.exit(1)
    n_fail = 0
    for r in runs:
        fails = check_run(r, args.expect)
        tag = os.path.basename(r)
        if fails:
            n_fail += 1
            print(f"FAIL {tag}")
            for m in fails:
                print(f"     - {m}")
        else:
            print(f"PASS {tag}")
    print(f"\n{len(runs) - n_fail}/{len(runs)} runs passed "
          f"(profile: {args.expect})")
    if n_fail == 0:
        print("telemetry checks 1-4 all hold; safe to parse and rerun models")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
