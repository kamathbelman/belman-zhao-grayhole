"""Convert NS-3 run outputs into the dataset schema of the Python simulator.

Input: a directory of per-run CSVs written by grayhole-manet.cc (one row per
node and window), with matching *.adj.csv adjacency and *.meta.json metadata.
Output in <outdir>: node_features.csv (per-node aggregates and label),
sequences.npz (per-node window sequences), topology.npz (adjacency, more than
20% contact time) and meta.json. Aggregates are mean, std and min over windows.

Run: python3 parse_ns3_to_dataset.py [runs_dir] [outdir] [--exclude-legacy]
"""
from __future__ import annotations
import os, sys, glob, re, json
import numpy as np
import pandas as pd

TAG_RE = re.compile(
    r"(?P<mob>rwp|gm|rw2d|grid|legacy)_n(?P<n>\d+)_d(?P<d>\d+)_s(?P<s>\d+)\.csv$")
# rw2d and grid both denote a bounded 2D random walk. "legacy" runs mirror the
# WIDECOM 2020 setup (random waypoint on a sparse 2000 m square field): they map
# to random_waypoint but keep their own scenario tag.
MOB = {"rwp": "random_waypoint", "gm": "gauss_markov",
       "rw2d": "random_walk_2d", "grid": "random_walk_2d",
       "legacy": "random_waypoint"}

# The 10 model features.
SEQ_KEYS = ["fwd_ratio", "drop_ratio", "data_rx_forward", "data_forwarded",
            "data_dropped", "rreq_forwarded", "rrep_sent",
            "data_originated", "avg_neighbors", "ctrl_to_data"]

# Validation-only columns (see validate_telemetry.py). They never enter the
# feature set.
VALIDATION_COLS = ["in_flight", "drop_selective", "drop_overflow",
                   "drop_no_route", "drop_route_error", "drop_ttl_expired",
                   "drop_arp_fail", "drop_ip_other", "rerr_sent",
                   "rreq_originated"]  # rreq_forwarded holds relayed RREQs only


def parse_run(csv_path):
    m = TAG_RE.search(os.path.basename(csv_path))
    if not m:
        return None
    df = pd.read_csv(csv_path)
    if "routes_advertised" in df.columns:
        raise SystemExit(
            f"{csv_path}: old-schema file (has routes_advertised). This "
            f"parser expects schema revision 3; regenerate the runs with the "
            f"current grayhole-manet.cc.")
    if "rreq_originated" not in df.columns:
        raise SystemExit(
            f"{csv_path}: older file (rreq_forwarded mixes originated "
            f"and relayed RREQs). Regenerate with the current binary.")
    # Ratios are defined over resolved outcomes (forwarded + dropped), so both
    # lie in [0,1]. With no outcomes, fwd_ratio is 1.0 and drop_ratio is 0.0.
    den = (df["data_forwarded"] + df["data_dropped"]).astype(float)
    df["fwd_ratio"] = np.where(den > 0, df["data_forwarded"] / np.maximum(den, 1), 1.0)
    df["drop_ratio"] = np.where(den > 0, df["data_dropped"] / np.maximum(den, 1), 0.0)
    df["ctrl_to_data"] = ((df["rreq_forwarded"] + df["rrep_sent"])
                          / (df["data_forwarded"] + 1))
    bad = ((df["drop_ratio"] < 0) | (df["drop_ratio"] > 1)
           | (df["fwd_ratio"] < 0) | (df["fwd_ratio"] > 1))
    if bad.any():
        raise SystemExit(f"{csv_path}: ratio outside [0,1], telemetry bug, "
                         f"run validate_telemetry.py")
    meta = dict(mobility=MOB[m.group("mob")], drop_ratio=int(m.group("d")) / 10.0,
                n_nodes=int(m.group("n")))
    return df, meta


def main(indir, outdir, exclude_legacy=False):
    runs = sorted(glob.glob(os.path.join(indir, "*.csv")))
    runs = [r for r in runs if not r.endswith(".adj.csv")]
    if exclude_legacy:
        # the legacy mirror scenarios are used only for the prior-work comparison
        runs = [r for r in runs
                if not os.path.basename(r).startswith("run_legacy_")]
    rows, seq_list, seq_labels, seq_scen, topo = [], [], [], [], {}
    sid, max_w = 0, 0
    for r in runs:
        parsed = parse_run(r)
        if parsed is None:
            continue
        df, meta = parsed
        n = meta["n_nodes"]
        piv = {k: df.pivot(index="node", columns="window", values=k)
                    .sort_index() for k in SEQ_KEYS}
        W = piv[SEQ_KEYS[0]].shape[1]
        max_w = max(max_w, W)
        labels = df.groupby("node")["is_gray"].first().sort_index().values
        seq = np.stack([piv[k].values for k in SEQ_KEYS], axis=-1)  # (n,W,F)
        agg = np.concatenate([seq.mean(1), seq.std(1), seq.min(1)], axis=1)
        agg_keys = ([f"{k}_mean" for k in SEQ_KEYS] + [f"{k}_std" for k in SEQ_KEYS]
                    + [f"{k}_min" for k in SEQ_KEYS])
        for i in range(n):
            row = {"scenario": sid, "node": i, "label": int(labels[i]),
                   "mobility": meta["mobility"], "drop_ratio": meta["drop_ratio"],
                   "n_nodes": n}
            row.update({k: float(v) for k, v in zip(agg_keys, agg[i])})
            rows.append(row)
            seq_list.append(seq[i].astype(np.float32))
            seq_labels.append(int(labels[i]))
            seq_scen.append(sid)
        adj_path = r + ".adj.csv"
        if os.path.exists(adj_path):
            frac = np.loadtxt(adj_path, delimiter=",")
            topo[str(sid)] = frac > 0.2
        else:
            topo[str(sid)] = np.zeros((n, n), dtype=bool)
        sid += 1

    if not rows:
        print("no parseable runs found in", indir); return
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "node_features.csv"), index=False)
    F = seq_list[0].shape[1]
    padded = np.zeros((len(seq_list), max_w, F), dtype=np.float32)
    mask = np.zeros((len(seq_list), max_w), dtype=np.float32)
    for j, s in enumerate(seq_list):
        w = s.shape[0]
        padded[j, max_w - w:] = s
        mask[j, max_w - w:] = 1.0
    np.savez_compressed(os.path.join(outdir, "sequences.npz"),
                        seq=padded, mask=mask,
                        labels=np.array(seq_labels, dtype=np.int64),
                        scenario=np.array(seq_scen, dtype=np.int64),
                        feat_keys=np.array(SEQ_KEYS))
    np.savez_compressed(os.path.join(outdir, "topology.npz"), **topo)
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump({"scale": "ns3", "n_scenarios": sid, "seq_keys": SEQ_KEYS,
                   "n_features": len(SEQ_KEYS),
                   "telemetry_rev": 3,
                   "ratio_definition": "outcome-based: x/(fwd+drop)",
                   "validation_cols": VALIDATION_COLS,
                   "max_windows": int(max_w)}, f, indent=2)
    n_gray = int(np.sum(seq_labels))
    print(f"wrote {outdir}: {sid} scenarios, {len(seq_list)} nodes, {n_gray} gray, "
          f"F={len(SEQ_KEYS)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--exclude-legacy"]
    indir = args[0] if len(args) > 0 else "ns3_runs"
    outdir = args[1] if len(args) > 1 else "../data_ns3"
    main(indir, outdir, exclude_legacy="--exclude-legacy" in sys.argv)
