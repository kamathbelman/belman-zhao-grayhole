"""Run the Python simulator over the grid and write a labeled dataset.

Grid: mobility model, drop_ratio and n_nodes, with seed replicates.
Written to ../data:
  node_features.csv  aggregate per-node features, label and scenario metadata
  sequences.npz      padded per-node window sequences
  topology.npz       per-scenario adjacency for HRM tier 2

Run: python3 generate_dataset.py [demo|full]
"""
from __future__ import annotations
import os, sys, itertools, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from manet_sim import SimConfig, ManetSim

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def build_grid(scale: str = "demo"):
    """Return (scenario id, SimConfig) pairs. scale is 'demo' (small) or 'full'."""
    if scale == "demo":
        mobilities = ["random_waypoint", "gauss_markov", "manhattan"]
        drops = [0.2, 0.4, 0.6]
        sizes = [30, 50, 80]
        seeds = range(3)
    else:
        mobilities = ["random_waypoint", "gauss_markov", "manhattan"]
        drops = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        sizes = [20, 40, 60, 80, 100]
        seeds = range(5)

    configs = []
    sid = 0
    for mob, drop, size, seed in itertools.product(mobilities, drops, sizes, seeds):
        cfg = SimConfig(
            n_nodes=size,
            mobility=mob,
            drop_ratio=drop,
            gray_fraction=0.20,
            duration=300.0,
            n_flows=max(8, size // 4),
            seed=1000 * sid + seed,
        )
        configs.append((sid, cfg))
        sid += 1
    return configs


def main(scale="demo"):
    configs = build_grid(scale)
    print(f"[generate] scale={scale}  scenarios={len(configs)}")

    rows = []
    seq_list, seq_labels, seq_scen = [], [], []
    topo = {}
    max_windows = 0
    agg_keys = None
    seq_keys = None

    for sid, cfg in configs:
        out = ManetSim(cfg).run()
        n = cfg.n_nodes
        labels = out["labels"]
        agg = out["agg"]
        agg_keys = out["agg_keys"]
        seq_keys = out["seq_feat_keys"]
        for i in range(n):
            row = {"scenario": sid, "node": i, "label": int(labels[i]),
                   "mobility": cfg.mobility, "drop_ratio": cfg.drop_ratio,
                   "n_nodes": cfg.n_nodes}
            row.update({k: float(v) for k, v in zip(agg_keys, agg[i])})
            rows.append(row)
        seq = out["seq"]                    # (n, W, F)
        max_windows = max(max_windows, seq.shape[1])
        for i in range(n):
            seq_list.append(seq[i])
            seq_labels.append(int(labels[i]))
            seq_scen.append(sid)
        topo[str(sid)] = out["adj"]
        if sid % 10 == 0:
            print(f"  scenario {sid}/{len(configs)}  n={n} mob={cfg.mobility} "
                  f"drop={cfg.drop_ratio} gray={labels.sum()}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(DATA_DIR, "node_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"[generate] wrote {csv_path}  ({len(df)} rows, {df['label'].sum()} gray)")

    # pad sequences to a common length, with zeros in front
    F = seq_list[0].shape[1]
    padded = np.zeros((len(seq_list), max_windows, F), dtype=np.float32)
    mask = np.zeros((len(seq_list), max_windows), dtype=np.float32)
    for j, s in enumerate(seq_list):
        w = s.shape[0]
        padded[j, max_windows - w:] = s
        mask[j, max_windows - w:] = 1.0
    np.savez_compressed(
        os.path.join(DATA_DIR, "sequences.npz"),
        seq=padded, mask=mask,
        labels=np.array(seq_labels, dtype=np.int64),
        scenario=np.array(seq_scen, dtype=np.int64),
        feat_keys=np.array(seq_keys),
    )
    print(f"[generate] wrote sequences.npz  seq={padded.shape}")

    np.savez_compressed(os.path.join(DATA_DIR, "topology.npz"),
                        **{k: v for k, v in topo.items()})
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump({"scale": scale, "n_scenarios": len(configs),
                   "agg_keys": agg_keys, "seq_keys": seq_keys,
                   "max_windows": int(max_windows)}, f, indent=2)
    print("[generate] done.")


if __name__ == "__main__":
    scale = sys.argv[1] if len(sys.argv) > 1 else "demo"
    main(scale)
