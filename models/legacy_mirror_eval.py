"""Prior-work comparison on the legacy-mirror scenarios.

Trains the WIDECOM-style RBF-SVM on the main NS-3 grid scenarios of
data_ns3_full and evaluates on the 15 legacy-mirror scenarios (the only
N=30 scenarios: 30 nodes, sparse 2000 m field, 600 s, random waypoint).
Writes results/legacy_mirror_svm.json.
Usage: python3 legacy_mirror_eval.py  (run after run_full_campaign_macos.sh)"""
from __future__ import annotations
import json, os
import pandas as pd
from sklearn.svm import SVC

import common

DATA = os.path.join(os.path.dirname(__file__), "..", "data_ns3_full")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    df = pd.read_csv(os.path.join(DATA, "node_features.csv"))
    feat = [c for c in df.columns
            if c not in ("scenario", "node", "label", "mobility",
                         "drop_ratio", "n_nodes")]
    y = df["label"].values.astype(int)
    grid = (df["n_nodes"] != 30).values      # legacy mirror is the only N=30
    legacy = ~grid
    n_leg_scen = df.loc[legacy, "scenario"].nunique()
    assert n_leg_scen == 15, f"expected 15 legacy scenarios, got {n_leg_scen}"

    std = common.Standardizer().fit(df[feat].values[grid])
    X = std.transform(df[feat].values)
    svm = SVC(kernel="rbf", C=4.0, gamma="scale",
              class_weight="balanced").fit(X[grid], y[grid])
    m = common.metrics(y[legacy], svm.predict(X[legacy]))
    out = os.path.join(RESULTS, "legacy_mirror_svm.json")
    with open(out, "w") as f:
        json.dump(m, f, indent=2, default=float)
    print(f"legacy mirror: acc={m['accuracy']:.3f} f1={m['f1']:.3f} "
          f"prec={m['precision']:.3f} rec={m['recall']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
