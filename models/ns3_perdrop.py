"""Per-attack-intensity breakdown on the NS-3 main grid (data_ns3_grid).

Trains the RBF-SVM (aggregate features) and the attention model (sequences)
on scenario-disjoint splits and reports F1 per attack drop ratio p_a, mean
+/- std over three seeds. Legacy-mirror scenarios (N=30) are excluded.
Writes results/ns3_full_perdrop.json. Usage: python3 ns3_perdrop.py"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from sklearn.svm import SVC

import common
from slm_numpy import SelfAttnClassifier

# Grid-only dataset (no legacy-mirror scenarios); the n_nodes != 30 filters
# below are a redundant guard.
DATA = os.path.join(os.path.dirname(__file__), "..", "data_ns3_grid")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def one_seed(seed):
    df = pd.read_csv(os.path.join(DATA, "node_features.csv"))
    feat = [c for c in df.columns
            if c not in ("scenario", "node", "label", "mobility",
                         "drop_ratio", "n_nodes")]
    y = df["label"].values.astype(int)
    scen = df["scenario"].values
    tr, te = common.scenario_split(scen, y, 0.3, seed)

    std = common.Standardizer().fit(df[feat].values[tr])
    X = std.transform(df[feat].values)
    svm = SVC(kernel="rbf", C=4.0, gamma="scale",
              class_weight="balanced").fit(X[tr], y[tr])
    svm_pred = svm.predict(X[te])

    z = np.load(os.path.join(DATA, "sequences.npz"), allow_pickle=True)
    seq, mask, ys, scn = z["seq"], z["mask"], z["labels"], z["scenario"]
    trs, tes = common.scenario_split(scn, ys, 0.3, seed)
    valid = mask[trs].astype(bool)
    flat = seq[trs][valid]
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    seqn = ((seq - mu) / sd) * mask[:, :, None]
    net = SelfAttnClassifier(seq.shape[2], seq.shape[1], 24, 32, seed=seed)
    net.fit(seqn[trs], mask[trs], ys[trs].astype(float), epochs=40, seed=seed)
    slm_pred = (net.predict_proba(seqn[tes], mask[tes]) >= 0.5).astype(int)

    scen_meta = df.groupby("scenario")[["drop_ratio", "n_nodes"]].first()
    d_te = df.loc[te, "drop_ratio"].values
    n_te = df.loc[te, "n_nodes"].values
    slm_d = scen_meta.loc[scn[tes], "drop_ratio"].values
    slm_n = scen_meta.loc[scn[tes], "n_nodes"].values

    out = {}
    for pa in sorted(df[df.n_nodes != 30]["drop_ratio"].unique()):
        mA = (d_te == pa) & (n_te != 30)
        mB = (slm_d == pa) & (slm_n != 30)
        out[str(pa)] = {
            "svm_f1": common.metrics(y[te][mA], svm_pred[mA])["f1"]
            if mA.sum() else None,
            "slm_f1": common.metrics(ys[tes][mB], slm_pred[mB])["f1"]
            if mB.sum() else None}
    return out


def main():
    seeds = (0, 1, 2)
    per = [one_seed(s) for s in seeds]
    drops = sorted(per[0].keys(), key=float)
    agg = {}
    print(f"{'pa':>5s} {'SVM F1':>16s} {'SLM F1':>16s}")
    for dr in drops:
        sv = np.array([p[dr]["svm_f1"] for p in per], float)
        sl = np.array([p[dr]["slm_f1"] for p in per], float)
        agg[dr] = {"svm_f1_mean": sv.mean(), "svm_f1_std": sv.std(),
                   "slm_f1_mean": sl.mean(), "slm_f1_std": sl.std()}
        print(f"{dr:>5s} {sv.mean():8.3f}±{sv.std():.3f} "
              f"{sl.mean():8.3f}±{sl.std():.3f}")
    out = os.path.join(RESULTS, "ns3_full_perdrop.json")
    with open(out, "w") as f:
        json.dump({"seeds": list(seeds), "telemetry_rev": 3,
                   "per_drop": agg}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
