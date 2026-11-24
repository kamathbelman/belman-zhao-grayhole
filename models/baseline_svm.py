"""SVM baseline in the style of Rao et al. (WIDECOM 2020): an RBF SVM on the
aggregate per-node features (mean, std and min of each window statistic).
A random forest is included as a second classical reference.
"""
from __future__ import annotations
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import common


def run(seed=0):
    df, feat_cols = common.load_aggregate()
    X = df[feat_cols].values.astype(np.float64)
    y = df["label"].values.astype(int)
    tr, te = common.scenario_split(df["scenario"].values, y, test_frac=0.3, seed=seed)

    std = common.Standardizer().fit(X[tr])
    Xtr, Xte = std.transform(X[tr]), std.transform(X[te])

    results = {}
    t0 = time.perf_counter()
    svm = SVC(kernel="rbf", C=4.0, gamma="scale", class_weight="balanced",
              probability=False)
    svm.fit(Xtr, y[tr])
    train_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = svm.predict(Xte)
    score = svm.decision_function(Xte)
    # amortized CPU time per node (batch time / n), not per-sample latency
    amort_t = (time.perf_counter() - t0) / len(Xte) * 1e3
    m = common.metrics(y[te], pred, score)
    m.update(train_time_s=train_t, amortized_cpu_ms_per_node=amort_t,
             n_test=int(te.sum()))
    results["SVM-RBF"] = m

    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=seed, n_jobs=-1)
    rf.fit(Xtr, y[tr])
    train_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = rf.predict(Xte)
    score = rf.predict_proba(Xte)[:, 1]
    amort_t = (time.perf_counter() - t0) / len(Xte) * 1e3
    m = common.metrics(y[te], pred, score)
    m.update(train_time_s=train_t, amortized_cpu_ms_per_node=amort_t,
             n_test=int(te.sum()))
    results["RandomForest"] = m
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, default=float))
