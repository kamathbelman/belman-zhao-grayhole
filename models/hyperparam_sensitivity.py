"""Sensitivity sweep over baseline hyperparameters on the bootstrap data.

Re-evaluates the SVM over a range of C and the random forest over tree-count
and depth grids, with the same scenario-disjoint protocol and split seeds as
Table II. The main experiments keep their fixed hyperparameters; the sweep is
not used for model selection.
Usage: python3 hyperparam_sensitivity.py
Writes: ../results/hyperparam_sensitivity.json"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results")
SEEDS = (0, 1, 2)  # the Table II split seeds


def eval_cfg(make):
    df, feat_cols = common.load_aggregate()
    y = df["label"].values.astype(int)
    scen = df["scenario"].values
    X = df[feat_cols].values.astype(float)
    f1s, aucs = [], []
    for seed in SEEDS:
        tr, te = common.scenario_split(scen, y, test_frac=0.3, seed=seed)
        std = common.Standardizer().fit(X[tr])
        m = make(seed)
        m.fit(std.transform(X[tr]), y[tr])
        Xte = std.transform(X[te])
        pred = m.predict(Xte)
        score = (m.decision_function(Xte) if hasattr(m, "decision_function")
                 else m.predict_proba(Xte)[:, 1])
        f1s.append(common.metrics(y[te], pred)["f1"])
        aucs.append(common.roc_auc(y[te], score))
    return {"f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "auc_mean": float(np.mean(aucs))}


def main():
    out = {"protocol": "scenario-disjoint, seeds (0,1,2), identical to "
                       "Table II; post-hoc sensitivity only, paper "
                       "constants unchanged",
           "reference": {"SLM_f1": 0.782, "paper_SVM_C4_f1": 0.693,
                         "paper_RF_200_f1": 0.735},
           "svm_C_sweep": {}, "rf_sweep": {}}
    for C in (0.5, 1, 2, 4, 8, 16, 32):
        r = eval_cfg(lambda seed, C=C: SVC(
            kernel="rbf", C=C, gamma="scale", class_weight="balanced"))
        out["svm_C_sweep"][str(C)] = r
        print(f"SVM C={C}: F1 {r['f1_mean']:.3f}+/-{r['f1_std']:.3f} "
              f"AUC {r['auc_mean']:.3f}")
    for n, d in itertools.product((100, 200, 400), (None, 8, 16)):
        r = eval_cfg(lambda seed, n=n, d=d: RandomForestClassifier(
            n_estimators=n, max_depth=d, class_weight="balanced",
            random_state=seed, n_jobs=-1))
        out["rf_sweep"][f"trees{n}_depth{d}"] = r
        print(f"RF trees={n} depth={d}: F1 {r['f1_mean']:.3f}"
              f"+/-{r['f1_std']:.3f} AUC {r['auc_mean']:.3f}")
    best_svm = max(v["f1_mean"] for v in out["svm_C_sweep"].values())
    best_rf = max(v["f1_mean"] for v in out["rf_sweep"].values())
    out["conclusion"] = {
        "best_svm_f1_any_C": best_svm,
        "best_rf_f1_any_config": best_rf,
        "slm_margin_over_best_baseline": 0.782 - max(best_svm, best_rf)}
    path = os.path.join(RESULTS, "hyperparam_sensitivity.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
