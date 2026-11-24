"""Diagnostic for in-sample versus out-of-fold training of HRM tier 3.

Reports, per dataset and split seed, tier-1 AUC and Brier score on training
rows (in-sample and out-of-fold) and test rows, plus tier-3 coefficients, fit
time and test metrics in each mode, and tier 1 alone at the 0.5 threshold. A
large in-sample to test gap is the optimism an in-sample tier 3 is trained on.
Usage (from models/): python3 hrm_oof_diagnostic.py
Output: ../results/hrm_oof_diagnostic.json"""
from __future__ import annotations
import json, os, time
import numpy as np

import common
import hrm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
Z_NAMES = ("p_local", "nbr_mean", "nbr_max", "nbr_gap", "deg_norm",
           "scen_ctx")


def _brier(y, p):
    return float(np.mean((p - y) ** 2))


def one(seed):
    df, feat_cols = common.load_aggregate()
    X = df[feat_cols].values.astype(np.float64)
    y = df["label"].values.astype(int)
    scen = df["scenario"].values
    topo = hrm._load_topology()
    tr, te = common.scenario_split(scen, y, test_frac=0.3, seed=seed)
    Xs = common.Standardizer().fit(X[tr]).transform(X)

    out = {"seed": seed}
    fits = {}
    for mode, oof in (("in_sample", False), ("out_of_fold", True)):
        t0 = time.perf_counter()
        tier1, tier3, _ = hrm.fit_hrm(df, Xs, y, scen, topo, tr, seed, oof)
        fits[mode] = (tier1, tier3, time.perf_counter() - t0)

    tier1 = fits["in_sample"][0]              # same final tier 1 in both modes
    p_all = tier1.predict_proba(Xs)[:, 1]
    p_oof = np.full(len(y), np.nan)
    for held in hrm.oof_folds(scen, tr, seed):
        ho = tr & np.isin(scen, held)
        g = hrm._new_tier1(seed).fit(Xs[tr & ~ho], y[tr & ~ho])
        p_oof[ho] = g.predict_proba(Xs[ho])[:, 1]
    out["tier1_scores"] = {
        "train_in_sample": {"auc": common.roc_auc(y[tr], p_all[tr]),
                            "brier": _brier(y[tr], p_all[tr])},
        "train_out_of_fold": {"auc": common.roc_auc(y[tr], p_oof[tr]),
                              "brier": _brier(y[tr], p_oof[tr])},
        "test": {"auc": common.roc_auc(y[te], p_all[te]),
                 "brier": _brier(y[te], p_all[te])}}
    m1 = common.metrics(y[te], (p_all[te] >= 0.5).astype(int), p_all[te])
    out["tier1_alone_test"] = {k: m1[k] for k in
                               ("accuracy", "precision", "recall", "f1", "auc")}

    Z = hrm._build_Z(df, scen, topo, p_all)
    for mode, (_, tier3, fit_t) in fits.items():
        s = tier3.predict_proba(Z)[:, 1][te]
        m = common.metrics(y[te], (s >= 0.5).astype(int), s)
        out[mode] = {
            "fit_time_s": fit_t,
            "tier3_coef": dict(zip(Z_NAMES, map(float, tier3.coef_[0]))),
            "tier3_intercept": float(tier3.intercept_[0]),
            "test": {k: m[k] for k in
                     ("accuracy", "precision", "recall", "f1", "auc")}}
    return out


def main():
    res = {"note": "diagnostic only; no paper number comes from this file"}
    for dataset in ("data", "data_ns3_grid"):
        common.DATA = os.path.join(ROOT, dataset)
        hrm.DATA = common.DATA
        res[dataset] = [one(seed) for seed in (0, 1, 2)]
        for r in res[dataset]:
            t = r["tier1_scores"]
            print(f"{dataset} seed {r['seed']}: tier-1 AUC train in-sample "
                  f"{t['train_in_sample']['auc']:.4f}, train out-of-fold "
                  f"{t['train_out_of_fold']['auc']:.4f}, test "
                  f"{t['test']['auc']:.4f} | HRM test F1 in-sample "
                  f"{r['in_sample']['test']['f1']:.4f}, out-of-fold "
                  f"{r['out_of_fold']['test']['f1']:.4f}, tier 1 alone "
                  f"{r['tier1_alone_test']['f1']:.4f}", flush=True)
    path = os.path.join(ROOT, "results", "hrm_oof_diagnostic.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
