"""Hierarchical reasoning model (HRM) with three tiers.

Tier 1 (local): a gradient-boosted scorer maps a node's own aggregate
features to a local anomaly score. Tier 2 (neighborhood): mean and max of the
neighbors' tier-1 scores and the node's gap to that mean. Tier 3 (consensus):
a logistic regression over these and the scenario context, fit on out-of-fold
tier-1 scores (see fit_hrm). In-sample variant: python3 hrm.py --insample
"""
from __future__ import annotations
import os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

import common

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

N_OOF_FOLDS = 5


def default_oof():
    """Out-of-fold tier-3 training unless HRM_INSAMPLE=1 is set."""
    return os.environ.get("HRM_INSAMPLE") != "1"


def _load_topology():
    z = np.load(os.path.join(DATA, "topology.npz"), allow_pickle=True)
    return {int(k): z[k] for k in z.files}


def _build_Z(df, scen, topo, p_local):
    """Tier-2 neighborhood aggregation and tier-3 scenario context features."""
    nbr_mean = np.zeros(len(df))
    nbr_max = np.zeros(len(df))
    nbr_gap = np.zeros(len(df))
    deg = np.zeros(len(df))
    for sid in np.unique(scen):
        idx = np.where(scen == sid)[0]
        # rows are already in node order 0..n-1 within a scenario
        adj = topo[sid]
        local_s = p_local[idx]
        n = len(idx)
        for node in range(n):
            nbrs = np.where(adj[node])[0]
            row = idx[node]
            if len(nbrs) > 0:
                ns = local_s[nbrs]
                nbr_mean[row] = ns.mean()
                nbr_max[row] = ns.max()
                nbr_gap[row] = local_s[node] - ns.mean()
                deg[row] = len(nbrs)
            else:
                nbr_mean[row] = local_s[node]
                nbr_max[row] = local_s[node]
                nbr_gap[row] = 0.0
                deg[row] = 0
    # scenario-level context: mean local score of the scenario
    scen_ctx = np.zeros(len(df))
    for sid in np.unique(scen):
        idx = scen == sid
        scen_ctx[idx] = p_local[idx].mean()
    return np.column_stack([p_local, nbr_mean, nbr_max, nbr_gap,
                            deg / (deg.max() + 1e-9), scen_ctx])


def _new_tier1(seed):
    """Tier-1 local scorer, shared by the final model and every fold model."""
    return GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                      random_state=seed)


def oof_folds(scen, tr, seed, n_folds=N_OOF_FOLDS):
    """Scenario-grouped folds over the training scenarios only: the sorted
    ids are shuffled with default_rng(seed) and dealt round-robin into
    n_folds folds. A scenario is never split across folds, so a node and its
    neighbors are scored by the same fold model."""
    train_scen = np.unique(scen[tr])
    perm = np.random.default_rng(seed).permutation(train_scen)
    return [np.sort(perm[k::n_folds]) for k in range(n_folds)]


def fit_hrm(df, Xs, y, scen, topo, tr, seed=0, oof=None):
    """Fit the three-tier HRM on the training rows `tr` (boolean mask).

    Returns (tier1, tier3, info). tier1 is fit on all training rows and is
    the model used at inference. With oof=True, tier 3 is fit on out-of-fold
    tier-1 scores: each fold of training scenarios is scored by a tier-1
    model fit on the other folds. With oof=False it is fit on the in-sample
    scores of tier1, which are sharper than the scores it sees at test time.
    Xs must already be standardized with training-row statistics.
    """
    if oof is None:
        oof = default_oof()
    tier1 = _new_tier1(seed).fit(Xs[tr], y[tr])
    p_local = tier1.predict_proba(Xs)[:, 1]          # local score for all nodes
    info = {"oof": bool(oof)}
    p_fit = p_local
    if oof:
        p_fit = p_local.copy()
        folds = oof_folds(scen, tr, seed)
        scored = np.zeros(len(y), dtype=int)
        for held in folds:
            ho = tr & np.isin(scen, held)            # held-out training rows
            fit = tr & ~ho                           # the other folds
            fold_model = _new_tier1(seed).fit(Xs[fit], y[fit])
            p_fit[ho] = fold_model.predict_proba(Xs[ho])[:, 1]
            scored[ho] += 1
        # every training row scored exactly once, no test row touched
        assert (scored[tr] == 1).all() and (scored[~tr] == 0).all()
        info["oof_folds"] = [f.tolist() for f in folds]
    # _build_Z works scenario by scenario, so Z_fit[tr] depends only on the
    # training-row scores in p_fit (the degree normalizer does not use scores)
    Z_fit = _build_Z(df, scen, topo, p_fit)
    tier3 = LogisticRegression(max_iter=1000, class_weight="balanced")
    tier3.fit(Z_fit[tr], y[tr])
    return tier1, tier3, info


def run(seed=0, oof=None):
    df, feat_cols = common.load_aggregate()
    X = df[feat_cols].values.astype(np.float64)
    y = df["label"].values.astype(int)
    scen = df["scenario"].values
    topo = _load_topology()
    tr, te = common.scenario_split(scen, y, test_frac=0.3, seed=seed)

    std = common.Standardizer().fit(X[tr])
    Xs = std.transform(X)

    t0 = time.perf_counter()
    # all tiers are fit on training rows only
    tier1, tier3, info = fit_hrm(df, Xs, y, scen, topo, tr, seed=seed, oof=oof)
    train_t = time.perf_counter() - t0

    # Amortized time per node for the whole pipeline (tiers 1 to 3), not
    # single-sample latency. Timing tier 3 alone would understate the cost.
    t0 = time.perf_counter()
    p_local_inf = tier1.predict_proba(Xs)[:, 1]
    Z_inf = _build_Z(df, scen, topo, p_local_inf)
    score_all = tier3.predict_proba(Z_inf)[:, 1]
    amort_t = (time.perf_counter() - t0) / len(df) * 1e3

    score = score_all[te]
    pred = (score >= 0.5).astype(int)

    m = common.metrics(y[te], pred, score)
    m.update(train_time_s=train_t, amortized_cpu_ms_per_node=amort_t,
             n_test=int(te.sum()),
             tier3_fit="out_of_fold" if info["oof"] else "in_sample")
    return {"HRM": m}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--insample", action="store_true",
                    help="fit tier 3 on in-sample tier-1 scores "
                         "instead of out-of-fold scores")
    a = ap.parse_args()
    print(json.dumps(run(a.seed, oof=False if a.insample else None),
                     indent=2, default=float))
