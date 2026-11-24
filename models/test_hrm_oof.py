"""Independent check of the out-of-fold HRM fit.

Re-derives the out-of-fold tier-3 training without calling hrm.fit_hrm and
confirms that hrm.fit_hrm gives the same tier-3 model. Also confirms that the
folds partition the training scenarios, that no test scenario appears in any
fold, and that tier 1 is the same in the out-of-fold and in-sample modes.
Usage (from models/): python3 test_hrm_oof.py [seed]. Exit code 0 on success.
"""
from __future__ import annotations
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import common
import hrm


def main(seed=0):
    df, feat = common.load_aggregate()
    X = df[feat].values.astype(np.float64)
    y = df["label"].values.astype(int)
    scen = df["scenario"].values
    topo = hrm._load_topology()
    tr, te = common.scenario_split(scen, y, 0.3, seed)
    Xs = common.Standardizer().fit(X[tr]).transform(X)

    t1, t3, info = hrm.fit_hrm(df, Xs, y, scen, topo, tr, seed, oof=True)
    t1_in, _, _ = hrm.fit_hrm(df, Xs, y, scen, topo, tr, seed, oof=False)

    train_scen = set(scen[tr].tolist())
    test_scen = set(scen[te].tolist())
    folds = info["oof_folds"]
    assert len(folds) == hrm.N_OOF_FOLDS
    assert sorted(s for f in folds for s in f) == sorted(train_scen), \
        "folds must partition the training scenarios"

    Z_full = hrm._build_Z(df, scen, topo, t1.predict_proba(Xs)[:, 1])
    Z_oof = np.zeros_like(Z_full)
    for held in folds:
        others = train_scen - set(held)
        assert not (set(held) | others) & test_scen, "test scenario leaked"
        fit_mask, ho_mask = np.isin(scen, list(others)), np.isin(scen, held)
        g = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                       random_state=seed)
        g.fit(Xs[fit_mask], y[fit_mask])
        Zh = hrm._build_Z(df[ho_mask], scen[ho_mask], topo,
                          g.predict_proba(Xs[ho_mask])[:, 1])
        Z_oof[ho_mask] = Zh
        # the degree column is normalized by the dataset-wide max degree,
        # which does not depend on any score
        Z_oof[ho_mask, 4] = Z_full[ho_mask, 4]
    ref = LogisticRegression(max_iter=1000, class_weight="balanced")
    ref.fit(Z_oof[tr], y[tr])

    d_coef = float(np.abs(ref.coef_ - t3.coef_).max())
    d_int = float(abs(ref.intercept_[0] - t3.intercept_[0]))
    d_score = float(np.abs(ref.predict_proba(Z_full[te])[:, 1]
                           - t3.predict_proba(Z_full[te])[:, 1]).max())
    same_t1 = bool(np.array_equal(t1.predict_proba(Xs), t1_in.predict_proba(Xs)))
    print(f"seed {seed}: tier-3 coef max |diff| {d_coef:.1e}, intercept "
          f"|diff| {d_int:.1e}, test score max |diff| {d_score:.1e}, final "
          f"tier 1 identical across modes: {same_t1}")
    ok = d_coef < 1e-9 and d_int < 1e-9 and d_score < 1e-9 and same_t1
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 0))
