"""Shared data loading, scenario-disjoint splitting and metrics."""
from __future__ import annotations
import os, json, time
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def load_aggregate():
    df = pd.read_csv(os.path.join(DATA, "node_features.csv"))
    # Identifiers and scenario configuration (mobility, drop_ratio, n_nodes)
    # are excluded: configuration is not observable telemetry.
    feat_cols = [c for c in df.columns
                 if c not in ("scenario", "node", "label", "mobility",
                              "drop_ratio", "n_nodes")]
    return df, feat_cols


def load_sequences():
    z = np.load(os.path.join(DATA, "sequences.npz"), allow_pickle=True)
    return dict(seq=z["seq"], mask=z["mask"], labels=z["labels"],
                scenario=z["scenario"], feat_keys=list(z["feat_keys"]))


def scenario_split(scenarios, labels, test_frac=0.3, seed=0):
    """Split by scenario id, so train and test nodes never share a scenario."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(scenarios)
    rng.shuffle(uniq)
    n_test = int(len(uniq) * test_frac)
    test_scen = set(uniq[:n_test].tolist())
    test_mask = np.array([s in test_scen for s in scenarios])
    return ~test_mask, test_mask


def metrics(y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    out = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
           "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    if y_score is not None:
        out["auc"] = roc_auc(y_true, np.asarray(y_score))
    return out


def roc_auc(y_true, y_score):
    # sklearn averages ranks for tied scores (matters for discrete scorers)
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(y_score)))


class Standardizer:
    def fit(self, X):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-8
        return self
    def transform(self, X):
        return (X - self.mu) / self.sd
    def fit_transform(self, X):
        return self.fit(X).transform(X)
