"""Raw per-split results and split identifiers for every model comparison.

For each track (see main) and split seed, writes the test scenario ids and
each model's metrics to ../results/per_split_metrics.json. The HRM is run
twice per split, with tier 3 fit on out-of-fold (HRM-oof) or in-sample
(HRM-insample) tier-1 scores. Timing is left out so the file is deterministic.
Usage (from models/): python3 per_split_metrics.py [--ns3-ten-extra]
"""
from __future__ import annotations
import argparse, json, os, platform, sys, time
import numpy as np

import common
import baseline_svm
import hrm
import run_experiments
import extra_experiments

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
KEEP = ("accuracy", "precision", "recall", "f1", "auc",
        "tp", "fp", "fn", "tn", "n_test")
MODELS = ("SVM-RBF", "RandomForest", "HRM-oof", "HRM-insample", "SLM-attn")


def _use(dataset):
    """Point every loader at one dataset folder (data or data_ns3_grid)."""
    common.DATA = os.path.join(ROOT, dataset)
    hrm.DATA = common.DATA


def _keep(m):
    return {k: (float(m[k]) if isinstance(m[k], float) else int(m[k]))
            for k in KEEP}


def one_split(seed):
    """All models on one scenario-disjoint split, plus the split's ids."""
    df, _ = common.load_aggregate()
    scen = df["scenario"].values
    y = df["label"].values.astype(int)
    tr, te = common.scenario_split(scen, y, test_frac=0.3, seed=seed)
    test_scen = np.unique(scen[te])
    train_scen = np.unique(scen[tr])
    # the sequence classifier splits on its own scenario array; both splits
    # must select the same scenarios or the models would not be paired
    d = common.load_sequences()
    _, tes = common.scenario_split(d["scenario"], d["labels"], 0.3, seed)
    assert np.array_equal(np.unique(d["scenario"][tes]), test_scen)
    assert not set(test_scen.tolist()) & set(train_scen.tolist())

    models = {}
    r = baseline_svm.run(seed)
    models["SVM-RBF"] = _keep(r["SVM-RBF"])
    models["RandomForest"] = _keep(r["RandomForest"])
    models["HRM-oof"] = _keep(hrm.run(seed, oof=True)["HRM"])
    models["HRM-insample"] = _keep(hrm.run(seed, oof=False)["HRM"])
    models["SLM-attn"] = _keep(run_experiments.run_slm(seed)["SLM-attn"])
    for m in models.values():
        assert m["n_test"] == int(te.sum())
    return {
        "seed": int(seed),
        "n_train_scenarios": int(len(train_scen)),
        "n_test_scenarios": int(len(test_scen)),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "test_scenarios": [int(s) for s in test_scen],
        "hrm_oof_folds_train_scenarios":
            [[int(s) for s in f] for f in hrm.oof_folds(scen, tr, seed)],
        "models": models,
    }


def run_track(dataset, seeds, note):
    _use(dataset)
    t0 = time.perf_counter()
    splits = []
    for seed in seeds:
        splits.append(one_split(seed))
        f1 = {k: round(v["f1"], 4) for k, v in splits[-1]["models"].items()}
        print(f"  [{dataset} seed {seed}] F1 {f1}", flush=True)
    return {"dataset": dataset + "/", "seeds": [int(s) for s in seeds],
            "note": note, "splits": splits,
            "wall_time_s": round(time.perf_counter() - t0, 1)}


def _meta():
    import sklearn, pandas
    return {
        "generated_pacific": time.strftime("%Y-%m-%d %H:%M %Z",
                                           time.localtime()),
        "python": sys.version.split()[0], "machine": platform.machine(),
        "numpy": np.__version__, "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "split_rule": "common.scenario_split(scenarios, labels, "
                      "test_frac=0.3, seed): default_rng(seed) shuffles the "
                      "sorted scenario ids, the first int(0.3 * n) are the "
                      "test scenarios",
        "hrm_oof_rule": f"hrm.oof_folds: default_rng(seed) shuffles the "
                        f"sorted TRAINING scenario ids, dealt round-robin "
                        f"into {hrm.N_OOF_FOLDS} folds; each fold is scored "
                        f"by a tier-1 model fit on the other folds",
        "threshold": "0.5 on the model score (SVM: sign of the decision "
                     "function, via predict)",
        "std_convention": "summary std elsewhere is np.std (ddof 0)",
        "models": list(MODELS),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns3-ten-extra", action="store_true",
                    help="run ONLY the extra ten splits on the NS-3 grid "
                         "(not part of the paper's protocol)")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    if a.ns3_ten_extra:
        out = {"meta": _meta(), "tracks": {}}
        out["meta"]["label"] = ("EXTRA, not part of the paper's protocol: "
                                "ten splits on the NS-3 grid")
        out["tracks"]["ns3_grid_ten_splits"] = run_track(
            "data_ns3_grid", range(10),
            "Supplementary: seeds 0-9 on the NS-3 main grid, same "
            "protocol as Table III otherwise (Table III uses seeds 0-2)")
        path = os.path.join(RESULTS,
                            "per_split_metrics_ns3_ten_splits_extra.json")
    else:
        out = {"meta": _meta(), "tracks": {}}
        out["tracks"]["bootstrap_table2"] = run_track(
            "data", (0, 1, 2), "Table II: bootstrap, seeds 0-2")
        out["tracks"]["ns3_grid_table3"] = run_track(
            "data_ns3_grid", (0, 1, 2), "Table III: NS-3 main grid, seeds 0-2")
        out["tracks"]["bootstrap_ten_splits"] = run_track(
            "data", range(10),
            "Sec. VI-D significance: bootstrap, seeds 0-9 (seeds 0-2 are the "
            "same splits as Table II)")
        _use("data")
        f1s = extra_experiments.ablation_raw(seeds=(0, 1, 2))
        out["ablation_attention"] = {
            "dataset": "data/", "seeds": [0, 1, 2],
            "note": "per-seed F1, attention vs attention disabled "
                    "(extra_experiments.py 6)",
            "f1": {k: [float(v) for v in vs] for k, vs in f1s.items()}}
        path = os.path.join(RESULTS, "per_split_metrics.json")

    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
