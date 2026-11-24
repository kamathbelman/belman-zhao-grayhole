"""Train and evaluate all detectors (SVM-RBF, RandomForest, HRM, SLM-attn) on
the same scenario-disjoint split, so no scenario has nodes in both train and
test. Results are the mean and std over seeds, saved as JSON and CSV.

Timing is amortized CPU inference time per node (batch time / n), not
single-sample latency. See benchmark_inference.py for a controlled benchmark.
"""
from __future__ import annotations
import os, json, time
import numpy as np

import common
import baseline_svm
import hrm
from slm_numpy import SelfAttnClassifier

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def run_slm(seed=0, epochs=40):
    d = common.load_sequences()
    seq, mask, y, scen = d["seq"], d["mask"], d["labels"], d["scenario"]
    tr, te = common.scenario_split(scen, y, test_frac=0.3, seed=seed)
    # standardize with the valid (unpadded) training timesteps only
    valid = mask[tr].astype(bool)
    flat = seq[tr][valid]
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    seqn = ((seq - mu) / sd) * mask[:, :, None]     # keep pads at 0

    net = SelfAttnClassifier(seq.shape[2], seq.shape[1], d_model=24, d_ff=32, seed=seed)
    t0 = time.perf_counter()
    net.fit(seqn[tr], mask[tr], y[tr].astype(float),
            epochs=epochs, batch=128, lr=3e-3, l2=1e-4, seed=seed)
    train_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    score = net.predict_proba(seqn[te], mask[te])
    # amortized CPU time per node (batch time / n_test), not latency
    amort_t = (time.perf_counter() - t0) / max(1, te.sum()) * 1e3
    pred = (score >= 0.5).astype(int)
    m = common.metrics(y[te], pred, score)
    m.update(train_time_s=train_t, amortized_cpu_ms_per_node=amort_t,
             n_test=int(te.sum()))
    return {"SLM-attn": m}


def aggregate_over_seeds(seeds=(0, 1, 2)):
    runs = {}
    for seed in seeds:
        r = {}
        r.update(baseline_svm.run(seed))
        r.update(hrm.run(seed))
        r.update(run_slm(seed))
        for model, m in r.items():
            runs.setdefault(model, []).append(m)
        print(f"[seed {seed}] done")

    summary = {}
    keys = ["accuracy", "precision", "recall", "f1", "auc",
            "amortized_cpu_ms_per_node", "train_time_s"]
    for model, ms in runs.items():
        summary[model] = {}
        for k in keys:
            vals = np.array([mm[k] for mm in ms], dtype=float)
            summary[model][k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return summary, runs


def main():
    summary, runs = aggregate_over_seeds(seeds=(0, 1, 2))
    with open(os.path.join(RESULTS, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    import csv
    order = ["SVM-RBF", "RandomForest", "HRM", "SLM-attn"]
    with open(os.path.join(RESULTS, "results_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "precision", "recall", "f1", "auc",
                    "amortized_cpu_ms/node"])
        for mdl in order:
            s = summary[mdl]
            w.writerow([mdl] +
                       [f"{s[k]['mean']:.3f}±{s[k]['std']:.3f}"
                        for k in ["accuracy", "precision", "recall", "f1", "auc"]] +
                       [f"{s['amortized_cpu_ms_per_node']['mean']:.3f}"])
    print("\n==== RESULTS (mean over seeds) ====")
    print(f"{'model':14s}{'acc':>8s}{'prec':>8s}{'rec':>8s}{'f1':>8s}{'auc':>8s}"
          f"{'am.ms/node':>12s}")
    for mdl in order:
        s = summary[mdl]
        print(f"{mdl:14s}"
              f"{s['accuracy']['mean']:8.3f}{s['precision']['mean']:8.3f}"
              f"{s['recall']['mean']:8.3f}{s['f1']['mean']:8.3f}"
              f"{s['auc']['mean']:8.3f}"
              f"{s['amortized_cpu_ms_per_node']['mean']:12.3f}")
    return summary


if __name__ == "__main__":
    main()
