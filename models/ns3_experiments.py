"""Train and evaluate all four detectors on the NS-3 main grid (Table III).

Uses data_ns3_grid (525 scenarios; the 15 legacy-mirror scenarios are
excluded) with the same scenario-disjoint protocol as the bootstrap
experiments, seeds 0-2. Writes results/ns3_full_results.json.

Usage: python3 ns3_experiments.py
"""
import json
import os

import numpy as np

import common

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
common.DATA = os.path.join(ROOT, "data_ns3_grid")
import hrm  # noqa: E402
hrm.DATA = common.DATA
import baseline_svm  # noqa: E402
import run_experiments  # noqa: E402

METRICS = ("accuracy", "precision", "recall", "f1", "auc")


def main(seeds=(0, 1, 2)):
    runs = {}
    for seed in seeds:
        r = {}
        r.update(baseline_svm.run(seed))
        r.update(hrm.run(seed))
        r.update(run_experiments.run_slm(seed))
        for k, m in r.items():
            runs.setdefault(k, []).append(m)
        print(f"seed {seed} done")
    out = {}
    for mdl, ms in runs.items():
        out[mdl] = {k: {"mean": float(np.mean([m[k] for m in ms])),
                        "std": float(np.std([m[k] for m in ms]))}
                    for k in METRICS}
        s = out[mdl]
        print(f"{mdl:14s} acc={s['accuracy']['mean']:.3f} "
              f"f1={s['f1']['mean']:.3f}+/-{s['f1']['std']:.3f} "
              f"auc={s['auc']['mean']:.3f}")
    path = os.path.join(ROOT, "results", "ns3_full_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/ns3_full_results.json")


if __name__ == "__main__":
    main()
