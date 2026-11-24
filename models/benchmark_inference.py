"""Controlled CPU inference benchmark for the four detectors.

BLAS/OpenMP thread pools are pinned before numpy is imported (1 thread by
default). After warm-up runs, medians of repeated time.perf_counter timings
are reported for batch size one and for the full test set. HRM tier 2 needs a
scenario's neighborhood, so its batch-1 unit is one scenario, not one node.
Usage: python3 benchmark_inference.py [--threads 4] [--repeats 50] [--warmup 10]
Output: ../results/benchmark_inference.json and a console table."""
from __future__ import annotations
import argparse, os, sys

# thread settings must be in place before numpy and sklearn are imported
_ap = argparse.ArgumentParser()
_ap.add_argument("--threads", type=int, default=1)
_ap.add_argument("--repeats", type=int, default=30)
_ap.add_argument("--warmup", type=int, default=5)
_ap.add_argument("--batch1-samples", type=int, default=100,
                 help="how many distinct single nodes to time per repeat")
ARGS = _ap.parse_args()
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = str(ARGS.threads)

import json, platform, time
import numpy as np

import common
from slm_numpy import SelfAttnClassifier

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def _stats(times_ms):
    a = np.asarray(times_ms)
    return {"median_ms": float(np.median(a)),
            "p25_ms": float(np.percentile(a, 25)),
            "p75_ms": float(np.percentile(a, 75)),
            "mean_ms": float(a.mean()), "std_ms": float(a.std()),
            "n_measurements": int(a.size)}


def _bench(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def bench_flat_model(name, fit, predict, X, y, tr, te, rng):
    """SVM / RF: aggregate-feature models with per-row prediction."""
    model = fit(X[tr], y[tr])
    Xte = X[te]
    # batch-1 latency: distinct single rows, one predict call each
    idx = rng.choice(len(Xte), size=min(ARGS.batch1_samples, len(Xte)),
                    replace=False)
    b1 = []
    for _ in range(ARGS.repeats):
        for i in idx[:max(1, len(idx) // ARGS.repeats)]:
            row = Xte[i:i + 1]
            t0 = time.perf_counter()
            predict(model, row)
            b1.append((time.perf_counter() - t0) * 1e3)
    # the loop above serves as warm-up; the batched run gets its own in _bench
    bt = _bench(lambda: predict(model, Xte), ARGS.warmup, ARGS.repeats)
    batched = _stats(bt)
    batched["amortized_ms_per_node"] = batched["median_ms"] / len(Xte)
    batched["nodes_per_second"] = 1e3 * len(Xte) / batched["median_ms"]
    batched["batch_size"] = int(len(Xte))
    return {"batch1_latency": _stats(b1), "batched": batched}


def main():
    rng = np.random.default_rng(0)
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    df, feat_cols = common.load_aggregate()
    X = df[feat_cols].values
    y = df["label"].values.astype(int)
    tr, te = common.scenario_split(df["scenario"].values, y, 0.3, 0)
    std = common.Standardizer().fit(X[tr])
    Xs = std.transform(X)

    d = common.load_sequences()
    seq, mask, ys, scen = d["seq"], d["mask"], d["labels"], d["scenario"]
    trs, tes = common.scenario_split(scen, ys, 0.3, 0)
    valid = mask[trs].astype(bool)
    flat = seq[trs][valid]
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    seqn = ((seq - mu) / sd) * mask[:, :, None]

    results = {"config": {
        "threads": ARGS.threads, "repeats": ARGS.repeats,
        "warmup": ARGS.warmup, "timer": "time.perf_counter",
        "batch1_samples": ARGS.batch1_samples},
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "machine": platform.machine(),
            "cpu_count_logical": os.cpu_count(),
            "python": sys.version.split()[0],
            "numpy": np.__version__},
        "models": {}}
    try:
        import sklearn
        results["hardware"]["sklearn"] = sklearn.__version__
    except Exception:
        pass

    print(f"[bench] threads={ARGS.threads} repeats={ARGS.repeats} "
          f"warmup={ARGS.warmup}")

    results["models"]["SVM-RBF"] = bench_flat_model(
        "SVM-RBF",
        lambda Xa, ya: SVC(kernel="rbf", C=4.0, gamma="scale",
                           class_weight="balanced").fit(Xa, ya),
        lambda m, Xq: m.decision_function(Xq),
        Xs, y, tr, te, rng)
    print("  SVM-RBF done")

    results["models"]["RandomForest"] = bench_flat_model(
        "RandomForest",
        lambda Xa, ya: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=0,
            n_jobs=1).fit(Xa, ya),
        lambda m, Xq: m.predict_proba(Xq)[:, 1],
        Xs, y, tr, te, rng)
    print("  RandomForest done")

    # sequence classifier (attention)
    net = SelfAttnClassifier(seq.shape[2], seq.shape[1], d_model=24, d_ff=32,
                             seed=0)
    net.fit(seqn[trs], mask[trs], ys[trs].astype(float), epochs=10, seed=0)
    Xq, Mq = seqn[tes], mask[tes]
    idx = rng.choice(len(Xq), size=min(ARGS.batch1_samples, len(Xq)),
                     replace=False)
    b1 = []
    for _ in range(ARGS.warmup):
        net.predict_proba(Xq[:1], Mq[:1])
    for _ in range(ARGS.repeats):
        for i in idx[:max(1, len(idx) // ARGS.repeats)]:
            t0 = time.perf_counter()
            net.predict_proba(Xq[i:i + 1], Mq[i:i + 1])
            b1.append((time.perf_counter() - t0) * 1e3)
    bt = _bench(lambda: net.predict_proba(Xq, Mq), ARGS.warmup, ARGS.repeats)
    batched = _stats(bt)
    batched["amortized_ms_per_node"] = batched["median_ms"] / len(Xq)
    batched["nodes_per_second"] = 1e3 * len(Xq) / batched["median_ms"]
    batched["batch_size"] = int(len(Xq))
    results["models"]["SLM-attn"] = {"batch1_latency": _stats(b1),
                                     "batched": batched}
    print("  SLM-attn done")

    # HRM: full 3-tier pipeline, smallest unit is one scenario
    try:
        import hrm as hrm_mod
        topo = hrm_mod._load_topology()
        scen_col = df["scenario"].values
        # Same fit as hrm.run(). The tier-3 training mode changes only its
        # coefficients, not the inference path timed below.
        tier1, tier3, _ = hrm_mod.fit_hrm(df, Xs, y, scen_col, topo, tr,
                                          seed=0)
        test_scens = np.unique(scen_col[te])
        one = rng.choice(test_scens, size=min(20, len(test_scens)),
                         replace=False)

        def _one_scenario(s):
            m = scen_col == s
            sub = df[m]
            p1 = tier1.predict_proba(Xs[m])[:, 1]
            Zs = hrm_mod._build_Z(sub, scen_col[m], topo, p1)
            return tier3.predict_proba(Zs)[:, 1]

        b1, per_node = [], []
        for _ in range(ARGS.warmup):
            _one_scenario(one[0])
        for _ in range(ARGS.repeats):
            for s in one[:max(1, len(one) // ARGS.repeats)]:
                n_in_s = int((scen_col == s).sum())
                t0 = time.perf_counter()
                _one_scenario(s)
                dt = (time.perf_counter() - t0) * 1e3
                b1.append(dt); per_node.append(dt / n_in_s)

        def _full():
            p1 = tier1.predict_proba(Xs)[:, 1]
            Zf = hrm_mod._build_Z(df, scen_col, topo, p1)
            return tier3.predict_proba(Zf)[:, 1]

        bt = _bench(_full, ARGS.warmup, ARGS.repeats)
        batched = _stats(bt)
        batched["amortized_ms_per_node"] = batched["median_ms"] / len(df)
        batched["nodes_per_second"] = 1e3 * len(df) / batched["median_ms"]
        batched["batch_size"] = int(len(df))
        results["models"]["HRM"] = {
            "batch1_latency": _stats(b1),
            "batch1_unit": "one scenario (tier-2 needs the neighborhood)",
            "batch1_ms_per_node_median": float(np.median(per_node)),
            "batched": batched}
        print("  HRM done")
    except Exception as e:  # keep the benchmark usable if HRM internals move
        results["models"]["HRM"] = {"error": str(e)}
        print(f"  HRM skipped: {e}")

    out = os.path.join(RESULTS, "benchmark_inference.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n==== CONTROLLED CPU BENCHMARK (threads={ARGS.threads}) ====")
    print(f"{'model':14s}{'batch-1 median':>18s}{'batched amortized':>20s}"
          f"{'throughput':>16s}")
    for mdl, r in results["models"].items():
        if "error" in r:
            print(f"{mdl:14s}  error: {r['error']}"); continue
        b1 = r["batch1_latency"]["median_ms"]
        am = r["batched"]["amortized_ms_per_node"]
        tp = r["batched"]["nodes_per_second"]
        unit = " /scen" if "batch1_unit" in r else " /node"
        print(f"{mdl:14s}{b1:14.4f} ms{unit}{am:16.5f} ms{tp:13.0f} n/s")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
