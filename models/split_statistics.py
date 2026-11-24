"""Statistics computed from the saved per-split values. Nothing is retrained.

Reads results/per_split_metrics.json (and the extra NS-3 ten-split file if
present) and writes results/split_statistics.json. Per track: mean and std
(np.std, ddof 0) per model, per-split paired differences, and the ordinary
and Nadeau-Bengio corrected paired t-tests with 95% intervals.
Usage (from models/): python3 split_statistics.py
"""
from __future__ import annotations
import json, math, os
import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
METRICS = ("accuracy", "precision", "recall", "f1", "auc")
COMPARISONS = (
    ("SLM-attn", "SVM-RBF"), ("SLM-attn", "RandomForest"),
    ("SLM-attn", "HRM-oof"), ("SLM-attn", "HRM-insample"),
    ("HRM-oof", "RandomForest"), ("HRM-oof", "SVM-RBF"),
    ("HRM-oof", "HRM-insample"),
)


def paired_stats(d, n_test, n_train):
    """Ordinary and Nadeau-Bengio corrected paired t for differences d.

    The correction accounts for the overlap between random splits of the
    same scenarios (Nadeau and Bengio, Machine Learning 52(3), 2003):
        t = mean(d) / sqrt((1/k + n_test/n_train) * var(d)),  df = k - 1
    with var(d) the sample variance (ddof 1), k the number of splits, and
    n_test / n_train counted in scenarios.
    """
    d = np.asarray(d, dtype=float)
    k = len(d)
    mean, var = float(d.mean()), float(d.var(ddof=1))
    df = k - 1
    tcrit = float(stats.t.ppf(0.975, df))
    out = {"k": k, "df": df, "diffs": d.tolist(), "mean_diff": mean,
           "sd_diff_ddof1": math.sqrt(var),
           "a_wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
           "n_test_scenarios": n_test, "n_train_scenarios": n_train}
    for name, factor in (("ordinary", 1.0 / k),
                         ("nadeau_bengio", 1.0 / k + n_test / n_train)):
        se = math.sqrt(factor * var)
        if se > 0:
            t = mean / se
            p = float(2 * stats.t.sf(abs(t), df))
        else:
            t, p = float("nan"), float("nan")
        out[name] = {"variance_factor": factor, "se": se, "t": t,
                     "p_two_sided": p,
                     "ci95": [mean - tcrit * se, mean + tcrit * se]}
    return out


def track_stats(track):
    splits = track["splits"]
    models = list(splits[0]["models"])
    n_te = splits[0]["n_test_scenarios"]
    n_tr = splits[0]["n_train_scenarios"]
    assert all(s["n_test_scenarios"] == n_te and s["n_train_scenarios"] == n_tr
               for s in splits)
    summary = {}
    for m in models:
        summary[m] = {}
        for k in METRICS:
            v = np.array([s["models"][m][k] for s in splits], dtype=float)
            summary[m][k] = {"mean": float(v.mean()), "std": float(v.std()),
                             "per_split": v.tolist()}
    paired = {}
    for a, b in COMPARISONS:
        for k in ("f1", "auc"):
            d = (np.array(summary[a][k]["per_split"])
                 - np.array(summary[b][k]["per_split"]))
            paired[f"{a} minus {b} [{k}]"] = paired_stats(d, n_te, n_tr)
    return {"dataset": track["dataset"], "seeds": track["seeds"],
            "note": track["note"], "summary_mean_std_ddof0": summary,
            "paired": paired}


def _print_track(name, st):
    print(f"\n=== {name} ({st['dataset']}, seeds {st['seeds']}) ===")
    print(f"{'model':14s}" + "".join(f"{k:>16s}" for k in METRICS))
    for m, s in st["summary_mean_std_ddof0"].items():
        print(f"{m:14s}" + "".join(
            f"{s[k]['mean']:9.4f}±{s[k]['std']:.4f}" for k in METRICS))
    print(f"{'paired comparison (F1)':38s}{'mean d':>8s}{'t ord':>8s}"
          f"{'t NB':>8s}{'p NB':>9s}{'95% CI (NB)':>20s}{'wins':>6s}")
    for key, p in st["paired"].items():
        if not key.endswith("[f1]"):
            continue
        nb = p["nadeau_bengio"]
        print(f"{key[:-5]:38s}{p['mean_diff']:8.4f}{p['ordinary']['t']:8.2f}"
              f"{nb['t']:8.2f}{nb['p_two_sided']:9.4f}"
              f"   [{nb['ci95'][0]:.3f}, {nb['ci95'][1]:.3f}]"
              f"{p['a_wins']:4d}/{p['k']}")


def main():
    out = {"method": {
        "std": "np.std, ddof 0 (as in the paper's tables)",
        "ordinary_paired_t": "mean(d) / (sd(d, ddof 1) / sqrt(k)), df = k-1",
        "nadeau_bengio": "mean(d) / sqrt((1/k + n_test/n_train) * var(d, "
                         "ddof 1)), df = k-1, n counted in scenarios",
        "reference": "Nadeau and Bengio, Machine Learning 52(3):239-281, "
                     "2003"}, "tracks": {}}
    with open(os.path.join(RESULTS, "per_split_metrics.json")) as f:
        raw = json.load(f)
    for name, track in raw["tracks"].items():
        out["tracks"][name] = track_stats(track)
        _print_track(name, out["tracks"][name])

    ab = raw.get("ablation_attention")
    if ab:
        ten = raw["tracks"]["bootstrap_ten_splits"]["splits"][0]
        d = np.array(ab["f1"]["SLM-attn"]) - np.array(ab["f1"]["SLM-noattn"])
        out["ablation_attention"] = {
            "f1_per_seed": ab["f1"],
            "mean_std_ddof0": {k: {"mean": float(np.mean(v)),
                                   "std": float(np.std(v))}
                               for k, v in ab["f1"].items()},
            "paired_attn_minus_noattn": paired_stats(
                d, ten["n_test_scenarios"], ten["n_train_scenarios"])}
        p = out["ablation_attention"]["paired_attn_minus_noattn"]
        print(f"\n=== ablation (attention minus no attention, F1) ===\n"
              f"diffs {np.round(d, 4).tolist()} mean {p['mean_diff']:.4f} "
              f"t ord {p['ordinary']['t']:.2f} t NB "
              f"{p['nadeau_bengio']['t']:.2f} "
              f"CI NB [{p['nadeau_bengio']['ci95'][0]:.3f}, "
              f"{p['nadeau_bengio']['ci95'][1]:.3f}]")

    extra = os.path.join(RESULTS, "per_split_metrics_ns3_ten_splits_extra.json")
    if os.path.exists(extra):
        with open(extra) as f:
            raw_x = json.load(f)
        out["extra_not_in_paper_protocol"] = {}
        for name, track in raw_x["tracks"].items():
            out["extra_not_in_paper_protocol"][name] = track_stats(track)
            _print_track("Supplementary " + name,
                         out["extra_not_in_paper_protocol"][name])

    path = os.path.join(RESULTS, "split_statistics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
