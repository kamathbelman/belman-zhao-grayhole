"""Comparison figures, written to ../results/figures.

  fig1_metric_comparison.png   grouped bars: accuracy/precision/recall/F1
  fig2_latency.png             amortized CPU inference time per node (log scale)
  fig3_f1_vs_dropratio.png     detection F1 vs gray-hole drop ratio
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
from slm_numpy import SelfAttnClassifier

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

COLORS = {"SVM-RBF": "#5B6C8F", "RandomForest": "#8FA07A",
          "HRM": "#C08552", "SLM-attn": "#8C4A5B"}


def fig_metric_comparison(summary):
    models = ["SVM-RBF", "RandomForest", "HRM", "SLM-attn"]
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics)); w = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, mdl in enumerate(models):
        means = [summary[mdl][k]["mean"] for k in metrics]
        errs = [summary[mdl][k]["std"] for k in metrics]
        ax.bar(x + (i - 1.5) * w, means, w, yerr=errs, capsize=3,
               label=mdl, color=COLORS[mdl])
    ax.set_xticks(x); ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Gray-hole detection: model comparison (bootstrap dataset, 3 seeds)")
    ax.legend(ncol=4, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_metric_comparison.png"), dpi=150)
    plt.close(fig)


def fig_latency(summary):
    models = ["SVM-RBF", "RandomForest", "HRM", "SLM-attn"]

    # Use the controlled benchmark (benchmark_inference.py) when available,
    # otherwise the amortized times from run_experiments.
    bench_path = os.path.join(RES, "benchmark_inference.json")
    bench = None
    if os.path.exists(bench_path):
        with open(bench_path) as f:
            b = json.load(f)
        if all(m in b.get("models", {}) and "batched" in b["models"][m]
               for m in models):
            bench = {m: b["models"][m]["batched"]["amortized_ms_per_node"]
                     for m in models}

    def _t(m):  # summary fallback, accepts either key name
        s = summary[m]
        return s.get("amortized_cpu_ms_per_node", s.get("infer_ms_per_sample"))

    vals = [max(1e-4, bench[m] if bench else _t(m)["mean"]) for m in models]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(models, vals, color=[COLORS[m] for m in models])
    ax.set_yscale("log")
    ax.set_ylabel("Amortized CPU inference time (ms / node, log scale)")
    ax.set_title("Amortized CPU inference time per node (batched)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_latency.png"), dpi=150)
    plt.close(fig)


def _per_drop_f1(seed):
    """Train the SVM and the sequence classifier on one scenario-disjoint
    split; return F1 per drop ratio. drop_ratio is used only to group test
    scenarios, never as a model feature (common.load_aggregate excludes it)."""
    from sklearn.svm import SVC
    df, feat_cols = common.load_aggregate()
    y = df["label"].values.astype(int)
    scen = df["scenario"].values
    drop = df["drop_ratio"].values
    tr, te = common.scenario_split(scen, y, test_frac=0.3, seed=seed)

    X = df[feat_cols].values.astype(float)
    std = common.Standardizer().fit(X[tr])
    svm = SVC(kernel="rbf", C=4.0, gamma="scale", class_weight="balanced")
    svm.fit(std.transform(X[tr]), y[tr])
    svm_pred = svm.predict(std.transform(X[te]))

    d = common.load_sequences()
    seq, mask, ys, scn = d["seq"], d["mask"], d["labels"], d["scenario"]
    trs, tes = common.scenario_split(scn, ys, test_frac=0.3, seed=seed)
    valid = mask[trs].astype(bool); flat = seq[trs][valid]
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    seqn = ((seq - mu) / sd) * mask[:, :, None]
    net = SelfAttnClassifier(seq.shape[2], seq.shape[1], 24, 32, seed=seed)
    net.fit(seqn[trs], mask[trs], ys[trs].astype(float), epochs=40, seed=seed)
    slm_pred = (net.predict_proba(seqn[tes], mask[tes]) >= 0.5).astype(int)

    scen_drop = {int(s): float(df[df.scenario == s]["drop_ratio"].iloc[0])
                 for s in np.unique(scen)}
    slm_drop = np.array([scen_drop[int(s)] for s in scn[tes]])
    dte = drop[te]; yte = y[te]
    drops = sorted(np.unique(drop))
    svm_f1, slm_f1 = [], []
    for dr in drops:
        mA = dte == dr
        svm_f1.append(common.metrics(yte[mA], svm_pred[mA])["f1"] if mA.sum() else np.nan)
        mB = slm_drop == dr
        slm_f1.append(common.metrics(ys[tes][mB], slm_pred[mB])["f1"] if mB.sum() else np.nan)
    return drops, svm_f1, slm_f1


def fig_f1_vs_dropratio(seeds=(0, 1, 2)):
    """F1 vs attack drop ratio, mean +/- std over several scenario-disjoint
    splits (a single split is too noisy)."""
    all_svm, all_slm, drops = [], [], None
    for seed in seeds:
        drops, svm_f1, slm_f1 = _per_drop_f1(seed)
        all_svm.append(svm_f1); all_slm.append(slm_f1)
    svm_m = np.nanmean(all_svm, 0); svm_s = np.nanstd(all_svm, 0)
    slm_m = np.nanmean(all_slm, 0); slm_s = np.nanstd(all_slm, 0)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.errorbar(drops, svm_m, yerr=svm_s, fmt="-o", capsize=3,
                color=COLORS["SVM-RBF"], label="SVM-RBF")
    ax.errorbar(drops, slm_m, yerr=slm_s, fmt="-s", capsize=3,
                color=COLORS["SLM-attn"], label="Sequence classifier")
    ax.set_xlabel("Gray-hole drop ratio (attack intensity)")
    ax.set_ylabel("F1 score")
    ax.set_title(f"Detection difficulty vs attack intensity "
                 f"(mean ± std, {len(seeds)} splits)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_f1_vs_dropratio.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    with open(os.path.join(RES, "results_summary.json")) as f:
        summary = json.load(f)
    fig_metric_comparison(summary)
    fig_latency(summary)
    fig_f1_vs_dropratio()
    print("figures written to", FIG)
