"""Additional experiments, selected by number on the command line.

1 transfer: train on two mobility models, test on the third. 2 noisy_obs:
Gaussian noise on test telemetry. 3 seeds10: 10 splits, paired t-test against
each baseline. 4 attention_maps: attention received per window. 5 adaptive:
pulsed on/off gray-holes. 6 ablation: attention disabled, mean pooling only.
Outputs: results/extra_results.json and results/figures/fig4_attention.png,
fig5_noisy_obs.png. Run: python3 extra_experiments.py [all|1..6]"""
from __future__ import annotations
import os, sys, json, math
import numpy as np

import common
from slm_numpy import SelfAttnClassifier, softmax
from sklearn.svm import SVC

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(RESULTS, "extra_results.json")


def _save(key, value):
    data = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            data = json.load(f)
    data[key] = value
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[extra] saved '{key}'")


def _load_all():
    df, feat_cols = common.load_aggregate()
    d = common.load_sequences()
    scen_mob = {int(s): df[df.scenario == s]["mobility"].iloc[0]
                for s in np.unique(df["scenario"])}
    return df, feat_cols, d, scen_mob


def _fit_svm(df, feat_cols, tr, seed=0):
    X = df[feat_cols].values.astype(float)
    y = df["label"].values.astype(int)
    std = common.Standardizer().fit(X[tr])
    svm = SVC(kernel="rbf", C=4.0, gamma="scale", class_weight="balanced")
    svm.fit(std.transform(X[tr]), y[tr])
    return svm, std, X, y


def _fit_slm(d, trs, seed=0, epochs=40):
    seq, mask, ys = d["seq"], d["mask"], d["labels"]
    valid = mask[trs].astype(bool)
    flat = seq[trs][valid]
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    seqn = ((seq - mu) / sd) * mask[:, :, None]
    net = SelfAttnClassifier(seq.shape[2], seq.shape[1], 24, 32, seed=seed)
    net.fit(seqn[trs], mask[trs], ys[trs].astype(float), epochs=epochs, seed=seed)
    return net, seqn


def transfer():
    df, feat_cols, d, scen_mob = _load_all()
    y = df["label"].values.astype(int)
    out = {}
    # mobility labels are read from the data: the bootstrap and ns-3 sets differ
    for held in sorted(df["mobility"].unique()):
        tr = (df["mobility"] != held).values
        te = ~tr
        svm, std, X, _ = _fit_svm(df, feat_cols, tr)
        pred = svm.predict(std.transform(X[te]))
        m_svm = common.metrics(y[te], pred)
        mobs = np.array([scen_mob[int(s)] for s in d["scenario"]])
        trs, tes = mobs != held, mobs == held
        net, seqn = _fit_slm(d, trs)
        sc = net.predict_proba(seqn[tes], d["mask"][tes])
        m_slm = common.metrics(d["labels"][tes], (sc >= 0.5).astype(int), sc)
        out[held] = {"SVM_f1": m_svm["f1"], "SLM_f1": m_slm["f1"],
                     "SLM_auc": m_slm["auc"]}
        print(f"  held-out {held}: SVM F1={m_svm['f1']:.3f} "
              f"SLM F1={m_slm['f1']:.3f} AUC={m_slm['auc']:.3f}")
    _save("transfer_mobility", out)


def noisy_obs(seed=0):
    df, feat_cols, d, _ = _load_all()
    y = df["label"].values.astype(int)
    tr, te = common.scenario_split(df["scenario"].values, y, 0.3, seed)
    svm, std, X, _ = _fit_svm(df, feat_cols, tr, seed)
    trs, tes = common.scenario_split(d["scenario"], d["labels"], 0.3, seed)
    net, seqn = _fit_slm(d, trs, seed)
    rng = np.random.default_rng(seed)
    Xsd = X[tr].std(0) + 1e-9
    Ssd = seqn[trs][d["mask"][trs].astype(bool)].std(0) + 1e-9
    out = {}
    for eps in (0.0, 0.1, 0.25, 0.5, 1.0):
        Xn = X[te] + rng.normal(0, eps, X[te].shape) * Xsd
        m_svm = common.metrics(y[te], svm.predict(std.transform(Xn)))
        Sn = seqn[tes] + (rng.normal(0, eps, seqn[tes].shape)
                          * Ssd) * d["mask"][tes][:, :, None]
        sc = net.predict_proba(Sn, d["mask"][tes])
        m_slm = common.metrics(d["labels"][tes], (sc >= 0.5).astype(int))
        out[str(eps)] = {"SVM_f1": m_svm["f1"], "SLM_f1": m_slm["f1"]}
        print(f"  eps={eps}: SVM F1={m_svm['f1']:.3f} SLM F1={m_slm['f1']:.3f}")
    _save("noisy_observability", out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epss = [float(k) for k in out]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(epss, [out[str(e)]["SVM_f1"] for e in epss], "-o",
            color="#5B6C8F", label="SVM-RBF")
    ax.plot(epss, [out[str(e)]["SLM_f1"] for e in epss], "-s",
            color="#8C4A5B", label="Sequence classifier")
    ax.set_xlabel("Test-time telemetry noise (fraction of feature std)")
    ax.set_ylabel("F1 score"); ax.set_ylim(0, 1.0)
    ax.set_title("Sensitivity to additive Gaussian telemetry noise")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_noisy_obs.png"), dpi=150)
    plt.close(fig)


def seeds10(seed_lo=0, seed_hi=10):
    import baseline_svm, hrm, run_experiments
    # Raw per-seed F1s. Starts from an empty list unless EXTRA_RESUME=1, so
    # runs on different data are not pooled by accident.
    raw = {}
    if os.environ.get("EXTRA_RESUME") == "1" and os.path.exists(OUT):
        with open(OUT) as f:
            raw = json.load(f).get("seeds10_raw", {})
    runs = {k: list(v) for k, v in raw.items()}
    for seed in range(seed_lo, seed_hi):
        r = {}
        r.update(baseline_svm.run(seed))
        r.update(hrm.run(seed))
        r.update(run_experiments.run_slm(seed))
        for mdl, m in r.items():
            runs.setdefault(mdl, []).append(m["f1"])
        print(f"  seed {seed} done")
    _save("seeds10_raw", runs)
    # how HRM tier 3 was fit: out_of_fold unless HRM_INSAMPLE=1
    _save("seeds10_hrm_tier3_fit",
          "out_of_fold" if hrm.default_oof() else "in_sample")
    out = {}
    slm = np.array(runs["SLM-attn"])
    for mdl, f1s in runs.items():
        f1s = np.array(f1s)
        entry = {"f1_mean": float(f1s.mean()), "f1_std": float(f1s.std())}
        if mdl != "SLM-attn":
            diff = slm - f1s                     # paired differences
            t = diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff)))
            entry["paired_t_vs_SLM"] = float(t)
            entry["slm_wins"] = int((diff > 0).sum())
        out[mdl] = entry
        print(f"  {mdl}: F1={entry['f1_mean']:.3f}±{entry['f1_std']:.3f}"
              + (f"  t={entry.get('paired_t_vs_SLM', 0):.2f}"
                 if mdl != "SLM-attn" else ""))
    _save("seeds10", out)


def attention_maps(seed=0):
    d = common.load_sequences()
    trs, tes = common.scenario_split(d["scenario"], d["labels"], 0.3, seed)
    net, seqn = _fit_slm(d, trs, seed)
    _, _, cache = net.forward(seqn[tes], d["mask"][tes])
    A = cache["A"]                               # (B, W, W) row=query col=key
    recv = A.mean(1)                             # attention received per window
    yte = d["labels"][tes]
    g, b = recv[yte == 1].mean(0), recv[yte == 0].mean(0)
    _save("attention_received", {"gray": g.tolist(), "benign": b.tolist()})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W = len(g)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(range(W), b, "-o", color="#5B6C8F", label="benign nodes")
    ax.plot(range(W), g, "-s", color="#8C4A5B", label="gray-hole nodes")
    ax.set_xlabel("Telemetry window index")
    ax.set_ylabel("Mean attention received")
    ax.set_title("Attention received per window (test set)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_attention.png"), dpi=150)
    plt.close(fig)
    print("  attention figure written")


def adaptive(seed=0):
    """Train on the standard (static-attack) data; test on pulsed attackers."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
    from manet_sim import ManetSim, SimConfig
    df, feat_cols, d, _ = _load_all()
    y = df["label"].values.astype(int)
    tr = np.ones(len(df), bool)                  # train on all static scenarios
    svm, std, X, _ = _fit_svm(df, feat_cols, tr, seed)
    trs = np.ones(len(d["labels"]), bool)
    net, seqn = _fit_slm(d, trs, seed)
    mu = seqn  # seqn is built from all-static stats inside _fit_slm

    # pulsed-attack test scenarios (n=50)
    rows_agg, seqs, labels = [], [], []
    for mob in ("random_waypoint", "gauss_markov", "manhattan"):
        for drop, sd in ((0.2, 1), (0.4, 2), (0.6, 3)):
            cfg = SimConfig(n_nodes=50, mobility=mob, drop_ratio=drop,
                            duration=300.0, n_flows=max(8, 50 // 4),
                            seed=7000 + sd, attack_mode="onoff")
            out = ManetSim(cfg).run()
            rows_agg.append(out["agg"]); seqs.append(out["seq"])
            labels.append(out["labels"])
    Xa = np.concatenate(rows_agg); ya = np.concatenate(labels)
    Sa = np.concatenate(seqs)
    m_svm = common.metrics(ya, svm.predict(std.transform(Xa)))
    # normalize with the training stats used in _fit_slm
    valid = d["mask"][trs].astype(bool)
    flat = d["seq"][trs][valid]
    mu, sdv = flat.mean(0), flat.std(0) + 1e-8
    San = (Sa - mu) / sdv
    mask = np.ones(San.shape[:2], dtype=np.float32)
    sc = net.predict_proba(San, mask)
    m_slm = common.metrics(ya, (sc >= 0.5).astype(int), sc)
    out = {"SVM_f1": m_svm["f1"], "SLM_f1": m_slm["f1"], "SLM_auc": m_slm["auc"],
           "n_test": int(len(ya)), "note": "train static, test pulsed on/off"}
    print(f"  adaptive attacker: SVM F1={m_svm['f1']:.3f} "
          f"SLM F1={m_slm['f1']:.3f} AUC={m_slm['auc']:.3f}")
    _save("adaptive_attacker", out)


class MeanPoolClassifier(SelfAttnClassifier):
    """Same parameter budget with the attention output zeroed, leaving
    position-wise transforms and masked mean pooling. Isolates the
    contribution of temporal attention."""
    def forward(self, X, mask, y=None):
        p = self.p
        B, W, F = X.shape
        h0 = X @ p["We"] + p["be"]
        h = h0 + p["P"][None, :W, :]
        h1 = h                                    # attention disabled
        z1 = h1 @ p["W1"] + p["b1"]
        a1 = np.maximum(0, z1)
        ff = a1 @ p["W2"] + p["b2"]
        h2 = h1 + ff
        m = mask[:, :, None]
        pooled = (h2 * m).sum(1) / (m.sum(1) + 1e-9)
        logit = pooled @ p["Wo"] + p["bo"]
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))[:, 0]
        cache = dict(X=X, mask=mask, W=W, h=h, h1=h1, z1=z1, a1=a1, h2=h2,
                     pooled=pooled, prob=prob,
                     Q=np.zeros((B, W, self.d_model)),
                     K=np.zeros((B, W, self.d_model)),
                     V=np.zeros((B, W, self.d_model)),
                     A=np.zeros((B, W, W)),
                     attn=np.zeros((B, W, self.d_model)))
        loss = None
        if y is not None:
            eps = 1e-7
            loss = -np.mean(y * np.log(prob + eps)
                            + (1 - y) * np.log(1 - prob + eps))
        return loss, prob, cache

    def backward(self, cache, y):
        g = super().backward(cache, y)
        for k in ("Wq", "Wk", "Wv"):
            g[k][:] = 0.0                         # attention params frozen
        return g


def ablation_raw(seeds=(0, 1, 2)):
    """Per-seed F1 of the attention model and of the no-attention ablation
    (the raw values behind ablation(); also used by per_split_metrics.py)."""
    d = common.load_sequences()
    f1s = {"SLM-attn": [], "SLM-noattn": []}
    for seed in seeds:
        trs, tes = common.scenario_split(d["scenario"], d["labels"], 0.3, seed)
        for name, cls in (("SLM-attn", SelfAttnClassifier),
                          ("SLM-noattn", MeanPoolClassifier)):
            valid = d["mask"][trs].astype(bool)
            flat = d["seq"][trs][valid]
            mu, sd = flat.mean(0), flat.std(0) + 1e-8
            seqn = ((d["seq"] - mu) / sd) * d["mask"][:, :, None]
            net = cls(d["seq"].shape[2], d["seq"].shape[1], 24, 32, seed=seed)
            net.fit(seqn[trs], d["mask"][trs], d["labels"][trs].astype(float),
                    epochs=40, seed=seed)
            sc = net.predict_proba(seqn[tes], d["mask"][tes])
            m = common.metrics(d["labels"][tes], (sc >= 0.5).astype(int))
            f1s[name].append(m["f1"])
    return f1s


def ablation(seeds=(0, 1, 2)):
    f1s = ablation_raw(seeds)
    out = {k: {"f1_mean": float(np.mean(v)), "f1_std": float(np.std(v))}
           for k, v in f1s.items()}
    print(f"  attn {out['SLM-attn']['f1_mean']:.3f} "
          f"vs no-attn {out['SLM-noattn']['f1_mean']:.3f}")
    _save("ablation_attention", out)
    # per-seed values, so a paired comparison does not need a rerun
    _save("ablation_attention_raw", {k: [float(x) for x in v]
                                     for k, v in f1s.items()})


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {"1": transfer, "2": noisy_obs, "3": seeds10,
            "4": attention_maps, "5": adaptive, "6": ablation}
    if which == "all":
        for f in jobs.values():
            f()
    else:
        jobs[which]()
