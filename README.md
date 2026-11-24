# Gray-hole detection in MANETs: code, data and results

This repository accompanies the paper

> A. Kamath Belman and J. Zhao, "Detecting Gray-Hole Attacks in Mobile Ad Hoc
> Networks with Attention-Based Sequence Models and Hierarchical Reasoning,"
> International Conference on Computing, Networking and Communications
> (ICNC), 2027, to appear.

It contains the two simulators, the datasets, the detection models, and the
result files behind every number in the paper.

## Contents

| Path | What it is |
|---|---|
| `sim/` | Discrete-time Python MANET simulator (the bootstrap track) and its dataset generator |
| `ns3/` | ns-3.40 scenario `grayhole-manet.cc`, campaign scripts, telemetry validator, dataset parser |
| `ns3/ns3_runs_full/` | Raw output of the 540 NS-3 runs: per-node window counters (`.csv`), adjacency (`.adj.csv`), run parameters and totals (`.meta.json`) |
| `data/` | Bootstrap dataset: 81 scenarios, 4,320 labeled nodes |
| `data_ns3_grid/` | NS-3 main grid: 525 scenarios, 31,500 labeled nodes |
| `data_ns3_full/` | NS-3 grid plus the 15 legacy-mirror scenarios: 540 scenarios, 31,950 nodes |
| `models/` | Detectors, experiments, statistics and figure scripts |
| `results/` | Result files and figures as reported |
| `reproduce.sh` | Reruns the experiments on the included datasets |

Each dataset folder holds `node_features.csv` (30 aggregate features per
node, the label, the scenario id, and the scenario settings), `sequences.npz`
(15 windows of 10 features per node, with a padding mask), `topology.npz`
(one adjacency matrix per scenario) and `meta.json`. The scenario settings
(`mobility`, `drop_ratio`, `n_nodes`) are used only to group results and are
never model inputs.

## Model names

The result files use the working names from the code.

| Key in the result files | Model in the paper |
|---|---|
| `SVM-RBF` | RBF-SVM baseline |
| `RandomForest` | Random forest baseline |
| `HRM` | Three-tier hierarchical neighborhood-aware classifier |
| `SLM-attn` | Compact attention-based sequence classifier |

## Setup

Python 3.13 with the pinned packages:

```bash
pip install -r requirements.txt
```

## Reproducing the results

```bash
bash reproduce.sh
```

This takes roughly 45 minutes on a laptop. It uses the included datasets and
rewrites the files in `results/`. The NS-3 simulations, the timing benchmark
and the language-model fine-tune are not rerun by this script.

| Paper item | Script | Result file |
|---|---|---|
| Table II (bootstrap) | `models/run_experiments.py` | `results/results_summary.json`, `results/results_table.csv` |
| Table III (NS-3 main grid) | `models/ns3_experiments.py` | `results/ns3_full_results.json` |
| F1 by attack intensity, bootstrap | `models/make_figures.py` | `results/figures/fig3_f1_vs_dropratio.png` |
| F1 by attack intensity, NS-3 | `models/ns3_perdrop.py` | `results/ns3_full_perdrop.json` |
| Comparison with prior work | `models/legacy_mirror_eval.py` | `results/legacy_mirror_svm.json` |
| Ten splits, mobility transfer, telemetry noise, pulsed attacker, attention maps, attention ablation | `models/extra_experiments.py` | `results/extra_results.json`, `results/figures/fig4_attention.png`, `results/figures/fig5_noisy_obs.png` |
| Per-split metrics and split ids | `models/per_split_metrics.py` | `results/per_split_metrics.json`, `results/per_split_metrics_ns3_ten_splits_extra.json` |
| Paired comparisons | `models/split_statistics.py` | `results/split_statistics.json` |
| Inference timing | `models/benchmark_inference.py` | `results/benchmark_inference.json` |
| Fine-tuned language model | `models/finetune_slm_mps.py` | `results/real_slm_results.json` |

Notes:

- Splits are by scenario: 30% of the scenarios form the test set and no
  scenario appears on both sides. `results/per_split_metrics.json` lists the
  test scenario ids of every split.
- The HRM's third tier is trained on out-of-fold tier-1 scores (five
  scenario-grouped folds of the training scenarios). Setting
  `HRM_INSAMPLE=1` trains it on in-sample scores instead;
  `per_split_metrics.json` reports both variants.
- `results/split_statistics.json` gives, for each pair of models, the
  per-split differences, the ordinary paired t-test and the corrected
  resampled t-test of Nadeau and Bengio (2003).
- The timing numbers depend on the machine. The reported ones were measured
  on an Apple M3 Max, single thread, with x86-64 Python under Rosetta 2.
- The language-model result comes from one split (seed 0) and one run of
  Phi-3-mini with LoRA. It needs `torch`, `transformers` and `peft`.
- Two further scripts are not used in the paper:
  `models/hyperparam_sensitivity.py` (baseline hyperparameter sweep) and
  `models/hrm_oof_diagnostic.py` (tier-1 scores in-sample, out-of-fold and
  on test data).

## Regenerating the data

Bootstrap dataset (fixed seeds):

```bash
python3 sim/generate_dataset.py demo
```

NS-3 campaign (ns-3.40, 540 runs of 300 s or 600 s simulated time):

```bash
bash ns3/run_full_campaign_macos.sh
```

The script downloads and builds ns-3.40 under `~/ns3-grayhole`, runs the
campaign in parallel, validates every run with `ns3/validate_telemetry.py`
and builds `data_ns3_full/` and `data_ns3_grid/` with
`ns3/parse_ns3_to_dataset.py`. `ns3/smoke_test.sh` runs a small grid first.
On other systems, build `ns3/grayhole-manet.cc` in the `scratch/` folder of
an ns-3.40 tree and use `ns3/run_campaign.sh`.

## Telemetry

All counters refer to one packet population: unicast transit application
data packets seen at a node's routing decision. Each packet is followed to
exactly one outcome at that node, forwarded (handed to the device) or
dropped, and the validator checks the conservation identity

```
arrived = forwarded + dropped + still in flight
```

in every window of every run. The ten per-window features are the forwarding
and drop ratios, the arrival, forward and drop counts, relayed RREQs, sent
RREPs, originated data packets, the mean number of neighbors, and the
control-to-data ratio.

## Citation

```bibtex
@inproceedings{kamathbelman2027grayhole,
  author    = {Kamath Belman, Amith and Zhao, Juzi},
  title     = {Detecting Gray-Hole Attacks in Mobile Ad Hoc Networks with
               Attention-Based Sequence Models and Hierarchical Reasoning},
  booktitle = {International Conference on Computing, Networking and
               Communications (ICNC)},
  year      = {2027},
  note      = {to appear}
}
```
