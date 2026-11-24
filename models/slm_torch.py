"""PyTorch version of the sequence classifier: a multi-head Transformer
encoder over the numeric routing-window telemetry. It mirrors slm_numpy.py
and uses the MPS or CUDA backend when one is available.

Requires torch (pip3 install torch). Run: python3 slm_torch.py
"""
from __future__ import annotations
import os, time, json
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch not installed. On the Mac: pip3 install torch")

import common


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SeqTransformer(nn.Module):
    def __init__(self, n_feat, max_w, d_model=64, nhead=4, ff=128, layers=2, drop=0.1):
        super().__init__()
        self.embed = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_w, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, drop, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x, mask):
        # x: (B,W,F)  mask: (B,W) 1=valid
        h = self.embed(x) + self.pos[:, : x.size(1)]
        pad = mask == 0
        h = self.enc(h, src_key_padding_mask=pad)
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(1) / (m.sum(1) + 1e-9)
        return self.head(pooled).squeeze(-1)


def train_eval(seed=0, epochs=60):
    dev = get_device()
    print("device:", dev)
    d = common.load_sequences()
    seq, mask, y, scen = d["seq"], d["mask"], d["labels"], d["scenario"]
    tr, te = common.scenario_split(scen, y, test_frac=0.3, seed=seed)
    valid = mask[tr].astype(bool); flat = seq[tr][valid]
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    seqn = ((seq - mu) / sd) * mask[:, :, None]

    def T(a): return torch.tensor(a, dtype=torch.float32, device=dev)
    Xtr, Mtr, ytr = T(seqn[tr]), T(mask[tr]), T(y[tr].astype(np.float32))
    Xte, Mte = T(seqn[te]), T(mask[te])

    net = SeqTransformer(seq.shape[2], seq.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())],
                              device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = len(Xtr); bs = 128
    t0 = time.time()
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            bi = perm[s:s + bs]
            opt.zero_grad()
            logit = net(Xtr[bi], Mtr[bi])
            loss = lossf(logit, ytr[bi])
            loss.backward(); opt.step()
    train_t = time.time() - t0

    net.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        score = torch.sigmoid(net(Xte, Mte)).cpu().numpy()
        # amortized time per node (batch time / n), not single-sample latency
        amort_ms = (time.perf_counter() - t0) / max(1, te.sum()) * 1e3
    pred = (score >= 0.5).astype(int)
    m = common.metrics(y[te], pred, score)
    m.update(train_time_s=train_t, amortized_cpu_ms_per_node=amort_ms,
             device=str(dev), n_test=int(te.sum()))
    return {"SLM-torch": m}


def finetune_hf_slm():
    """Placeholder. LoRA fine-tuning of a small language model (Phi-3-mini or
    Gemma-2B) on tokenized telemetry is implemented in finetune_slm_mps.py."""
    raise NotImplementedError("enable after tokenization scheme is finalised")


if __name__ == "__main__":
    print(json.dumps(train_eval(), indent=2, default=float))
