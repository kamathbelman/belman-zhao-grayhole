"""Compact self-attention sequence classifier over routing windows, in NumPy.

    embed -> +positional -> single-head self-attention (+residual)
          -> position-wise FFN (+residual) -> masked mean-pool -> linear head

Trained with Adam and manual backprop. The backward pass is verified by
numerical gradient checking (grad_check(), run as the main program).
"""
from __future__ import annotations
import numpy as np
import time


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class SelfAttnClassifier:
    def __init__(self, n_feat, max_w, d_model=24, d_ff=32, seed=0):
        rng = np.random.default_rng(seed)
        s = 0.1
        self.p = {
            "We": rng.normal(0, s, (n_feat, d_model)),
            "be": np.zeros(d_model),
            "P":  rng.normal(0, s, (max_w, d_model)),
            "Wq": rng.normal(0, s, (d_model, d_model)),
            "Wk": rng.normal(0, s, (d_model, d_model)),
            "Wv": rng.normal(0, s, (d_model, d_model)),
            "W1": rng.normal(0, s, (d_model, d_ff)),
            "b1": np.zeros(d_ff),
            "W2": rng.normal(0, s, (d_ff, d_model)),
            "b2": np.zeros(d_model),
            "Wo": rng.normal(0, s, (d_model, 1)),
            "bo": np.zeros(1),
        }
        self.d_model = d_model

    def forward(self, X, mask, y=None):
        p = self.p
        B, W, F = X.shape
        d = self.d_model
        h0 = X @ p["We"] + p["be"]                 # (B,W,d)
        h = h0 + p["P"][None, :W, :]
        Q = h @ p["Wq"]; K = h @ p["Wk"]; V = h @ p["Wv"]
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d)   # (B,W,W)
        keymask = (mask[:, None, :] - 1.0) * 1e9         # pad keys -> -inf
        scores = scores + keymask
        A = softmax(scores, axis=-1)                     # (B,W,W)
        attn = A @ V                                     # (B,W,d)
        h1 = h + attn
        z1 = h1 @ p["W1"] + p["b1"]                       # (B,W,dff)
        a1 = np.maximum(0, z1)
        ff = a1 @ p["W2"] + p["b2"]                       # (B,W,d)
        h2 = h1 + ff
        m = mask[:, :, None]
        pooled = (h2 * m).sum(1) / (m.sum(1) + 1e-9)     # (B,d)
        logit = pooled @ p["Wo"] + p["bo"]               # (B,1)
        prob = sigmoid(logit)[:, 0]
        cache = dict(X=X, mask=mask, W=W, h=h, Q=Q, K=K, V=V, A=A, attn=attn,
                     h1=h1, z1=z1, a1=a1, h2=h2, pooled=pooled, prob=prob)
        loss = None
        if y is not None:
            eps = 1e-7
            loss = -np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))
        return loss, prob, cache

    def backward(self, cache, y):
        p = self.p
        X, mask = cache["X"], cache["mask"]
        B, W, F = X.shape
        d = self.d_model
        g = {k: np.zeros_like(v) for k, v in p.items()}

        prob = cache["prob"]
        dlogit = ((prob - y) / B)[:, None]               # (B,1) d loss/d logit
        g["Wo"] += cache["pooled"].T @ dlogit
        g["bo"] += dlogit.sum(0)
        dpooled = dlogit @ p["Wo"].T                      # (B,d)
        m = mask[:, :, None]
        denom = (m.sum(1) + 1e-9)                         # (B,1)
        dh2 = (dpooled / denom)[:, None, :] * m           # (B,W,d)

        # FFN residual: h2 = h1 + ff
        dh1 = dh2.copy()
        dff = dh2
        g["W2"] += np.einsum("bwf,bwd->fd", cache["a1"], dff)
        g["b2"] += dff.reshape(-1, d).sum(0)
        da1 = dff @ p["W2"].T                              # (B,W,dff)
        dz1 = da1 * (cache["z1"] > 0)
        g["W1"] += np.einsum("bwd,bwf->df", cache["h1"], dz1)
        g["b1"] += dz1.reshape(-1, dz1.shape[-1]).sum(0)
        dh1 += dz1 @ p["W1"].T                             # (B,W,d)

        # attention residual: h1 = h + attn
        dh = dh1.copy()
        dattn = dh1
        # attn = A @ V
        dA = dattn @ cache["V"].transpose(0, 2, 1)         # (B,W,W)
        dV = cache["A"].transpose(0, 2, 1) @ dattn         # (B,W,d)
        # softmax backward: dscores = A * (dA - sum(dA*A))
        dscores = cache["A"] * (dA - (dA * cache["A"]).sum(-1, keepdims=True))
        dscores /= np.sqrt(d)
        dQ = dscores @ cache["K"]                          # (B,W,d)
        dK = dscores.transpose(0, 2, 1) @ cache["Q"]       # (B,W,d)
        g["Wq"] += np.einsum("bwd,bwe->de", cache["h"], dQ)
        g["Wk"] += np.einsum("bwd,bwe->de", cache["h"], dK)
        g["Wv"] += np.einsum("bwd,bwe->de", cache["h"], dV)
        dh += dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T

        # positional + embedding: h = X@We + be + P
        g["P"][:W] += dh.sum(0)
        g["We"] += np.einsum("bwf,bwd->fd", X, dh)
        g["be"] += dh.reshape(-1, d).sum(0)
        return g

    def fit(self, X, mask, y, Xval=None, mval=None, yval=None,
            epochs=40, batch=128, lr=3e-3, l2=1e-4, verbose=False, seed=0):
        rng = np.random.default_rng(seed)
        mt = {k: np.zeros_like(v) for k, v in self.p.items()}
        vt = {k: np.zeros_like(v) for k, v in self.p.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        n = len(X); step = 0
        for ep in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                loss, _, cache = self.forward(X[bi], mask[bi], y[bi])
                g = self.backward(cache, y[bi])
                step += 1
                for k in self.p:
                    grad = g[k] + l2 * self.p[k]
                    mt[k] = b1 * mt[k] + (1 - b1) * grad
                    vt[k] = b2 * vt[k] + (1 - b2) * grad ** 2
                    mhat = mt[k] / (1 - b1 ** step)
                    vhat = vt[k] / (1 - b2 ** step)
                    self.p[k] -= lr * mhat / (np.sqrt(vhat) + eps)
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                tr_loss = self.forward(X, mask, y)[0]
                msg = f"  epoch {ep:3d}  train_loss={tr_loss:.4f}"
                if Xval is not None:
                    vl = self.forward(Xval, mval, yval)[0]
                    msg += f"  val_loss={vl:.4f}"
                print(msg)
        return self

    def predict_proba(self, X, mask):
        return self.forward(X, mask)[1]


# numerical gradient check of backward()
def grad_check():
    rng = np.random.default_rng(0)
    B, W, F = 4, 5, 3
    X = rng.normal(size=(B, W, F))
    mask = np.ones((B, W)); mask[0, :2] = 0; mask[1, 0] = 0
    y = rng.integers(0, 2, B).astype(float)
    net = SelfAttnClassifier(F, W, d_model=6, d_ff=7, seed=1)
    _, _, cache = net.forward(X, mask, y)
    g = net.backward(cache, y)
    eps = 1e-5
    max_rel = 0.0
    for k in net.p:
        flat = net.p[k].ravel()
        gflat = g[k].ravel()
        for _ in range(min(8, flat.size)):
            i = rng.integers(flat.size)
            orig = flat[i]
            flat[i] = orig + eps
            lp = net.forward(X, mask, y)[0]
            flat[i] = orig - eps
            lm = net.forward(X, mask, y)[0]
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-9)
            max_rel = max(max_rel, rel)
    return max_rel


if __name__ == "__main__":
    mr = grad_check()
    print(f"gradient check max relative error: {mr:.2e}  ->",
          "PASS" if mr < 1e-4 else "FAIL")
