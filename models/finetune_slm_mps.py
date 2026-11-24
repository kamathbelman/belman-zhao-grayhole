"""Fine-tune a small language model (Phi-3-mini or Gemma-2B) with LoRA on
serialized routing telemetry, on the Metal/MPS backend or a CUDA GPU.

Setup: pip3 install torch transformers peft accelerate numpy pandas scikit-learn
       (Gemma additionally needs: huggingface-cli login)
Run from models/:
    python3 finetune_slm_mps.py                        # Phi-3-mini, bootstrap data
    python3 finetune_slm_mps.py --data ../data_ns3     # NS-3 data
    python3 finetune_slm_mps.py --model google/gemma-2b
    python3 finetune_slm_mps.py --dry-run              # print example prompts only
Serialization: raw floats tokenize badly, so each feature is binned into a
decile digit 0..9 (edges from the training split only), one letter plus digit
per feature: "W00 f9 d0 r3 w3 z0 q2 p1 o4 n5 c2 | W01 f8 d1 ...". The model
answers yes or no; the loss uses only that token and P(yes) is the score."""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

import common
from common import metrics, scenario_split

PROMPT = ("You are analyzing per-window routing telemetry from one node of a "
          "mobile ad hoc network. Features are decile-binned (0=lowest, "
          "9=highest): f=forwarding ratio, d=drop ratio, r=data received to "
          "forward, w=data forwarded, z=data dropped, q=RREQs relayed, "
          "p=RREPs sent, o=data originated, "
          "n=avg neighbors, c=control-to-data ratio.\n"
          "Telemetry:\n{telemetry}\n"
          "Question: is this node a gray-hole attacker that selectively drops "
          "data it agreed to forward? Answer yes or no.\nAnswer:")
CODES = "fdrwzqponc"           # one letter per feature, order = feat_keys


def bin_edges_from_train(seq, mask, tr):
    """Per-feature decile edges from valid training windows only."""
    valid = mask[tr].astype(bool)
    flat = seq[tr][valid]                                  # (N_windows, F)
    qs = np.linspace(0.1, 0.9, 9)
    return np.stack([np.quantile(flat[:, j], qs) for j in range(flat.shape[1])])


def serialize(x, m, edges):
    """One node's (T,F) matrix + validity mask -> compact token string."""
    parts = []
    for t in range(x.shape[0]):
        if m[t] == 0:
            continue
        bins = [int(np.searchsorted(edges[j], x[t, j])) for j in range(x.shape[1])]
        fields = " ".join(f"{CODES[j]}{bins[j]}" for j in range(x.shape[1]))
        parts.append(f"W{t:02d} {fields}")
    return " | ".join(parts)


def build_texts(data_dir, seed=0):
    common.DATA = data_dir
    d = common.load_sequences()
    seq, mask, y, scen = d["seq"], d["mask"], d["labels"], d["scenario"]
    tr, te = scenario_split(scen, y, test_frac=0.3, seed=seed)
    edges = bin_edges_from_train(seq, mask, tr)
    texts = [PROMPT.format(telemetry=serialize(seq[i], mask[i], edges))
             for i in range(len(seq))]
    return np.array(texts), y, tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__),
                                                   "..", "data"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=3000,
                    help="cap on training examples (subsample for speed)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    texts, y, tr, te = build_texts(args.data, args.seed)
    print(f"dataset: {len(texts)} nodes ({int(y.sum())} gray), "
          f"train={int(tr.sum())} test={int(te.sum())}")
    if args.dry_run:
        i = int(np.where(y[tr] == 1)[0][0])
        print("---- example (gray) ----\n", texts[tr][i][:900])
        return

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    # trust_remote_code stays off: the modeling code bundled with Phi-3 fails
    # on transformers >= 4.45, and the library supports Phi-3 natively.
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # answer-position indexing assumes right pad
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager")
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules="all-linear", task_type="CAUSAL_LM"))
    model.to(dev).train()
    model.print_trainable_parameters()

    yes_id = tok(" yes", add_special_tokens=False).input_ids[-1]
    no_id = tok(" no", add_special_tokens=False).input_ids[-1]

    rng = np.random.default_rng(args.seed)
    tr_idx = np.where(tr)[0]
    if len(tr_idx) > args.max_train:
        tr_idx = rng.choice(tr_idx, args.max_train, replace=False)

    def batch_logits(idxs):
        enc = tok([texts[i] for i in idxs], return_tensors="pt",
                  padding=True, truncation=True, max_length=1024).to(dev)
        out = model(**enc).logits
        last = enc["attention_mask"].sum(1) - 1        # answer position
        rows = torch.arange(len(idxs), device=dev)
        return out[rows, last][:, [no_id, yes_id]]      # (B, 2)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    lossf = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    n_steps = (len(tr_idx) + args.batch - 1) // args.batch
    for ep in range(args.epochs):
        rng.shuffle(tr_idx)
        tot = 0.0
        for k, s in enumerate(range(0, len(tr_idx), args.batch)):
            bi = tr_idx[s:s + args.batch]
            opt.zero_grad()
            logits = batch_logits(bi)
            target = torch.tensor(y[bi], device=dev, dtype=torch.long)
            loss = lossf(logits, target)
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(bi)
            if (k + 1) % 25 == 0 or k + 1 == n_steps:
                el = time.time() - t0
                print(f"  ep {ep} step {k+1}/{n_steps} "
                      f"loss={tot/((k+1)*args.batch):.4f} "
                      f"({el:.0f}s, {el/((ep*n_steps)+k+1):.1f}s/step)",
                      flush=True)
        print(f"epoch {ep}: loss={tot/len(tr_idx):.4f} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    model.eval()
    te_idx = np.where(te)[0]
    scores = []
    with torch.no_grad():
        for s in range(0, len(te_idx), args.batch):
            bi = te_idx[s:s + args.batch]
            p = torch.softmax(batch_logits(bi), dim=-1)[:, 1]
            scores.extend(p.float().cpu().numpy().tolist())
    scores = np.array(scores)
    m = metrics(y[te_idx], (scores >= 0.5).astype(int), scores)
    print({k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
    import json
    with open(os.path.join(os.path.dirname(__file__), "..", "results",
                           "real_slm_results.json"), "w") as f:
        json.dump({"model": args.model, "data": args.data, **m}, f, indent=2)


if __name__ == "__main__":
    main()
