import os
import json
import numpy as np
import torch
import chess
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import math

from dataset import FenMoveDataset, collate_fn
from models import PolicyNet, ChessResNet

class MaskedCrossEntropyLoss(nn.Module):
    def forward(self, logits, target, legal_mask):
        masked_logits = logits.masked_fill(legal_mask==0, float("-inf"))
        log_probs = F.log_softmax(masked_logits, dim=-1)
        B,V = logits.shape
        idx = torch.arange(B, device=logits.device)
        valid = (target >= 0) & (target < V)
        tgt_logp = log_probs[idx, target]
        tgt_logp[~valid] = 0.0
        if valid.any():
            return -(tgt_logp[valid].mean())
        else:
            return torch.tensor(0.0, device=logits.device)

class Trainer:
    def __init__(self, model, train_dl, val_dl, device="cpu", lr=1e-3):
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = MaskedCrossEntropyLoss()

    def run_epoch(self, train=True):
        self.model.train(train)
        total_loss, total = 0.0, 0
        with torch.set_grad_enabled(train):
            for X,y,mask in self.train_dl if train else self.val_dl:
                X,y,mask = X.to(self.device), y.to(self.device), mask.to(self.device)
                logits = self.model(X)
                loss = self.criterion(logits, y, mask)
                if train:
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                bs = X.size(0)
                total_loss += loss.item() * bs
                total += bs
        return total_loss / max(total,1)

    def fit(self, epochs=5):
        for ep in range(epochs):
            tr = self.run_epoch(True)
            va = self.run_epoch(False)
            print(f"Epoch {ep}: train {tr:.4f} | val {va:.4f}")

def masked_ce(logits, targets, legal, label_smoothing=0.05):
    masked = torch.full_like(logits, float("-inf"))
    for i, ids in enumerate(legal):
        if ids: masked[i, ids] = logits[i, ids]
        else:   masked[i] = logits[i]
    return F.cross_entropy(masked, targets, label_smoothing=label_smoothing)

def evaluate(model, val_dl, device):
    model.eval()
    tot_loss=n=0; top1=top3=0
    for xb, yb, legal in val_dl:
        xb, yb = xb.to(device), yb.to(device)

        safe_y = yb.clone()
        for i,(yi,L) in enumerate(zip(yb.tolist(), legal)):
            if (yi<0) or (yi not in L):
                safe_y[i] = torch.tensor(L[0] if L else 0, device=device)

        with autocast("cuda", enabled=(device=="cuda")):
            logits = model(xb)
            loss = masked_ce(logits, safe_y, legal, label_smoothing=0.05)

        B = xb.size(0); tot_loss += float(loss)*B; n += B

        for i in range(B):
            ids = legal[i] if legal[i] else torch.arange(logits.size(1)).tolist()
            li = logits[i, ids]
            k = min(3, len(ids))
            topk = torch.topk(li, k=k).indices.cpu().tolist()
            pred1 = ids[topk[0]]; gold = int(safe_y[i])
            if pred1==gold: top1+=1
            if gold in [ids[j] for j in topk]: top3+=1

    loss = tot_loss/max(1,n); ppl = math.exp(min(20,loss))
    return {"val_loss":loss, "val_ppl":ppl, "val_top1":top1/n, "val_top3":top3/n}

if __name__ == "__main__":
    workdir = "data/working"
    train_path = os.path.join(workdir, "train.jsonl")
    val_path   = os.path.join(workdir, "val.jsonl")
    vocab_path = os.path.join(workdir, "move_vocab_4672.json")

    # Load vocab
    with open(vocab_path, "r") as f:
        move2id = json.load(f)
    vocab_size = len(move2id)
    print(f"Loaded vocab size = {vocab_size}")

    # Dataset + DataLoader
    train_ds = FenMoveDataset(train_path, move2id)
    val_ds   = FenMoveDataset(val_path, move2id)

    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Model + Trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PolicyNet(vocab_size)
    trainer = Trainer(model, train_dl, val_dl, device=device, lr=1e-3)

    # Train
    trainer.fit(epochs=30)

    # Save model checkpoint
    ckpt_path = "models/policy_model.pt"
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": trainer.opt.state_dict(),
        "vocab": move2id
    }, ckpt_path)

    print(f"Model saved to {ckpt_path}")