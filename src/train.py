"""
Supervised imitation learning stage for the chess policy network.

This script trains a policy model to imitate expert moves extracted from PGN
data. The resulting checkpoint can then be fine-tuned via reinforcement
learning in the RL pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import FenMoveDataset, collate_fn
from .features import mask_logits
from .models import PolicyNet


class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss that ignores illegal moves by masking logits."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        masked_logits = mask_logits(logits, legal_mask)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        batch, vocab = logits.shape
        indices = torch.arange(batch, device=logits.device)
        valid = (targets >= 0) & (targets < vocab)
        selected = log_probs[indices, targets.clamp(min=0)]
        selected = torch.where(valid, selected, torch.zeros_like(selected))
        return -selected[valid].mean() if valid.any() else torch.zeros((), device=logits.device)


class SupervisedTrainer:
    """Encapsulates the imitation learning training loop."""

    def __init__(
        self,
        model: PolicyNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = MaskedCrossEntropyLoss()
        self.scaler = GradScaler(enabled=device.type == "cuda")

    def _run_epoch(self, train: bool = True) -> float:
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)
        epoch_loss, total_samples = 0.0, 0

        for planes, targets, legal_mask in tqdm(loader, leave=False, desc="train" if train else "valid"):
            planes = planes.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            legal_mask = legal_mask.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train), autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits = self.model(planes)
                loss = self.criterion(logits, targets, legal_mask)

            batch_size = planes.size(0)
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        return epoch_loss / max(1, total_samples)

    def fit(self, epochs: int = 10) -> None:
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train=True)
            val_loss = self._run_epoch(train=False)
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")


@torch.no_grad()
def evaluate(model: PolicyNet, data_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate the model on a validation dataloader, returning loss and accuracy
    metrics (top-1/top-3) under legality constraints.
    """
    model.eval()
    total_loss = total_samples = top1 = top3 = 0

    criterion = MaskedCrossEntropyLoss()

    for planes, targets, legal_mask in data_loader:
        planes = planes.to(device)
        targets = targets.to(device)
        legal_mask = legal_mask.to(device)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(planes)
            loss = criterion(logits, targets, legal_mask)

        batch = planes.size(0)
        total_loss += loss.item() * batch
        total_samples += batch

        masked_logits = mask_logits(logits, legal_mask)
        probs = torch.softmax(masked_logits, dim=-1)
        topk = torch.topk(probs, k=3, dim=-1).indices

        gold = targets.unsqueeze(-1)
        top1 += (topk[:, 0:1] == gold).any(dim=-1).sum().item()
        top3 += (topk == gold).any(dim=-1).sum().item()

    mean_loss = total_loss / max(1, total_samples)
    perplexity = math.exp(min(20.0, mean_loss))
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "top1": top1 / max(1, total_samples),
        "top3": top3 / max(1, total_samples),
    }


def load_vocab(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_dataloaders(workdir: Path, vocab: Dict[str, int], batch_size: int = 256, num_workers: int = 2):
    train_ds = FenMoveDataset(workdir / "train.jsonl", vocab)
    val_ds = FenMoveDataset(workdir / "val.jsonl", vocab)
    pin_memory = torch.cuda.is_available()
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    return train_dl, val_dl


def save_checkpoint(model: PolicyNet, optimizer: torch.optim.Optimizer, vocab: Dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab": vocab,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised pre-training of the chess policy network.")
    parser.add_argument("--workdir", type=Path, default=Path("data/working"))
    parser.add_argument("--vocab", type=Path, default=Path("data/working/move_vocab_4672.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/policy_model.pt"))
    parser.add_argument("--resume", type=Path, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def run_supervised_training(
    workdir: Path,
    vocab_path: Path,
    checkpoint_path: Path,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_workers: int = 2,
    resume: Path | None = None,
) -> Dict[str, float]:
    vocab = load_vocab(vocab_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(workdir, vocab, batch_size=batch_size, num_workers=num_workers)

    model = PolicyNet(len(vocab))

    trainer = SupervisedTrainer(model, train_loader, val_loader, device=device, lr=lr)

    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"Resumed from checkpoint {resume}")

    trainer.fit(epochs=epochs)

    save_checkpoint(trainer.model, trainer.optimizer, vocab, checkpoint_path)
    metrics = evaluate(trainer.model, val_loader, device)
    print(f"Validation metrics after training: {metrics}")
    return metrics


def main():
    args = parse_args()
    run_supervised_training(
        workdir=args.workdir,
        vocab_path=args.vocab,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
