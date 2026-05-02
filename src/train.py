"""
Training script for the spoken language identification CNN.

Usage:
    python src/train.py
"""

import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Allow running from project root  (python src/train.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CHECKPOINT_DIR,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    ES_PATIENCE,
    ES_MIN_DELTA,
    LR_PATIENCE,
    LR_FACTOR,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
)
from model import LanguageCNN, count_parameters
from dataset import get_dataloaders


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = ES_PATIENCE, min_delta: float = ES_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for mel, labels in tqdm(loader, desc="  train", leave=False):
        mel, labels = mel.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(mel)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mel.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += mel.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for mel, labels in tqdm(loader, desc="  val  ", leave=False):
        mel, labels = mel.to(device), labels.to(device)
        logits = model(mel)
        loss = criterion(logits, labels)

        running_loss += loss.item() * mel.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += mel.size(0)

    return running_loss / total, correct / total


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    loaders = get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Model
    model = LanguageCNN().to(device)
    print(f"Model parameters: {count_parameters(model):,}\n")

    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=LR_FACTOR,
                                  patience=LR_PATIENCE)
    early_stop = EarlyStopping()

    best_val_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("=" * 60)
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>10}  {'Val Acc':>9}  {'LR':>10}")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, loaders["val"], criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"{epoch:5d}  {train_loss:10.4f}  {train_acc:8.1%}  "
              f"{val_loss:10.4f}  {val_acc:8.1%}  {current_lr:10.2e}  "
              f"({elapsed:.1f}s)")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = CHECKPOINT_DIR / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"         -> saved best model (val_loss={val_loss:.4f})")

        if early_stop.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save training history
    history_path = CHECKPOINT_DIR / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
