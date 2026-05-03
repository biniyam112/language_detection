"""
Evaluation script for the spoken language identification model.

Produces:
  - Classification report (precision, recall, F1 per language)
  - Confusion matrix heatmap
  - PCA scatter plot of CNN embeddings coloured by language
  - Training history curves (loss & accuracy)

Usage:
    python src/evaluate.py
    python src/evaluate.py --model cnn
    python src/evaluate.py --model lstm
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CHECKPOINT_DIR,
    LANGUAGES,
    IDX_TO_LANG,
    BATCH_SIZE,
    PROJECT_ROOT,
    MODEL_TYPE,
)
from model import build_model
from dataset import get_dataloaders

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_CHOICES = ("cnn", "lstm")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a spoken language identification model."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=MODEL_TYPE,
        help=f"Model architecture to evaluate. Defaults to MODEL_TYPE={MODEL_TYPE!r}.",
    )
    return parser.parse_args()


def load_best_model(
    device: torch.device,
    model_type: str,
    checkpoint_dir: Path,
) -> torch.nn.Module:
    ckpt_path = checkpoint_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Train first.")

    model = build_model(model_type).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.4f}, val_acc={ckpt['val_acc']:.1%})")
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run model on a DataLoader and return true labels, predictions, and
    128-d embeddings."""
    all_labels = []
    all_preds = []
    all_embeds = []

    for mel, labels in tqdm(loader, desc="Evaluating"):
        mel = mel.to(device)
        logits = model(mel)
        preds = logits.argmax(dim=1).cpu()
        embeds = model.extract_embeddings(mel).cpu()

        all_labels.append(labels)
        all_preds.append(preds)
        all_embeds.append(embeds)

    return (
        torch.cat(all_labels).numpy(),
        torch.cat(all_preds).numpy(),
        torch.cat(all_embeds).numpy(),
    )


def plot_confusion_matrix(y_true, y_pred, results_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[l.capitalize() for l in LANGUAGES],
        yticklabels=[l.capitalize() for l in LANGUAGES],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    path = results_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pca_embeddings(embeddings, labels, results_dir: Path, model_type: str):
    """Run PCA on 128-d embeddings and make a 2-D scatter plot."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, lang in enumerate(LANGUAGES):
        mask = labels == idx
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=lang.capitalize(), alpha=0.6, s=20,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"PCA of {model_type.upper()} Embeddings by Language")
    ax.legend()
    fig.tight_layout()
    path = results_dir / "pca_embeddings.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_training_history(checkpoint_dir: Path, results_dir: Path):
    """Plot loss and accuracy curves from the saved history JSON."""
    history_path = checkpoint_dir / "history.json"
    if not history_path.exists():
        print("No training history found, skipping curves.")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()

    fig.tight_layout()
    path = results_dir / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    args = parse_args()
    model_type = args.model.lower()
    checkpoint_dir = CHECKPOINT_DIR / model_type
    results_dir = RESULTS_DIR / model_type
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model type: {model_type}")
    model = load_best_model(device, model_type, checkpoint_dir)
    loaders = get_dataloaders(batch_size=BATCH_SIZE)

    y_true, y_pred, embeddings = collect_predictions(
        model, loaders["test"], device
    )

    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    target_names = [l.capitalize() for l in LANGUAGES]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)

    acc = accuracy_score(y_true, y_pred)
    print(f"Overall accuracy: {acc:.1%}")

    # Save report to file
    report_path = results_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
        f.write(f"\nOverall accuracy: {acc:.1%}\n")
    print(f"Saved: {report_path}")

    # Plots
    plot_confusion_matrix(y_true, y_pred, results_dir)
    plot_pca_embeddings(embeddings, y_true, results_dir, model_type)
    plot_training_history(checkpoint_dir, results_dir)

    print(f"\nAll results saved to {results_dir}/")


if __name__ == "__main__":
    main()
