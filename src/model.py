"""
CNN model for spoken language identification.

Architecture
------------
Input:  (batch, 1, 128, T)   — single-channel Mel-spectrogram

    Conv2d(1,   32,  3×3)  + BatchNorm + ReLU + MaxPool(2×2)
    Conv2d(32,  64,  3×3)  + BatchNorm + ReLU + MaxPool(2×2)
    Conv2d(64,  128, 3×3)  + BatchNorm + ReLU + MaxPool(2×2)
    Conv2d(128, 256, 3×3)  + BatchNorm + ReLU + AdaptiveAvgPool(1×1)

    Flatten
    Linear(256, 128) + ReLU + Dropout(0.3)
    Linear(128, num_classes)

AdaptiveAvgPool makes the model agnostic to the exact time-frame count,
so minor variations in audio length are handled gracefully.
"""

import torch
import torch.nn as nn

from config import NUM_CLASSES


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → pool."""

    def __init__(self, in_ch: int, out_ch: int, pool: nn.Module):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            pool,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LanguageCNN(nn.Module):
    """4-layer CNN for language classification from Mel-spectrograms."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32, nn.MaxPool2d(2)),
            ConvBlock(32, 64, nn.MaxPool2d(2)),
            ConvBlock(64, 128, nn.MaxPool2d(2)),
            ConvBlock(128, 256, nn.AdaptiveAvgPool2d(1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Mel-spectrogram tensor of shape ``(batch, 1, n_mels, T)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 128-d penultimate-layer embeddings (useful for PCA
        visualisation and nearest-neighbour analysis)."""
        x = self.features(x)
        x = self.classifier[0](x)  # Flatten
        x = self.classifier[1](x)  # Linear(256, 128)
        x = self.classifier[2](x)  # ReLU
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LanguageCNN()
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    dummy = torch.randn(2, 1, 128, 157)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
