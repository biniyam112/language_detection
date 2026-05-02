"""
CNN model for spoken language identification.

Architecture
------------
Input:  (batch, 1, 128, T)  -- single-channel Mel-spectrogram

5 residual double-conv blocks, each with:
    Conv2d(3x3) -> BN -> ReLU -> Conv2d(3x3) -> BN -> (+residual) -> ReLU -> Pool

Block 1:   1 -> 32,  MaxPool(2)
Block 2:  32 -> 64,  MaxPool(2)
Block 3:  64 -> 128, MaxPool(2)
Block 4: 128 -> 256, MaxPool(2)
Block 5: 256 -> 512, AdaptiveAvgPool(1)

Classifier:
    Flatten -> Linear(512,256) -> ReLU -> Dropout(0.4)
           -> Linear(256,128) -> ReLU -> Dropout(0.3)
           -> Linear(128, num_classes)
"""

import torch
import torch.nn as nn

from config import NUM_CLASSES


class ConvBlock(nn.Module):
    """Double Conv2d with a residual skip connection, followed by pooling."""

    def __init__(self, in_ch: int, out_ch: int, pool: nn.Module):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = pool

        # 1x1 conv to match channel dimensions for the residual path
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + identity)
        return self.pool(out)


class LanguageCNN(nn.Module):
    """Deeper residual CNN for language classification from Mel-spectrograms."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32, nn.MaxPool2d(2)),
            ConvBlock(32, 64, nn.MaxPool2d(2)),
            ConvBlock(64, 128, nn.MaxPool2d(2)),
            ConvBlock(128, 256, nn.MaxPool2d(2)),
            ConvBlock(256, 512, nn.AdaptiveAvgPool2d(1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),          # 0
            nn.Linear(512, 256),   # 1
            nn.ReLU(inplace=True), # 2
            nn.Dropout(0.4),       # 3
            nn.Linear(256, 128),   # 4
            nn.ReLU(inplace=True), # 5
            nn.Dropout(0.3),       # 6
            nn.Linear(128, num_classes),  # 7
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return 128-d penultimate-layer embeddings for PCA visualisation."""
        x = self.features(x)
        x = self.classifier[0](x)  # Flatten
        x = self.classifier[1](x)  # Linear(512, 256)
        x = self.classifier[2](x)  # ReLU
        x = self.classifier[3](x)  # Dropout
        x = self.classifier[4](x)  # Linear(256, 128)
        x = self.classifier[5](x)  # ReLU
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
