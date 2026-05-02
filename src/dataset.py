"""
PyTorch Dataset and DataLoader utilities for spoken language identification.

Handles:
  - Discovering audio files from data/raw/{language}/ directories
  - Stratified train / val / test splitting
  - On-the-fly Mel-spectrogram extraction
  - SpecAugment-style data augmentation (time & frequency masking)
"""

import random
from pathlib import Path
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import (
    CACHE_FEATURES,
    FEATURE_CACHE_DIR,
    RAW_DIR,
    LANGUAGES,
    LANG_TO_IDX,
    BATCH_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
    N_MELS,
)
from features import load_audio, extract_mel_spectrogram


# ── SpecAugment helpers ──────────────────────────────────────────────────

def _frequency_mask(spec: torch.Tensor, max_width: int = 15) -> torch.Tensor:
    """Zero out a random horizontal band of the spectrogram."""
    _, n_mels, _ = spec.shape
    width = random.randint(1, max_width)
    start = random.randint(0, max(0, n_mels - width))
    spec[:, start : start + width, :] = 0.0
    return spec


def _time_mask(spec: torch.Tensor, max_width: int = 25) -> torch.Tensor:
    """Zero out a random vertical band of the spectrogram."""
    _, _, n_frames = spec.shape
    width = random.randint(1, max_width)
    start = random.randint(0, max(0, n_frames - width))
    spec[:, :, start : start + width] = 0.0
    return spec


# ── File discovery ───────────────────────────────────────────────────────

def _discover_files() -> Tuple[List[str], List[int]]:
    """Walk data/raw/{language}/ and return (file_paths, labels)."""
    paths: List[str] = []
    labels: List[int] = []

    for lang in LANGUAGES:
        lang_dir = RAW_DIR / lang
        if not lang_dir.exists():
            continue
        for ext in ("*.mp3", "*.wav", "*.flac", "*.ogg"):
            for f in lang_dir.glob(ext):
                paths.append(str(f))
                labels.append(LANG_TO_IDX[lang])

    return paths, labels


def cache_path_for_audio(audio_path: str) -> Path:
    """Return the cached Mel-spectrogram path for an audio file."""
    p = Path(audio_path)
    lang = p.parent.name
    return FEATURE_CACHE_DIR / lang / f"{p.stem}.pt"


def load_or_create_mel(audio_path: str) -> torch.Tensor:
    """Load a cached Mel-spectrogram, or compute and cache it on first use."""
    cache_path = cache_path_for_audio(audio_path)

    if CACHE_FEATURES and cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    waveform = load_audio(audio_path)
    mel = extract_mel_spectrogram(waveform)

    if CACHE_FEATURES:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mel, cache_path)

    return mel


# ── Dataset class ────────────────────────────────────────────────────────

class LanguageDataset(Dataset):
    """Loads audio on the fly, extracts Mel-spectrograms, and optionally
    applies SpecAugment augmentation."""

    def __init__(self, file_paths: List[str], labels: List[int],
                 augment: bool = False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            mel = load_or_create_mel(path)  # (1, n_mels, T)
        except Exception:
            mel = torch.zeros(1, N_MELS, 157)  # fallback for corrupt files
            label = self.labels[idx]

        if self.augment:
            mel = _frequency_mask(mel.clone())
            mel = _time_mask(mel)

        return mel, label


# ── Splitting & DataLoader factory ───────────────────────────────────────

def get_splits() -> Dict[str, Tuple[List[str], List[int]]]:
    """Return stratified train/val/test splits as a dict of
    ``{split_name: (file_paths, labels)}``."""
    paths, labels = _discover_files()

    if len(paths) == 0:
        raise RuntimeError(
            "No audio files found. Run `python src/download_data.py` first."
        )

    val_test_ratio = VAL_RATIO + TEST_RATIO
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=SEED,
    )

    relative_test = TEST_RATIO / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=SEED,
    )

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def get_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = 0
                    ) -> Dict[str, DataLoader]:
    """Build DataLoaders for train (with augmentation), val, and test."""
    splits = get_splits()

    loaders = {}
    for split_name, (paths, labels) in splits.items():
        augment = split_name == "train"
        ds = LanguageDataset(paths, labels, augment=augment)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    print(f"Dataset sizes — "
          f"train: {len(splits['train'][0])}, "
          f"val: {len(splits['val'][0])}, "
          f"test: {len(splits['test'][0])}")

    return loaders
