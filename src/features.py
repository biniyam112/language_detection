"""
Audio loading and feature extraction utilities.

Pipeline:
    Audio file  ->  load & resample to 16 kHz
                ->  pad / truncate to fixed length
                ->  Mel-spectrogram  (128 x T)  ->  log scale  ->  normalise
                ->  (optional) MFCC extraction
"""

import random

import librosa
import numpy as np
import torch

from config import (
    SAMPLE_RATE,
    NUM_SAMPLES,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    N_MFCC,
    USE_VAD_TRIM,
)

def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load an audio file, convert to mono, and resample to *target_sr*.

    Uses librosa which handles MP3, WAV, FLAC, OGG out of the box.
    Returns a 1-D tensor of shape ``(num_samples,)``.
    """
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return torch.from_numpy(y).float()


def trim_silence(waveform: torch.Tensor, top_db: int = 30) -> torch.Tensor:
    """Remove leading/trailing silence while preserving non-empty clips."""
    if not USE_VAD_TRIM:
        return waveform

    y = waveform.detach().cpu().numpy()
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if trimmed.size == 0:
        return waveform
    return torch.from_numpy(trimmed).float()


def pad_or_truncate(waveform: torch.Tensor, length: int = NUM_SAMPLES) -> torch.Tensor:
    """Ensure waveform is exactly *length* samples (pad with zeros or truncate)."""
    if waveform.shape[0] > length:
        return waveform[:length]
    if waveform.shape[0] < length:
        padding = torch.zeros(length - waveform.shape[0])
        return torch.cat([waveform, padding])
    return waveform


def center_crop_or_pad(waveform: torch.Tensor, length: int = NUM_SAMPLES) -> torch.Tensor:
    """Deterministically center-crop or pad a waveform to *length* samples."""
    n = waveform.shape[0]
    if n > length:
        start = (n - length) // 2
        return waveform[start:start + length]
    if n < length:
        total_pad = length - n
        left = total_pad // 2
        right = total_pad - left
        return torch.nn.functional.pad(waveform, (left, right))
    return waveform


def random_crop_or_pad(waveform: torch.Tensor, length: int = NUM_SAMPLES) -> torch.Tensor:
    """Randomly crop long clips and randomly place short clips within silence."""
    n = waveform.shape[0]
    if n > length:
        start = random.randint(0, n - length)
        return waveform[start:start + length]
    if n < length:
        total_pad = length - n
        left = random.randint(0, total_pad)
        right = total_pad - left
        return torch.nn.functional.pad(waveform, (left, right))
    return waveform


def augment_waveform(waveform: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Apply lightweight waveform augmentation for training robustness."""
    y = waveform.detach().cpu().numpy().astype(np.float32)

    # Random gain: simulate speaker/microphone loudness differences.
    gain = random.uniform(0.7, 1.3)
    y = y * gain

    # Low-level background noise.
    if random.random() < 0.5:
        signal_std = float(np.std(y)) if np.std(y) > 1e-6 else 0.01
        noise = np.random.normal(0.0, signal_std * random.uniform(0.003, 0.02), size=y.shape)
        y = y + noise.astype(np.float32)

    # Pitch shift in semitones.
    if random.random() < 0.3:
        steps = random.uniform(-1.5, 1.5)
        y = librosa.effects.pitch_shift(y=y, sr=sample_rate, n_steps=steps)

    # Time stretch while keeping final model input length fixed later.
    if random.random() < 0.3:
        rate = random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y=y, rate=rate)

    y = np.clip(y, -1.0, 1.0)
    return torch.from_numpy(y).float()


def prepare_waveform(waveform: torch.Tensor, training: bool = False) -> torch.Tensor:
    """Full waveform preprocessing before Mel extraction."""
    waveform = trim_silence(waveform)
    if training:
        waveform = random_crop_or_pad(waveform)
        waveform = augment_waveform(waveform)
        waveform = random_crop_or_pad(waveform)
    else:
        waveform = center_crop_or_pad(waveform)
    return waveform


def extract_mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    """Compute a log-scaled Mel-spectrogram.

    Args:
        waveform: 1-D tensor of shape ``(num_samples,)``.

    Returns:
        Tensor of shape ``(1, n_mels, time_frames)`` suitable for a 2-D CNN.
    """
    waveform = center_crop_or_pad(waveform)
    y = waveform.detach().cpu().numpy()
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = torch.from_numpy(mel_db).float()

    # Normalise to zero mean / unit variance per spectrogram
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

    return mel_db.unsqueeze(0)  # (1, n_mels, T)


def extract_mfcc(waveform: torch.Tensor) -> torch.Tensor:
    """Compute MFCCs from a waveform.

    Returns:
        Tensor of shape ``(n_mfcc, time_frames)``.
    """
    waveform = center_crop_or_pad(waveform)
    y = waveform.detach().cpu().numpy()
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    return torch.from_numpy(mfcc).float()


def audio_to_mel(path: str) -> torch.Tensor:
    """Convenience: load audio file and return a normalised Mel-spectrogram."""
    waveform = load_audio(path)
    waveform = prepare_waveform(waveform, training=False)
    return extract_mel_spectrogram(waveform)


def audio_to_mfcc(path: str) -> np.ndarray:
    """Convenience: load audio file and return MFCCs as a numpy array."""
    waveform = load_audio(path)
    waveform = prepare_waveform(waveform, training=False)
    return extract_mfcc(waveform).numpy()
