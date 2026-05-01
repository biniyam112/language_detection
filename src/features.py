"""
Audio loading and feature extraction utilities.

Pipeline:
    Audio file  ->  load & resample to 16 kHz
                ->  pad / truncate to fixed length
                ->  Mel-spectrogram  (128 x T)  ->  log scale  ->  normalise
                ->  (optional) MFCC extraction
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np

from config import (
    SAMPLE_RATE,
    NUM_SAMPLES,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    N_MFCC,
)

# Pre-built transforms (created once, reused)
_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)

_mfcc_transform = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": N_MELS},
)


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load an audio file, convert to mono, and resample to *target_sr*.

    Returns a 1-D tensor of shape ``(num_samples,)``.
    """
    waveform, sr = torchaudio.load(path)

    # Convert to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0)  # (num_samples,)


def pad_or_truncate(waveform: torch.Tensor, length: int = NUM_SAMPLES) -> torch.Tensor:
    """Ensure waveform is exactly *length* samples (pad with zeros or truncate)."""
    if waveform.shape[0] > length:
        return waveform[:length]
    if waveform.shape[0] < length:
        padding = torch.zeros(length - waveform.shape[0])
        return torch.cat([waveform, padding])
    return waveform


def extract_mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    """Compute a log-scaled Mel-spectrogram.

    Args:
        waveform: 1-D tensor of shape ``(num_samples,)``.

    Returns:
        Tensor of shape ``(1, n_mels, time_frames)`` suitable for a 2-D CNN.
    """
    waveform = pad_or_truncate(waveform)
    mel = _mel_transform(waveform)  # (n_mels, T)
    mel_db = T.AmplitudeToDB()(mel)

    # Normalise to zero mean / unit variance per spectrogram
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

    return mel_db.unsqueeze(0)  # (1, n_mels, T)


def extract_mfcc(waveform: torch.Tensor) -> torch.Tensor:
    """Compute MFCCs from a waveform.

    Returns:
        Tensor of shape ``(n_mfcc, time_frames)``.
    """
    waveform = pad_or_truncate(waveform)
    mfcc = _mfcc_transform(waveform)  # (n_mfcc, T)
    return mfcc


def audio_to_mel(path: str) -> torch.Tensor:
    """Convenience: load audio file and return a normalised Mel-spectrogram."""
    waveform = load_audio(path)
    return extract_mel_spectrogram(waveform)


def audio_to_mfcc(path: str) -> np.ndarray:
    """Convenience: load audio file and return MFCCs as a numpy array."""
    waveform = load_audio(path)
    return extract_mfcc(waveform).numpy()
