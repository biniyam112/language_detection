"""
Audio loading and feature extraction utilities.

Pipeline:
    Audio file  ->  load & resample to 16 kHz
                ->  pad / truncate to fixed length
                ->  Mel-spectrogram  (128 x T)  ->  log scale  ->  normalise
                ->  (optional) MFCC extraction
"""

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
)

def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load an audio file, convert to mono, and resample to *target_sr*.

    Uses librosa which handles MP3, WAV, FLAC, OGG out of the box.
    Returns a 1-D tensor of shape ``(num_samples,)``.
    """
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return torch.from_numpy(y).float()


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
    waveform = pad_or_truncate(waveform)
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
    return extract_mel_spectrogram(waveform)


def audio_to_mfcc(path: str) -> np.ndarray:
    """Convenience: load audio file and return MFCCs as a numpy array."""
    waveform = load_audio(path)
    return extract_mfcc(waveform).numpy()
