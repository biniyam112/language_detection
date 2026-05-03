"""
Precompute Mel-spectrogram tensors for faster training.

Run once after downloading data:
    python src/precompute_features.py

Training will then load cached .pt tensors from data/processed/mel_spectrograms/
instead of decoding audio and computing features every epoch.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import _discover_files, cache_path_for_audio
from features import load_audio, extract_mel_spectrogram, prepare_waveform


def precompute_one(audio_path: str, overwrite: bool = False) -> tuple[str, bool, str]:
    """Precompute one Mel-spectrogram.

    Returns:
        (audio_path, success, message)
    """
    cache_path = cache_path_for_audio(audio_path)
    if cache_path.exists() and not overwrite:
        return audio_path, True, "cached"

    try:
        waveform = load_audio(audio_path)
        waveform = prepare_waveform(waveform, training=False)
        mel = extract_mel_spectrogram(waveform)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mel, cache_path)
        return audio_path, True, "created"
    except Exception as exc:
        return audio_path, False, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Precompute Mel-spectrogram cache")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker threads")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute files even if cached tensors exist")
    args = parser.parse_args()

    paths, _ = _discover_files()
    if not paths:
        raise RuntimeError("No audio files found in data/raw/")

    created = 0
    cached = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(precompute_one, path, args.overwrite)
            for path in paths
        ]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Precomputing"):
            _, ok, message = future.result()
            if ok and message == "created":
                created += 1
            elif ok and message == "cached":
                cached += 1
            else:
                failed += 1

    print("\nPrecompute summary")
    print("-" * 40)
    print(f"Created: {created}")
    print(f"Cached:  {cached}")
    print(f"Failed:  {failed}")


if __name__ == "__main__":
    main()
