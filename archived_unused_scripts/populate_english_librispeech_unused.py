"""
Add external English clips until data/raw/english reaches CLIPS_PER_LANGUAGE.

Source: LibriSpeech train-clean-100 via Hugging Face.

Usage:
    python populate_english.py
    python populate_english.py --target 10000
    python populate_english.py --min-duration 3 --max-duration 8
"""

import argparse
import random
import sys
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import RAW_DIR
from download_data import CLIPS_PER_LANGUAGE


LANGUAGE = "english"
DATASET_NAME = "openslr/librispeech_asr"
DATASET_CONFIG = "clean"
DEFAULT_SPLIT = "train.100"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def count_existing_clips(lang_dir: Path) -> int:
    return sum(
        1
        for path in lang_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def next_output_path(lang_dir: Path, index: int) -> Path:
    while True:
        path = lang_dir / f"librispeech_english_{index:05d}.wav"
        if not path.exists():
            return path
        index += 1


def save_audio(row: dict, path: Path, min_duration: float, max_duration: float) -> bool:
    audio = row.get("audio")
    if not audio:
        return False

    samples = audio["array"]
    sample_rate = audio["sampling_rate"]
    duration = len(samples) / sample_rate

    if duration < min_duration or duration > max_duration:
        return False

    sf.write(path, samples, sample_rate)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Top up data/raw/english with external English audio."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=CLIPS_PER_LANGUAGE,
        help="Target English clip count. Defaults to CLIPS_PER_LANGUAGE.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.SystemRandom().randint(0, 2**32 - 1),
        help="Random seed used to shuffle source clips.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help="LibriSpeech split to sample from.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum clip duration in seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=20.0,
        help="Maximum clip duration in seconds.",
    )
    args = parser.parse_args()

    if args.min_duration <= 0 or args.max_duration <= args.min_duration:
        raise ValueError("--max-duration must be greater than --min-duration")

    lang_dir = RAW_DIR / LANGUAGE
    lang_dir.mkdir(parents=True, exist_ok=True)

    existing = count_existing_clips(lang_dir)
    needed = max(0, args.target - existing)

    print(f"English clips currently on disk: {existing}")
    print(f"Target English clips: {args.target}")
    print(f"Need to add: {needed}")
    print(f"Keeping clips between {args.min_duration:.1f}s and {args.max_duration:.1f}s")

    if needed == 0:
        print("English already meets the target. Nothing to do.")
        return

    print(
        f"\nLoading {DATASET_NAME} ({DATASET_CONFIG}/{args.split}) "
        f"with seed {args.seed}..."
    )
    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=args.split,
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    added = 0
    skipped = 0
    failures = 0
    output_index = existing + 1

    with tqdm(total=needed, desc="Saving English clips") as bar:
        for row in ds:
            if added >= needed:
                break

            output_path = next_output_path(lang_dir, output_index)
            output_index += 1

            try:
                if save_audio(row, output_path, args.min_duration, args.max_duration):
                    added += 1
                    bar.update(1)
                else:
                    skipped += 1
            except Exception as exc:
                failures += 1
                print(f"\n[WARN] Skipped one clip: {exc}")

    final_count = count_existing_clips(lang_dir)
    print("\nDone.")
    print(f"Added clips: {added}")
    print(f"Skipped clips outside duration range: {skipped}")
    print(f"Failed clips: {failures}")
    print(f"Final English clip count: {final_count}")

    if final_count < args.target:
        print(
            "Warning: source dataset ended before reaching the target. "
            "Try a wider duration range or another LibriSpeech split."
        )


if __name__ == "__main__":
    main()
