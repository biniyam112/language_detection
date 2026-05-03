"""
Add external Amharic clips until data/raw/amharic reaches MOZILLA_CLIPS_PER_LANGUAGE.

Source: ALFFA Amharic via Hugging Face.

Usage:
    python helpers/populate_amharic.py
    python helpers/populate_amharic.py --target 10000
"""

import argparse
import random
import sys
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import MOZILLA_CLIPS_PER_LANGUAGE, RAW_DIR, SEED


LANGUAGE = "amharic"
DATASET_NAME = "hadamard-2/alffa-amharic-v2"
DEFAULT_SPLIT = "train"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def count_existing_clips(lang_dir: Path) -> int:
    return sum(
        1
        for path in lang_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def next_output_path(lang_dir: Path, index: int) -> Path:
    while True:
        path = lang_dir / f"alffa_amharic_{index:05d}.wav"
        if not path.exists():
            return path
        index += 1


def save_audio(row: dict, path: Path) -> bool:
    audio = row.get("audio")
    if not audio:
        return False

    sf.write(path, audio["array"], audio["sampling_rate"])
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Top up data/raw/amharic with external Amharic audio."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=MOZILLA_CLIPS_PER_LANGUAGE,
        help="Target Amharic clip count. Defaults to MOZILLA_CLIPS_PER_LANGUAGE.",
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
        help="Dataset split to sample from.",
    )
    args = parser.parse_args()

    lang_dir = RAW_DIR / LANGUAGE
    lang_dir.mkdir(parents=True, exist_ok=True)

    existing = count_existing_clips(lang_dir)
    needed = max(0, args.target - existing)

    print(f"Amharic clips currently on disk: {existing}")
    print(f"Target Amharic clips: {args.target}")
    print(f"Need to add: {needed}")

    if needed == 0:
        print("Amharic already meets the target. Nothing to do.")
        return

    print(f"\nLoading {DATASET_NAME} ({args.split}) with seed {args.seed}...")
    ds = load_dataset(
        DATASET_NAME,
        split=args.split,
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    added = 0
    failures = 0
    output_index = existing + 1

    for row in tqdm(ds, total=needed, desc="Saving Amharic clips"):
        if added >= needed:
            break

        output_path = next_output_path(lang_dir, output_index)
        output_index += 1

        try:
            if save_audio(row, output_path):
                added += 1
            else:
                failures += 1
        except Exception as exc:
            failures += 1
            print(f"\n[WARN] Skipped one clip: {exc}")

    final_count = count_existing_clips(lang_dir)
    print("\nDone.")
    print(f"Added clips: {added}")
    print(f"Skipped clips: {failures}")
    print(f"Final Amharic clip count: {final_count}")

    if final_count < args.target:
        print(
            "Warning: source dataset ended before reaching the target. "
            "Run again with another source if needed."
        )


if __name__ == "__main__":
    main()
