import argparse
import shutil
import sys
from pathlib import Path
import random

import soundfile as sf
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import LANGUAGES
from dataset import get_splits


FLEURS_LANGS = {
    "italian": "it_it",
    "russian": "ru_ru",
    "amharic": "am_et",
    "hindi": "hi_in",
    "chinese": "cmn_hans_cn",
    "arabic": "ar_eg",
    "french": "fr_fr",
    "spanish": "es_419",
    "english": "en_us",
}


def reset_output_dir(out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def generate_local_samples(out_dir: Path, samples_per_language: int, seed: int):
    """Copy samples from the same held-out test split used by evaluate.py."""
    splits = get_splits()
    test_paths, test_labels = splits["test"]

    by_lang = {lang: [] for lang in LANGUAGES}
    for path, label in zip(test_paths, test_labels):
        lang = LANGUAGES[label]
        by_lang[lang].append(Path(path))

    rng = random.Random(seed)
    for lang, paths in by_lang.items():
        rng.shuffle(paths)
        selected = paths[:samples_per_language]
        lang_dir = out_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        print(f"Copying {len(selected)} local held-out samples for {lang}...")
        for i, src in enumerate(selected, start=1):
            dest = lang_dir / f"{lang}_{i}{src.suffix.lower()}"
            shutil.copy2(src, dest)
            print(" ", dest)


def generate_fleurs_samples(out_dir: Path, samples_per_language: int, seed: int):
    """Download FLEURS examples. This is an out-of-domain stress test."""
    for lang in LANGUAGES:
        code = FLEURS_LANGS.get(lang)
        if code is None:
            print(f"Skipping {lang}: no FLEURS code configured")
            continue

        print(f"Downloading {lang} FLEURS samples with seed {seed}...")
        ds = load_dataset(
            "google/fleurs",
            code,
            split="test",
            trust_remote_code=True,
        )
        ds = ds.shuffle(seed=seed).select(range(samples_per_language))

        lang_dir = out_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        for i, row in enumerate(ds, start=1):
            audio = row["audio"]
            path = lang_dir / f"{lang}_{i}.wav"
            sf.write(path, audio["array"], audio["sampling_rate"])
            print(" ", path)


def main():
    parser = argparse.ArgumentParser(description="Generate sample prediction audio")
    parser.add_argument("--source", choices=["local", "fleurs"], default="local",
                        help="local = held-out project test split; fleurs = external stress test")
    parser.add_argument("--out-dir", type=Path, default=Path("samples/predict_test"))
    parser.add_argument("--samples-per-language", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**32 - 1)
    reset_output_dir(args.out_dir)

    if args.source == "local":
        generate_local_samples(args.out_dir, args.samples_per_language, seed)
    else:
        generate_fleurs_samples(args.out_dir, args.samples_per_language, seed)

    print(f"\nGenerated samples in {args.out_dir}")
    print(f"Source: {args.source}")
    print(f"Seed: {seed}")


if __name__ == "__main__":
    main()