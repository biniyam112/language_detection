import argparse
import random
import shutil
import sys
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import LANGUAGES, RAW_DIR
from dataset import get_splits


FLEURS_LANGS = {
    "amharic": "am_et",
    "arabic": "ar_eg",
    "chinese": "cmn_hans_cn",
    "english": "en_us",
    "french": "fr_fr",
    "hindi": "hi_in",
    "italian": "it_it",
    "russian": "ru_ru",
    "spanish": "es_419",
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
        by_lang[LANGUAGES[label]].append(Path(path))

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


def generate_source_samples(out_dir: Path, samples_per_language: int, seed: int,
                            source: str):
    """Copy samples from a specific local source folder.

    Supported source values:
      - mozilla: files directly under data/raw/{language}/, excluding fleurs/
      - fleurs_local: files under data/raw/{language}/fleurs/
      - auxiliary: currently matches non-FLEURS files with names commonly used
        by auxiliary population scripts, such as alffa_* and librispeech_*.
    """
    rng = random.Random(seed)
    audio_exts = {".mp3", ".wav", ".flac", ".ogg"}
    auxiliary_prefixes = ("alffa_", "librispeech_")

    for lang in LANGUAGES:
        lang_dir = RAW_DIR / lang
        if source == "fleurs_local":
            candidates = [
                p for p in (lang_dir / "fleurs").rglob("*")
                if p.is_file() and p.suffix.lower() in audio_exts
            ]
        else:
            candidates = [
                p for p in lang_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in audio_exts
                and "fleurs" not in [part.lower() for part in p.relative_to(lang_dir).parts]
            ]
            if source == "auxiliary":
                candidates = [
                    p for p in candidates
                    if p.name.lower().startswith(auxiliary_prefixes)
                ]
            elif source == "mozilla":
                candidates = [
                    p for p in candidates
                    if not p.name.lower().startswith(auxiliary_prefixes)
                ]

        rng.shuffle(candidates)
        selected = candidates[:samples_per_language]
        lang_out = out_dir / lang
        lang_out.mkdir(parents=True, exist_ok=True)

        print(f"Copying {len(selected)} {source} samples for {lang}...")
        for i, src in enumerate(selected, start=1):
            dest = lang_out / f"{lang}_{source}_{i}{src.suffix.lower()}"
            shutil.copy2(src, dest)
            print(" ", dest)


def main():
    parser = argparse.ArgumentParser(description="Generate sample prediction audio")
    parser.add_argument(
        "--source",
        choices=["local", "fleurs", "mozilla", "fleurs_local", "auxiliary"],
        default="local",
        help=(
            "local = held-out project test split; fleurs = external FLEURS; "
            "mozilla/fleurs_local/auxiliary = local source folders"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("samples/predict_test"))
    parser.add_argument("--samples-per-language", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**32 - 1)
    reset_output_dir(args.out_dir)

    if args.source == "local":
        generate_local_samples(args.out_dir, args.samples_per_language, seed)
    elif args.source == "fleurs":
        generate_fleurs_samples(args.out_dir, args.samples_per_language, seed)
    else:
        generate_source_samples(args.out_dir, args.samples_per_language, seed, args.source)

    print(f"\nGenerated samples in {args.out_dir}")
    print(f"Source: {args.source}")
    print(f"Seed: {seed}")


if __name__ == "__main__":
    main()