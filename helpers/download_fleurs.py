"""
Download FLEURS training audio for the active language list.

This adds a second audio domain to improve cross-dataset robustness:

    python helpers/download_fleurs.py

Files are saved under:

    data/raw/{language}/fleurs/{language}_fleurs_00001.wav
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import FLEURS_CLIPS_PER_LANGUAGE, LANGUAGES, RAW_DIR, SEED


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


def existing_count(lang: str) -> int:
    out_dir = RAW_DIR / lang / "fleurs"
    if not out_dir.exists():
        return 0
    return len(list(out_dir.glob("*.wav")))


def download_language(lang: str, limit: int, overwrite: bool = False) -> int:
    code = FLEURS_LANGS.get(lang)
    if code is None:
        print(f"[SKIP] {lang}: no FLEURS code configured")
        return 0

    out_dir = RAW_DIR / lang / "fleurs"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = existing_count(lang)
    if existing >= limit and not overwrite:
        print(f"[SKIP] {lang}: already has {existing} FLEURS clips")
        return existing

    if overwrite:
        for wav in out_dir.glob("*.wav"):
            wav.unlink()

    print(f"\nDownloading {lang} FLEURS train split ({code})...")
    ds = load_dataset(
        "google/fleurs",
        code,
        split="train",
        trust_remote_code=True,
    )
    n = min(limit, len(ds))
    ds = ds.shuffle(seed=SEED).select(range(n))

    for i, row in enumerate(tqdm(ds, desc=f"{lang}"), start=1):
        audio = row["audio"]
        path = out_dir / f"{lang}_fleurs_{i:05d}.wav"
        sf.write(path, audio["array"], audio["sampling_rate"])

    return n


def main():
    parser = argparse.ArgumentParser(description="Download FLEURS training clips")
    parser.add_argument("--languages", nargs="+", default=LANGUAGES,
                        help="Languages to download; defaults to active LANGUAGES")
    parser.add_argument("--limit", type=int, default=FLEURS_CLIPS_PER_LANGUAGE,
                        help="Max FLEURS clips per language")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing FLEURS clips")
    args = parser.parse_args()

    total = 0
    for lang in args.languages:
        total += download_language(lang.lower(), args.limit, args.overwrite)

    print("\nFLEURS download summary")
    print("-" * 40)
    for lang in args.languages:
        print(f"{lang.lower():12s} {existing_count(lang.lower()):5d} clips")
    print("-" * 40)
    print(f"{'Total':12s} {total:5d} clips processed")


if __name__ == "__main__":
    main()
