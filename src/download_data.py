"""
Automated data acquisition for Spoken Language Identification.

Downloads Common Voice datasets via the Mozilla Data Collective API,
extracts a balanced subset of clips per language, and cleans up archives
to minimize disk usage.

Usage:
    # Download all 7 languages (smallest first)
    python src/download_data.py --api

    # Download only specific languages
    python src/download_data.py --api --languages amharic russian

    # Just verify what's already on disk
    python src/download_data.py --verify

    # Use a local Common Voice extraction instead of the API
    python src/download_data.py --cv-root /path/to/cv-corpus
"""

import csv
import os
import sys
import shutil
import random
import tarfile
import argparse
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, LANGUAGES, SEED, PROJECT_ROOT

random.seed(SEED)

# ── Constants ────────────────────────────────────────────────────────────

load_dotenv(PROJECT_ROOT / ".env")

API_BASE = "https://mozilladatacollective.com/api"
CLIPS_PER_LANGUAGE = 2000
MIN_CLIPS_TO_SKIP = 100

# Dataset IDs for Common Voice Scripted Speech 25.0 (ordered smallest → largest)
DATASET_IDS = {
    "amharic":  "cmn29lq6f0164o10748yd3o7w",   #  58.86 MB
    "hindi":    "cmn2cxzy701iumm077t5ayw0e",   # 544.38 MB
    "russian":  "cmn2h1dg201gro107lpynbbd6",   #   6.55 GB
    "italian":  "cmn2h0yei01msmm07u8z5vu87",   #   9.71 GB
    "english":  "cmn2hx8i401hjmm07ywky5h3f",   #  14.06 GB
    "spanish":  "cmn2hz47s01j8mm076t0a12b3",   #  14.46 GB
    "french":   "cmn2i1f0g01qkmm07qgz1nmyz",   #  14.68 GB
    "german":   "cmn2i26o201rmmm07c6e09c6a",   #  14.86 GB
}

LOCALE_MAP = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ru": "russian",
    "am": "amharic",
}

DATASET_SIZES = {
    "amharic":  "58.86 MB",
    "hindi":    "544.38 MB",
    "russian":  "6.55 GB",
    "italian":  "9.71 GB",
    "english":  "14.06 GB",
    "spanish":  "14.46 GB",
    "french":   "14.68 GB",
    "german":   "14.86 GB",
}


# ── API helpers ──────────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.environ.get("MDC_API_KEY", "")
    if not key:
        print("ERROR: MDC_API_KEY not found.")
        print("Create a .env file in the project root with:")
        print("  MDC_API_KEY=your_key_here")
        sys.exit(1)
    return "5d159acfca54affcb1cb3419d0e5a901843b07ee5dc85e3b8e491e89f3bff0d4"


def _get_download_url(dataset_id: str, api_key: str) -> dict:
    """POST to the API to get a presigned download URL."""
    url = f"{API_BASE}/datasets/{dataset_id}/download"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, timeout=30)

    if resp.status_code != 200:
        try:
            body = resp.json()
        except Exception:
            body = resp.text[:300]
        print(f"  [DEBUG] HTTP {resp.status_code}: {body}")

        if resp.status_code == 403:
            return {"error": "terms_not_accepted"}
        if resp.status_code == 401:
            return {"error": "auth_failed"}
        if resp.status_code == 429:
            return {"error": "rate_limited"}
        return {"error": f"http_{resp.status_code}"}

    return resp.json()


def _download_file(url: str, dest: Path) -> bool:
    """Stream-download a file with a progress bar. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Download failed: {e}")
        return False

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"  Downloading", leave=True,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))
    return True


# ── Extraction & subsampling ─────────────────────────────────────────────

def _find_clips_in_extracted(extract_dir: Path):
    """Locate the clips directory and validated.tsv inside an extracted
    Common Voice archive. Returns (clips_dir, tsv_path_or_None)."""
    for root, dirs, files in os.walk(extract_dir):
        root_path = Path(root)
        if root_path.name == "clips" and any(
            f.endswith(".mp3") for f in os.listdir(root_path)[:20]
        ):
            tsv = root_path.parent / "validated.tsv"
            return root_path, tsv if tsv.exists() else None
    return None, None


def _subsample_clips(clips_dir: Path, tsv_path, dest: Path, limit: int):
    """Pick up to *limit* validated clips and copy them to *dest*."""
    audio_files = []

    if tsv_path and Path(tsv_path).exists():
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                p = clips_dir / row["path"]
                if p.exists():
                    audio_files.append(p)
    else:
        audio_files = list(clips_dir.glob("*.mp3"))

    if not audio_files:
        return 0

    random.shuffle(audio_files)
    subset = audio_files[:limit]

    dest.mkdir(parents=True, exist_ok=True)
    for src in subset:
        shutil.copy2(src, dest / src.name)

    return len(subset)


# ── Main download pipeline ───────────────────────────────────────────────

def _count_clips(lang: str) -> int:
    lang_dir = RAW_DIR / lang
    if not lang_dir.exists():
        return 0
    return (
        len(list(lang_dir.glob("*.mp3")))
        + len(list(lang_dir.glob("*.wav")))
        + len(list(lang_dir.glob("*.flac")))
    )


def download_language(lang: str, api_key: str) -> bool:
    """Download and extract clips for a single language. Returns True on success."""
    dataset_id = DATASET_IDS.get(lang)
    if not dataset_id:
        print(f"  [SKIP] No dataset ID configured for {lang}")
        return False

    existing = _count_clips(lang)
    if existing >= MIN_CLIPS_TO_SKIP:
        print(f"  [SKIP] {lang}: already has {existing} clips")
        return True

    size_str = DATASET_SIZES.get(lang, "unknown size")
    print(f"\n{'='*60}")
    print(f"  {lang.upper()} ({size_str})")
    print(f"{'='*60}")

    # Step 1: Get presigned download URL
    print("  Requesting download URL...")
    result = _get_download_url(dataset_id, api_key)

    if "error" in result:
        err = result["error"]
        if err == "terms_not_accepted":
            print(f"  [ERROR] You must agree to the terms of use for the")
            print(f"          {lang.capitalize()} dataset on the Mozilla Data")
            print(f"          Collective website before downloading via API.")
            print(f"          https://datacollective.mozillafoundation.org/datasets/{dataset_id}")
        elif err == "auth_failed":
            print("  [ERROR] API key is invalid. Check your .env file.")
        elif err == "rate_limited":
            print("  [ERROR] Rate limited. Try again later (limit: 30 downloads/day).")
        else:
            print(f"  [ERROR] API returned: {err}")
        return False

    download_url = result.get("downloadUrl")
    if not download_url:
        print("  [ERROR] No download URL in API response")
        return False

    # Step 2: Download archive to a temp file
    with tempfile.TemporaryDirectory(prefix=f"cv_{lang}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        filename = result.get("filename", f"{lang}.tar.gz")
        archive_path = tmp_path / filename

        print(f"  Downloading {filename}...")
        if not _download_file(download_url, archive_path):
            return False

        # Step 3: Extract
        print("  Extracting archive (this may take a while)...")
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        try:
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=extract_dir, filter="data")
        except (tarfile.TarError, EOFError) as e:
            print(f"  [ERROR] Extraction failed: {e}")
            return False

        # Delete archive immediately to free disk space
        archive_path.unlink()
        print("  Archive deleted to free space.")

        # Step 4: Find clips and subsample
        clips_dir, tsv_path = _find_clips_in_extracted(extract_dir)
        if clips_dir is None:
            print("  [ERROR] Could not find clips directory in extracted archive")
            return False

        dest = RAW_DIR / lang
        copied = _subsample_clips(clips_dir, tsv_path, dest, CLIPS_PER_LANGUAGE)
        print(f"  Copied {copied} clips to {dest}")

        # tmp_dir auto-cleaned by TemporaryDirectory context manager

    return copied > 0


def download_all(api_key: str, languages=None):
    """Download datasets for all (or specified) languages."""
    target_langs = languages if languages else list(DATASET_IDS.keys())

    print("=" * 60)
    print("  MOZILLA DATA COLLECTIVE — AUTOMATED DOWNLOAD")
    print("=" * 60)
    print(f"\n  Languages: {', '.join(l.capitalize() for l in target_langs)}")
    print(f"  Clips per language: {CLIPS_PER_LANGUAGE}")
    print(f"  Destination: {RAW_DIR}\n")

    results = {}
    for lang in target_langs:
        if lang not in DATASET_IDS:
            print(f"\n  [SKIP] {lang}: no dataset ID configured")
            results[lang] = False
            continue
        results[lang] = download_language(lang, api_key)

    # Summary
    print(f"\n{'='*60}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for lang, ok in results.items():
        count = _count_clips(lang)
        status = f"{count} clips" if ok else "FAILED"
        print(f"  {lang:10s}  {status}")

    verify_data()


# ── Legacy helpers (kept for backward compatibility) ─────────────────────

def organise_from_cv_root(cv_root: Path):
    """Copy a balanced subset from a locally extracted Common Voice corpus."""
    cv_root = Path(cv_root)
    for locale, lang in LOCALE_MAP.items():
        clips_dir = cv_root / locale / "clips"
        tsv_path = cv_root / locale / "validated.tsv"

        if not clips_dir.exists():
            alt = cv_root / "clips"
            if alt.exists():
                clips_dir = alt
            else:
                print(f"[SKIP] {clips_dir} not found")
                continue

        dest = RAW_DIR / lang
        copied = _subsample_clips(clips_dir, tsv_path, dest, CLIPS_PER_LANGUAGE)
        print(f"[OK]   {lang}: copied {copied} clips to {dest}")


def verify_data():
    """Print a summary of how many clips are available per language."""
    print("\nData verification:")
    print("-" * 40)
    all_ok = True
    for lang in LANGUAGES:
        count = _count_clips(lang)
        status = "OK" if count >= 100 else "LOW" if count > 0 else "EMPTY"
        if status != "OK":
            all_ok = False
        print(f"  {lang:10s}  {count:5d} clips  [{status}]")

    if all_ok:
        print("\nAll languages have sufficient data. Ready to train!")
    else:
        print("\nSome languages need more clips. Run with --api to download.")


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare audio data for language identification."
    )
    parser.add_argument(
        "--api", action="store_true", default=False,
        help="Download datasets via Mozilla Data Collective API (default action)",
    )
    parser.add_argument(
        "--languages", nargs="+", type=str, default=None,
        help="Download only these languages (e.g. --languages amharic russian)",
    )
    parser.add_argument(
        "--cv-root", type=str, default=None,
        help="Path to a locally extracted Common Voice corpus",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Just verify existing data and exit",
    )
    args = parser.parse_args()

    if args.verify:
        verify_data()
    elif args.cv_root:
        organise_from_cv_root(Path(args.cv_root))
        verify_data()
    elif args.api or args.languages:
        api_key = _get_api_key()
        langs = [l.lower() for l in args.languages] if args.languages else None
        download_all(api_key, languages=langs)
    else:
        # Default: show help and current status
        parser.print_help()
        print()
        verify_data()
