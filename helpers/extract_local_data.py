"""
Extract a fixed number of audio clips from one local Common Voice .tar.gz file.

This script is intentionally local-only:
  - no Mozilla API calls
  - no Hugging Face datasets
  - no network access

Example:
    python helpers/extract_local_data.py --archive /path/to/english.tar.gz --language english
"""

import argparse
import csv
import io
import os
import random
import sys
import tarfile
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import LANGUAGES, MOZILLA_CLIPS_PER_LANGUAGE, RAW_DIR, SEED


DEFAULT_LIMIT = MOZILLA_CLIPS_PER_LANGUAGE
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg")

random.seed(SEED)


def is_audio_member(member: tarfile.TarInfo) -> bool:
    """Return True for audio files inside the archive."""
    return member.isfile() and member.name.lower().endswith(AUDIO_EXTENSIONS)


def destination_path(dest_dir: Path, member_name: str) -> Path:
    """Flatten archive paths so clips land directly in data/raw/{language}."""
    return dest_dir / os.path.basename(member_name)


def read_validated_filenames(archive_path: Path) -> list[str]:
    """Read validated.tsv rows from the archive and return validated filenames."""
    filenames: list[str] = []

    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith("validated.tsv"):
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            text = io.TextIOWrapper(extracted, encoding="utf-8")
            reader = csv.DictReader(text, delimiter="\t")
            for row in reader:
                clip_path = row.get("path")
                if clip_path:
                    filenames.append(os.path.basename(clip_path))

    random.shuffle(filenames)
    return filenames


def extract_members(
    archive_path: Path,
    dest_dir: Path,
    members: list[tarfile.TarInfo],
    limit: int,
    extracted_names: set[str],
    overwrite: bool,
    description: str,
) -> int:
    """Extract selected archive members to dest_dir."""
    extracted_count = 0

    with tarfile.open(archive_path, "r:*") as tar:
        member_lookup = {member.name: member for member in members}

        for member_name in tqdm(member_lookup, desc=description):
            if extracted_count >= limit:
                break

            member = member_lookup[member_name]
            basename = os.path.basename(member.name)
            if basename in extracted_names:
                continue

            out_path = destination_path(dest_dir, member.name)
            if out_path.exists() and not overwrite:
                extracted_names.add(basename)
                continue

            source = tar.extractfile(member)
            if source is None:
                continue

            with open(out_path, "wb") as out:
                out.write(source.read())

            extracted_count += 1
            extracted_names.add(basename)

    return extracted_count


def extract_local_archive(
    archive_path: Path,
    language: str,
    limit: int = DEFAULT_LIMIT,
    overwrite: bool = False,
) -> int:
    """Extract up to limit clips from one local Common Voice archive."""
    archive_path = archive_path.expanduser().resolve()
    language = language.lower()

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if language not in LANGUAGES:
        raise ValueError(
            f"Unknown language '{language}'. Expected one of: {', '.join(LANGUAGES)}"
        )
    if limit <= 0:
        raise ValueError("--limit must be greater than 0")

    dest_dir = RAW_DIR / language
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  LOCAL COMMON VOICE EXTRACTION")
    print("=" * 60)
    print(f"Archive:     {archive_path}")
    print(f"Language:    {language}")
    print(f"Destination: {dest_dir}")
    print(f"Target:      {limit} clips")
    print(f"Overwrite:   {overwrite}")

    print("\nScanning for validated.tsv...")
    validated_filenames = read_validated_filenames(archive_path)
    validated_targets = set(validated_filenames[:limit])

    if validated_filenames:
        print(f"Found {len(validated_filenames)} validated rows.")
    else:
        print("No validated.tsv found. Falling back to all audio files.")

    print("\nScanning archive audio files...")
    with tarfile.open(archive_path, "r:*") as tar:
        audio_members = [member for member in tar if is_audio_member(member)]

    random.shuffle(audio_members)
    print(f"Found {len(audio_members)} audio files in archive.")

    extracted_names: set[str] = set()
    extracted_total = 0

    if validated_targets:
        validated_members = [
            member
            for member in audio_members
            if os.path.basename(member.name) in validated_targets
        ]
        extracted_total += extract_members(
            archive_path=archive_path,
            dest_dir=dest_dir,
            members=validated_members,
            limit=limit,
            extracted_names=extracted_names,
            overwrite=overwrite,
            description="Validated clips",
        )

    remaining = limit - extracted_total
    if remaining > 0:
        extracted_total += extract_members(
            archive_path=archive_path,
            dest_dir=dest_dir,
            members=audio_members,
            limit=remaining,
            extracted_names=extracted_names,
            overwrite=overwrite,
            description="Fallback clips",
        )

    return extracted_total


def main():
    parser = argparse.ArgumentParser(
        description="Extract a sample of audio clips from a local Common Voice .tar.gz."
    )
    parser.add_argument(
        "--archive",
        required=True,
        type=Path,
        help="Path to the local Common Voice .tar.gz archive.",
    )
    parser.add_argument(
        "--language",
        required=True,
        choices=LANGUAGES,
        help="Language folder to populate under data/raw/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of clips to extract. Defaults to 10000.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files with matching names in the target folder.",
    )
    args = parser.parse_args()

    extracted = extract_local_archive(
        archive_path=args.archive,
        language=args.language,
        limit=args.limit,
        overwrite=args.overwrite,
    )

    print("\nDone.")
    print(f"Extracted: {extracted}")
    print(f"Output:    {RAW_DIR / args.language}")

    if extracted < args.limit:
        print(
            f"Warning: requested {args.limit} clips, but only extracted {extracted}."
        )


if __name__ == "__main__":
    main()
"""
Automated data acquisition for Spoken Language Identification.

Downloads Common Voice datasets via the Mozilla Data Collective API,
extracts a balanced subset of clips per language, and cleans up archives
to minimize disk usage.

Usage:
    # Download all configured datasets (smallest first)
    python src/download_data.py --api

    # Download only specific languages
    python src/download_data.py --api --languages amharic russian

    # Just verify what's already on disk
    python src/download_data.py --verify

    # Use a local Common Voice extraction instead of the API
    python "src/download_data copy.py" --cv-root /path/to/cv-corpus --languages english

    # Use a locally downloaded Common Voice archive instead of the API
    python "src/download_data copy.py" --local-archive /path/to/english.tar.gz
"""

import csv
import os
import sys
import shutil
import random
import tarfile
import argparse
import time
from pathlib import Path

import requests
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import RAW_DIR, LANGUAGES, SEED, PROJECT_ROOT

random.seed(SEED)

# ── Constants ────────────────────────────────────────────────────────────

load_dotenv(PROJECT_ROOT / ".env")

API_BASE = "https://mozilladatacollective.com/api"
CLIPS_PER_LANGUAGE = MOZILLA_CLIPS_PER_LANGUAGE
LOCAL_CLIPS_PER_LANGUAGE = MOZILLA_CLIPS_PER_LANGUAGE
MIN_CLIPS_TO_SKIP = 100
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_TIMEOUT = (30, 300)  # connect timeout, read timeout
DOWNLOAD_RETRIES = 10
DOWNLOAD_RETRY_DELAY_SEC = 10

# Dataset IDs for Common Voice Scripted Speech 25.0 (ordered smallest → largest)
DATASET_IDS = {
    "amharic":  "cmn29lq6f0164o10748yd3o7w",   #  58.86 MB
    "hindi":    "cmn2cxzy701iumm077t5ayw0e",   # 544.38 MB
    "arabic":   "cmn2g7uu701fqo1072r5na25l",   #   3.28 GB
    "russian":  "cmn2h1dg201gro107lpynbbd6",   #   6.55 GB
    "italian":  "cmn2h0yei01msmm07u8z5vu87",   #   9.71 GB
    "chinese":  "cmn3iaztg00e4mb070uvufz7q",   #  21.38 GB
    "french":   "cmn5zugst00w3nv07upovf2bg",   #  28.39 GB
    "german":   "cmn4rsdh6009unz07jdn2ol9p",   #  34.69 GB
    "spanish":  "cmn4z1n52000knv07h01532dd",   #  48.23 GB
    "english":  "cmndapwry02jnmh07dyo46mot",   #  87.84 GB
}

LOCALE_MAP = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ru": "russian",
    "am": "amharic",
    "ar": "arabic",
    "zh-CN": "chinese",
}

DATASET_SIZES = {
    "amharic":  "58.86 MB",
    "hindi":    "544.38 MB",
    "arabic":   "3.28 GB",
    "russian":  "6.55 GB",
    "italian":  "9.71 GB",
    "chinese":  "21.38 GB",
    "french":   "28.39 GB",
    "spanish":  "48.23 GB",
    "english":  "87.84 GB",
    "german":   "34.69 GB",
}


# ── API helpers ──────────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.environ.get("MDC_API_KEY", "")
    if not key:
        print("ERROR: MDC_API_KEY not found.")
        print("Create a .env file in the project root with:")
        print("  MDC_API_KEY=your_key_here")
        sys.exit(1)
    return key


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


def _remote_file_size(url: str) -> int | None:
    """Return remote file size when the server exposes Content-Length."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=DOWNLOAD_TIMEOUT)
        if resp.status_code == 200 and resp.headers.get("content-length"):
            return int(resp.headers["content-length"])
    except requests.RequestException:
        pass
    return None


def _download_file(url: str, dest: Path) -> bool:
    """Download a file with retries and resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    remote_size = _remote_file_size(url)
    existing_size = dest.stat().st_size if dest.exists() else 0

    if remote_size and existing_size == remote_size:
        print("  Existing archive is complete; reusing it.")
        return True
    if remote_size and existing_size > remote_size:
        print("  Existing partial archive is larger than expected; restarting.")
        dest.unlink()
        existing_size = 0

    mode = "ab" if existing_size else "wb"
    total = remote_size or existing_size or None

    with tqdm(
        total=total,
        initial=existing_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="  Downloading",
        leave=True,
    ) as bar:
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            downloaded = dest.stat().st_size if dest.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=DOWNLOAD_TIMEOUT,
                )

                if resp.status_code == 416 and remote_size and downloaded == remote_size:
                    return True

                # Some servers ignore Range and return 200. Restart cleanly so the
                # archive is not corrupted by appending duplicate bytes.
                if downloaded and resp.status_code == 200:
                    print("\n  Server did not resume partial download; restarting.")
                    dest.unlink()
                    downloaded = 0
                    mode = "wb"
                    bar.reset(total=remote_size)
                else:
                    resp.raise_for_status()
                    mode = "ab" if downloaded else "wb"

                with open(dest, mode) as f:
                    for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bar.update(len(chunk))

                final_size = dest.stat().st_size
                if remote_size is None or final_size >= remote_size:
                    return True

                print(
                    f"\n  Download stopped early "
                    f"({final_size}/{remote_size} bytes); retrying..."
                )

            except requests.RequestException as e:
                print(f"\n  Download attempt {attempt}/{DOWNLOAD_RETRIES} failed: {e}")

            if attempt < DOWNLOAD_RETRIES:
                time.sleep(DOWNLOAD_RETRY_DELAY_SEC)

    print("  Download failed after retries.")
    return False


# ── Extraction & subsampling ─────────────────────────────────────────────

def _selective_extract(archive_path: Path, dest: Path, limit: int) -> int:
    """Stream through a tar archive and extract only the MP3 clips we need.

    First pass: read validated.tsv files (small) to build a priority list.
    Second pass: extract validated MP3 files first, up to *limit*.
    Third pass: if validated clips are fewer than *limit*, fill the rest with
    other MP3 files from the archive.
    """
    import io

    validated_filenames = []

    # First pass: read all validated.tsv files without extracting to disk.
    print("  Scanning archive for validated clips...")
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            if member.name.endswith("validated.tsv"):
                f = tar.extractfile(member)
                if f is not None:
                    text = io.TextIOWrapper(f, encoding="utf-8")
                    reader = csv.DictReader(text, delimiter="\t")
                    for row in reader:
                        # Normalise to bare filename for reliable matching
                        validated_filenames.append(os.path.basename(row["path"]))

    if validated_filenames:
        random.shuffle(validated_filenames)
        target_files = set(validated_filenames[:limit])
        print(f"  Found {len(validated_filenames)} validated clips, "
              f"targeting {len(target_files)}")
    else:
        target_files = set()
        print("  No validated.tsv found, will extract MP3 files from archive")

    extracted = 0
    extracted_names = set()

    # Second pass: extract validated clips directly to dest.
    if target_files:
        print("  Extracting validated clips...")
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar:
                if extracted >= limit:
                    break

                if not member.name.endswith(".mp3"):
                    continue

                basename = os.path.basename(member.name)
                if basename not in target_files:
                    continue

                member.name = basename
                tar.extract(member, path=dest, filter="data")
                extracted += 1
                extracted_names.add(basename)

                if extracted % 500 == 0:
                    print(f"    {extracted} / {limit} clips extracted...")

    # Third pass: fill remaining quota with any other MP3 clips. This keeps
    # extraction fast while avoiding tiny datasets when validated.tsv is small.
    if extracted < limit:
        remaining = limit - extracted
        print(f"  Filling remaining {remaining} clips from non-validated MP3s...")
        candidates = []
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar:
                if member.name.endswith(".mp3"):
                    basename = os.path.basename(member.name)
                    if basename not in extracted_names:
                        candidates.append(member.name)

        random.shuffle(candidates)
        fallback_files = set(os.path.basename(name) for name in candidates[:remaining])

        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar:
                if extracted >= limit:
                    break

                if not member.name.endswith(".mp3"):
                    continue

                basename = os.path.basename(member.name)
                if basename not in fallback_files or basename in extracted_names:
                    continue

                member.name = basename
                tar.extract(member, path=dest, filter="data")
                extracted += 1
                extracted_names.add(basename)

                if extracted % 500 == 0:
                    print(f"    {extracted} / {limit} clips extracted...")

    return extracted


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

    # Step 2: Download archive to a persistent cache so failed downloads can resume.
    filename = result.get("filename", f"{lang}.tar.gz")
    download_dir = RAW_DIR.parent / "downloads"
    archive_path = download_dir / filename

    print(f"  Downloading {filename}...")
    if not _download_file(download_url, archive_path):
        return False

    # Step 3: Selective extraction -- only pull the clips we need
    dest = RAW_DIR / lang
    dest.mkdir(parents=True, exist_ok=True)

    try:
        copied = _selective_extract(archive_path, dest, CLIPS_PER_LANGUAGE)
    except (tarfile.TarError, EOFError) as e:
        print(f"  [ERROR] Extraction failed: {e}")
        return False

    # Delete archive to free disk space after a successful extraction.
    archive_path.unlink()
    print(f"  Archive deleted. Extracted {copied} clips to {dest}")

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

def _locale_for_language(language: str) -> str | None:
    for locale, lang in LOCALE_MAP.items():
        if lang == language:
            return locale
    return None


def extract_from_local_archive(
    archive_path: Path,
    language: str = "english",
    limit: int = LOCAL_CLIPS_PER_LANGUAGE,
) -> bool:
    """Extract a subset of clips from a locally downloaded Common Voice archive."""
    archive_path = Path(archive_path)
    if not archive_path.exists():
        print(f"[ERROR] Archive not found: {archive_path}")
        return False

    dest = RAW_DIR / language
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LOCAL ARCHIVE -> {language.upper()} ({limit} clips)")
    print(f"{'='*60}")
    print(f"  Source: {archive_path}")
    print(f"  Destination: {dest}")

    try:
        copied = _selective_extract(archive_path, dest, limit)
    except (tarfile.TarError, EOFError) as e:
        print(f"  [ERROR] Extraction failed: {e}")
        return False

    print(f"  Extracted {copied} clips to {dest}")
    return copied > 0


def organise_from_cv_root(
    cv_root: Path,
    languages=None,
    limit: int = LOCAL_CLIPS_PER_LANGUAGE,
):
    """Copy a subset from a locally extracted Common Voice corpus."""
    cv_root = Path(cv_root)
    target_langs = [lang.lower() for lang in languages] if languages else ["english"]

    for lang in target_langs:
        locale = _locale_for_language(lang)
        if locale is None:
            print(f"[SKIP] {lang}: no locale mapping configured")
            continue

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
        copied = _subsample_clips(clips_dir, tsv_path, dest, limit)
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
        "--local-archive", type=str, default=None,
        help="Path to a locally downloaded Common Voice .tar.gz archive",
    )
    parser.add_argument(
        "--local-limit", type=int, default=LOCAL_CLIPS_PER_LANGUAGE,
        help="Number of clips to extract/copy from local data",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Just verify existing data and exit",
    )
    args = parser.parse_args()

    if args.verify:
        verify_data()
    elif args.local_archive:
        lang = args.languages[0].lower() if args.languages else "english"
        extract_from_local_archive(
            Path(args.local_archive),
            language=lang,
            limit=args.local_limit,
        )
        verify_data()
    elif args.cv_root:
        organise_from_cv_root(
            Path(args.cv_root),
            languages=args.languages,
            limit=args.local_limit,
        )
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
