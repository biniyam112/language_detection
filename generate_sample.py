from pathlib import Path
import random

import soundfile as sf
from datasets import load_dataset

OUT = Path("samples/predict_test")
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES_PER_LANGUAGE = 5
SEED = random.SystemRandom().randint(0, 2**32 - 1)

langs = {
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

for lang, code in langs.items():
    print(f"Downloading {lang} samples with seed {SEED}...")
    ds = load_dataset(
        "google/fleurs",
        code,
        split="test",
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=SEED).select(range(SAMPLES_PER_LANGUAGE))

    lang_dir = OUT / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(ds):
        audio = row["audio"]
        path = lang_dir / f"{lang}_{i+1}.wav"
        sf.write(path, audio["array"], audio["sampling_rate"])
        print(" ", path)