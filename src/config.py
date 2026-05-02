import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ── Languages ────────────────────────────────────────────────────────────
LANGUAGES = ["amharic", "arabic", "chinese", "english", "french", "hindi", "italian", "russian", "spanish"]
LANG_TO_IDX = {lang: idx for idx, lang in enumerate(LANGUAGES)}
IDX_TO_LANG = {idx: lang for idx, lang in enumerate(LANGUAGES)}
NUM_CLASSES = len(LANGUAGES)

# ── Audio ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
DURATION_SEC = 5
NUM_SAMPLES = SAMPLE_RATE * DURATION_SEC  # 80 000

# ── Mel-Spectrogram ─────────────────────────────────────────────────────
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

# ── MFCC (secondary features) ───────────────────────────────────────────
N_MFCC = 40

# ── Training ─────────────────────────────────────────────────────────────
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
CACHE_FEATURES = True
FEATURE_CACHE_DIR = PROCESSED_DIR / "mel_spectrograms"

# Early stopping
ES_PATIENCE = 8
ES_MIN_DELTA = 1e-4

# LR scheduler
LR_PATIENCE = 4
LR_FACTOR = 0.5

# ── Data split ───────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Reproducibility ─────────────────────────────────────────────────────
SEED = 42

# ── Ensure directories exist ────────────────────────────────────────────
for d in [RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
for lang in LANGUAGES:
    (RAW_DIR / lang).mkdir(parents=True, exist_ok=True)
    (FEATURE_CACHE_DIR / lang).mkdir(parents=True, exist_ok=True)
