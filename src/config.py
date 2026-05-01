import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ── Languages ────────────────────────────────────────────────────────────
LANGUAGES = ["italian", "russian", "amharic", "hindi"]
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
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Early stopping
ES_PATIENCE = 5
ES_MIN_DELTA = 1e-4

# LR scheduler
LR_PATIENCE = 3
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
for lang in LANGUAGES:
    (RAW_DIR / lang).mkdir(parents=True, exist_ok=True)
