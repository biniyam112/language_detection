from pathlib import Path
import argparse
import csv
import sys
from collections import defaultdict

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from src.config import MODEL_TYPE
from src.predict import MODEL_CHOICES, load_model, predict


SAMPLES_DIR = PROJECT_ROOT / "samples" / "predict_test"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch predictions for generated sample audio files."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=MODEL_TYPE,
        help=f"Model architecture to use. Defaults to MODEL_TYPE={MODEL_TYPE!r}.",
    )
    return parser.parse_args()


def find_audio_files(samples_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in samples_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def main():
    args = parse_args()
    model_type = args.model.lower()
    results_path = PROJECT_ROOT / "results" / model_type / "sample_predictions.csv"

    audio_files = find_audio_files(SAMPLES_DIR)
    if not audio_files:
        print(f"No audio files found in {SAMPLES_DIR}")
        print("Run generate_sample.py first, or place audio files under that folder.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, model_type)

    results_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    correct_by_lang = defaultdict(int)
    total_by_lang = defaultdict(int)
    print(f"Model: {model_type}")
    print(f"\nFound {len(audio_files)} sample files\n")
    print(f"{'Expected':12s} {'Predicted':12s} {'Confidence':>10s}  File")
    print("-" * 80)

    for audio_path in audio_files:
        expected = audio_path.parent.name
        predicted, confidences = predict(str(audio_path), model, device)
        confidence = confidences[predicted]
        total_by_lang[expected] += 1
        if predicted == expected:
            correct_by_lang[expected] += 1

        print(
            f"{expected.capitalize():12s} "
            f"{predicted.capitalize():12s} "
            f"{confidence:9.1%}  "
            f"{audio_path.relative_to(PROJECT_ROOT)}"
        )

        row = {
            "file": str(audio_path.relative_to(PROJECT_ROOT)),
            "expected": expected,
            "predicted": predicted,
            "confidence": confidence,
        }
        row.update({f"score_{lang}": score for lang, score in confidences.items()})
        rows.append(row)

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved detailed results to {results_path}")

    total = sum(total_by_lang.values())
    correct = sum(correct_by_lang.values())
    print("\nSample accuracy")
    print("-" * 40)
    for lang in sorted(total_by_lang):
        lang_total = total_by_lang[lang]
        lang_correct = correct_by_lang[lang]
        print(f"{lang.capitalize():12s} {lang_correct:2d}/{lang_total:<2d} "
              f"({lang_correct / lang_total:.1%})")
    print("-" * 40)
    print(f"{'Overall':12s} {correct:2d}/{total:<2d} ({correct / total:.1%})")


if __name__ == "__main__":
    main()
