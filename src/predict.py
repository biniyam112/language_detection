"""
Single-file inference for spoken language identification.

Usage:
    python src/predict.py --audio path/to/audio.wav
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CHECKPOINT_DIR, IDX_TO_LANG, LANGUAGES
from model import LanguageCNN
from features import load_audio, extract_mel_spectrogram


def load_model(device: torch.device) -> LanguageCNN:
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Train the model first with: python src/train.py"
        )
    model = LanguageCNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(audio_path: str, model: LanguageCNN, device: torch.device):
    """Run inference on a single audio file.

    Returns:
        (predicted_language, confidence_dict) where confidence_dict maps
        each language name to its probability.
    """
    waveform = load_audio(audio_path)
    mel = extract_mel_spectrogram(waveform).unsqueeze(0).to(device)  # (1,1,n_mels,T)
    logits = model(mel)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = probs.argmax().item()
    pred_lang = IDX_TO_LANG[pred_idx]

    confidences = {IDX_TO_LANG[i]: probs[i].item() for i in range(len(LANGUAGES))}

    return pred_lang, confidences


def main():
    parser = argparse.ArgumentParser(
        description="Predict the spoken language of an audio file."
    )
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to an audio file (.wav, .mp3, .flac, .ogg)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: file not found — {audio_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    pred_lang, confidences = predict(str(audio_path), model, device)

    print(f"\nPredicted language: {pred_lang.capitalize()}")
    print(f"Confidence: {confidences[pred_lang]:.1%}")
    print("\nAll scores:")
    for lang, prob in sorted(confidences.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"  {lang.capitalize():10s}  {prob:6.1%}  {bar}")


if __name__ == "__main__":
    main()
