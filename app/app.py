"""
Gradio web app for live spoken language identification.

Features:
  - Record audio directly from the browser microphone
  - Upload a pre-recorded audio file
  - Displays predicted language with a confidence bar chart

Launch:
    python app/app.py

The app prints a local URL and an optional public share link
(great for demoing on a phone during a class presentation).
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import IDX_TO_LANG, LANGUAGES, MODEL_CHECKPOINT_DIR, SAMPLE_RATE
from model import build_model
from features import extract_mel_spectrogram, prepare_waveform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None


def get_model() -> torch.nn.Module:
    global MODEL
    if MODEL is None:
        ckpt_path = MODEL_CHECKPOINT_DIR / "best_model.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                "No trained model found. Run `python src/train.py` first."
            )
        MODEL = build_model().to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        MODEL.load_state_dict(ckpt["model_state_dict"])
        MODEL.eval()
    return MODEL


@torch.no_grad()
def identify_language(audio):
    """Gradio callback: receives audio tuple (sample_rate, numpy_array)
    and returns a dict of {language: confidence}."""
    if audio is None:
        return {lang.capitalize(): 0.0 for lang in LANGUAGES}

    sr, waveform_np = audio

    # Convert to float32 mono
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=1)
    waveform_np = waveform_np.astype(np.float32)

    # Normalise int16 range to [-1, 1]
    if waveform_np.max() > 1.0 or waveform_np.min() < -1.0:
        waveform_np = waveform_np / (np.abs(waveform_np).max() + 1e-9)

    waveform = torch.from_numpy(waveform_np)

    # Resample if needed
    if sr != SAMPLE_RATE:
        import librosa
        waveform_np = librosa.resample(
            waveform_np,
            orig_sr=sr,
            target_sr=SAMPLE_RATE,
        )
        waveform = torch.from_numpy(waveform_np).float()

    waveform = prepare_waveform(waveform, training=False)
    mel = extract_mel_spectrogram(waveform).unsqueeze(0).to(DEVICE)

    model = get_model()
    logits = model(mel)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    return {lang.capitalize(): float(probs[i]) for i, lang in enumerate(LANGUAGES)}


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Spoken Language Identifier", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Spoken Language Identifier\n"
            "Record or upload an audio clip and the model will predict "
            "which language is being spoken.\n\n"
            "**Supported:** Amharic, Arabic, Chinese, English, French, Hindi, Italian, Russian, Spanish"
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Audio Input",
                    sources=["microphone", "upload"],
                    type="numpy",
                )
                predict_btn = gr.Button("Identify Language", variant="primary")

            with gr.Column(scale=1):
                output_label = gr.Label(
                    label="Prediction",
                    num_top_classes=9,
                )

        predict_btn.click(
            fn=identify_language,
            inputs=audio_input,
            outputs=output_label,
        )

        audio_input.change(
            fn=identify_language,
            inputs=audio_input,
            outputs=output_label,
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=True)
