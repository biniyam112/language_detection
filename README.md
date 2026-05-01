# Spoken Language Identification

A CNN-based deep learning model that identifies the spoken language from audio
recordings. Built with PyTorch for a machine learning course project.

## Supported Languages

English, Spanish, French, German, Italian, Russian, Amharic

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (see instructions printed by the script)
python src/download_data.py

# 3. Train the model
python src/train.py

# 4. Evaluate on the test set
python src/evaluate.py

# 5. Predict a single file
python src/predict.py --audio path/to/audio.wav

# 6. Launch the live demo app
python app/app.py
```

## Project Structure

```
language_detection/
  data/raw/{language}/      Audio clips organized by language
  data/processed/           Cached feature arrays
  src/
    config.py               Hyperparameters and paths
    features.py             Audio loading and feature extraction
    dataset.py              PyTorch Dataset with augmentation
    model.py                CNN architecture
    train.py                Training loop
    evaluate.py             Metrics and visualizations
    predict.py              Single-file inference
  app/app.py                Gradio live demo
  checkpoints/              Saved model weights
  notebooks/                Exploratory analysis
```

## Model Architecture

4-layer 2D CNN over Mel-spectrograms (128 mel bands):

```
Conv2d(1,32)  -> BatchNorm -> ReLU -> MaxPool
Conv2d(32,64) -> BatchNorm -> ReLU -> MaxPool
Conv2d(64,128) -> BatchNorm -> ReLU -> MaxPool
Conv2d(128,256) -> BatchNorm -> ReLU -> AdaptiveAvgPool
Flatten -> Linear(256,128) -> ReLU -> Dropout -> Linear(128,7)
```

## Dataset

[Mozilla Common Voice](https://commonvoice.mozilla.org/) — open-source,
crowd-sourced speech corpus with recordings in dozens of languages.
