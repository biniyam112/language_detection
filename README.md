# Spoken Language Identification

A PyTorch spoken language identification project that predicts the spoken
language from an audio recording. The main model is a residual CNN over
Mel-spectrograms, with an optional LSTM baseline for comparison.

## Supported Languages

Amharic, Arabic, Chinese, English, French, Hindi, Italian, Russian, Spanish

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify the local dataset folders
python helpers/download_mozilla_data.py --verify

# 3. Optional: precompute deterministic Mel-spectrograms for faster validation/test
# Enable multithreading by using workers
python src/precompute_features.py --workers 4
python src/precompute_features.py --workers 4 --overwrite

# 4. Train a model
python src/train.py --model cnn
python src/train.py --model lstm

# 5. Evaluate the held-out test split
python src/evaluate.py

# 6. Predict one audio file
python src/predict.py --audio path/to/audio.wav

# 7. Launch the live Gradio demo
python app/app.py
```

## Dataset Layout

Training uses up to `10,000` primary Mozilla/local clips plus up to `5,000`
FLEURS clips per language. Amharic can use auxiliary local clips to fill the
primary `10,000`-clip bucket when Mozilla has fewer examples available.

Expected audio layout:

```text
data/raw/
  english/
    common_voice_clip_1.mp3
    fleurs/
      english_fleurs_00001.wav
  amharic/
    common_voice_clip_1.mp3
    alffa_00001.wav
    fleurs/
      amharic_fleurs_00001.wav
```

The source caps are configured in `src/config.py`:

```python
MOZILLA_CLIPS_PER_LANGUAGE = 10_000
FLEURS_CLIPS_PER_LANGUAGE = 5_000
MAX_CLIPS_PER_LANGUAGE = MOZILLA_CLIPS_PER_LANGUAGE + FLEURS_CLIPS_PER_LANGUAGE
```

## Data Helper Scripts

```bash
# Verify existing data counts
python helpers/download_mozilla_data.py --verify

# Download/extract Mozilla data with the Mozilla Data Collective API (This will take a while)
python helpers/download_mozilla_data.py --api

# Download FLEURS training clips into data/raw/{language}/fleurs/
python helpers/download_fleurs.py

# Extract data from a local download .tar.xz file
python helpers/extract_local_data.py --archive path/to/archive.tar.gz --language english

# Top up Amharic with the auxiliary ALFFA dataset since Mozilla Amharic is lacking
python helpers/populate_amharic.py --target 10000
```

## Training And Evaluation

Choose the model to train and evaluate with the `--model` argument:

```bash
python src/train.py --model cnn
python src/evaluate.py --model cnn

python src/train.py --model lstm
python src/evaluate.py --model lstm
```

If `--model` is omitted, `src/train.py` and `src/evaluate.py` use the default
`MODEL_TYPE` value from `src/config.py`.

Checkpoints and training history are saved separately for each model type:

```text
checkpoints/cnn/best_model.pth
checkpoints/cnn/history.json
checkpoints/lstm/best_model.pth
checkpoints/lstm/history.json
```

Evaluation outputs are saved separately by model type, including the
classification report, confusion matrix, PCA embedding plot, and training
curves:

```text
results/cnn/
results/lstm/
```


## Generate Samples for Testing

Use `generate_sample.py` to create sample files under `samples/predict_test/`.
The folder is reset each time the script runs.

```bash
# Samples from the same held-out local test split used by evaluate.py
python generate_sample.py --source local --samples-per-language 5

# Fresh external FLEURS test samples from Hugging Face
python generate_sample.py --source fleurs --samples-per-language 5

# Samples copied from local Mozilla-style files
python generate_sample.py --source mozilla --samples-per-language 5

# Samples copied from local FLEURS training files
python generate_sample.py --source fleurs_local --samples-per-language 5

# Samples copied from auxiliary files, such as Amharic ALFFA padding
python generate_sample.py --source auxiliary --samples-per-language 5
```

After generating samples, run predictions on all files in
`samples/predict_test/`:

```bash
python run_sample_predictions.py
```

This prints per-file predictions, per-language sample accuracy, overall sample
accuracy, and saves detailed scores to `results/sample_predictions.csv`.


## Model Architecture

Default model: 5-block residual 2D CNN over 128-band Mel-spectrograms.

```text
Residual CNN blocks: 1 -> 32 -> 64 -> 128 -> 256 -> 512
Classifier: Linear(512,256) -> Linear(256,128) -> Linear(128,9)
```

Optional comparison model: bidirectional LSTM over Mel-spectrogram time frames.
It uses the same data loading, preprocessing, training, evaluation, and
prediction scripts as the CNN.

## Audio Preprocessing

Training and inference use the same audio duration and feature settings:

```text
Sample rate: 16 kHz
Duration: 5 seconds
Features: 128-bin Mel-spectrograms
```

By default, training uses cached deterministic Mel-spectrograms when available
for speed and Windows DataLoader stability, then applies SpecAugment masks.
Set `TRAIN_CACHE_FEATURES = False` in `src/config.py` if you want live waveform
preprocessing and augmentation during training. Validation, test, sample
prediction, and app inference use deterministic preprocessing.

## Project Structure

```text
language_detection/
  data/raw/{language}/      Audio clips organized by language/source
  data/processed/           Cached Mel-spectrogram tensors
  src/
    config.py               Hyperparameters, paths, language list, model type
    features.py             Audio loading, preprocessing, feature extraction
    dataset.py              Source-aware dataset discovery and DataLoaders
    model.py                CNN and LSTM architectures
    train.py                Training loop
    evaluate.py             Metrics and visualizations
    predict.py              Single-file inference
    precompute_features.py  Feature cache builder
  helpers/                  Download, extraction, and data population scripts
  archived_unused_scripts/  Older scripts kept for reference only
  app/app.py                Gradio live demo
  generate_sample.py        Sample generator for multiple sources
  run_sample_predictions.py Batch prediction for generated samples
  checkpoints/{model_type}/ Saved model weights and training history
  results/                  Evaluation reports, plots, and sample CSV output
```
