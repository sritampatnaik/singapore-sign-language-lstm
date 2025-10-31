## Sign Language Recognition (NUS-ISS Sem 2)

A notebook-driven project for recognizing a small set of sign-language actions using MediaPipe keypoints and an LSTM classifier. The workflow covers data collection, preprocessing, training (3 and 6 classes), evaluation, real-time inference, and optional web export.

### Features

- Collect training clips via webcam and organize per class
- Extract MediaPipe keypoints (full body or hands-only)
- Train LSTM models for 3-class and 6-class setups
- Export artifacts (saved models, logs, confusion matrices)
- Run real-time inference from webcam
- Optional: convert model for the web

## Quickstart

### Requirements

- Python 3.10–3.11
- macOS (tested), should work on Linux/Windows with compatible dependencies

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional (only if you plan to export for web):

```bash
pip install tensorflowjs
```

### Project structure

```text
sign-lang/
  MP_Videos_All/                 # Raw videos per class (recorded)
  MP_Videos_All_annotated*/      # Preprocessed keypoints/frames
  MP_Videos_All_keyframes*/      # Keyframes extracted from videos
  artifacts_3_classes/           # Trained 3-class model + reports
  artifacts_6_classes/           # Trained 6-class model + reports
  01_sign_data_collection.ipynb  # Record labeled clips
  02_data_pre_processing.ipynb   # Preprocess (full keypoints)
  02_data_pre_processing_hands_only.ipynb
  03_train_lstm_3_classes.ipynb  # Train 3-class LSTM
  03_train_lstm_6_classes.ipynb  # Train 6-class LSTM
  04_real_time.ipynb             # Live webcam inference
  05_covnert_model_for_web.ipynb # Export model for web
  requirements.txt
```

## Workflow

### 1) Collect data

Open `01_sign_data_collection.ipynb` and run cells to capture webcam clips. Videos will be saved under `MP_Videos_All/<CLASS_NAME>/...mp4`.

### 2) Preprocess

Use either:

- `02_data_pre_processing.ipynb` for full-body keypoints, or
- `02_data_pre_processing_hands_only.ipynb` for hands-only

These notebooks will extract/annotate keypoints and write derived datasets into `MP_Videos_All_annotated*` and/or `MP_Videos_All_keyframes*`.

### 3) Train models

- `03_train_lstm_3_classes.ipynb` trains a 3-class LSTM
- `03_train_lstm_6_classes.ipynb` trains a 6-class LSTM

Artifacts are saved to `artifacts_3_classes/` and `artifacts_6_classes/`, including:

- `action_lstm.keras` and `*.weights.h5`
- `classes.npy`
- Train/val/test confusion matrices (`*.png`, `*.csv`)
- `training_log.csv` with loss/metrics

### 4) Real-time inference

Run `04_real_time.ipynb` to perform live predictions from the webcam using the trained model. Ensure your virtual environment is active and camera access is allowed.

### 5) Optional: Export for web

Run `05_covnert_model_for_web.ipynb` to convert the trained model to a web-friendly format. If using TensorFlow.js, install `tensorflowjs` as shown above.

## Tips & troubleshooting

- Ensure your Python version is within the range in `requirements.txt`.
- If you face camera permission issues on macOS, grant Terminal/IDE camera access in System Settings.
- For best results, record consistent lighting and background during data collection.

## Results

See `artifacts_3_classes/` and `artifacts_6_classes/` for confusion matrices, logs, and saved models. Example files:

- `train_cm.png`, `val_cm.png`, `test_cm.png`
- `train_cm.csv`, `val_cm.csv`, `test_cm.csv`
- `training_log.csv`

## Acknowledgements

- MediaPipe for landmark detection
- TensorFlow/Keras for model training
