# Live ISL recognition app

Camera → MediaPipe landmarks → `weights/` model → on-screen word + spoken output.

## Setup

```powershell
cd C:\Users\Vidit\OneDrive\Desktop\IPD
python -m pip install -r app/requirements.txt
python -m pip install -r models/requirements.txt
```

Weights expected under `weights/` (already extracted if you have the zips):

- `mediapipe_transformer/model.pt`
- `landmark_tcn/model.pt`
- `labels.json`

First run may download `cache/holistic_landmarker.task` (MediaPipe Tasks model).

Set a Gemini API key to enable automatic English sentence formation after you
lower your hands:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
```

The optional `GEMINI_MODEL` environment variable overrides the default model.

## Run

```powershell
python -m app
```

```powershell
python -m app --model mediapipe_transformer
python -m app --model landmark_tcn
python -m app --camera 1 --conf 0.4
```

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Predict now from buffer |
| `S` | Speak last prediction |
| `M` | Switch model |
| `C` | Clear buffer |
| `Q` / `Esc` | Quit |

Auto-predict when the ~2s buffer is full. Speaks when confidence ≥ `--conf` (default 0.35).

## Modules

```
app/
  camera.py      # webcam + frame buffer
  holistic.py    # MediaPipe Holistic (solutions or Tasks)
  predictor.py   # load weights/ + infer
  display.py     # OpenCV overlay
  speech.py      # pyttsx3 TTS
  pipeline.py    # live loop
  config.py      # paths / defaults
```
