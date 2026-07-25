#!/usr/bin/env bash
# Full ISL training pipeline for NVIDIA T4 (~16GB, Linux).
#
# Steps:
#   1) Clone GitHub repo (or use existing checkout)
#   2) Download dataset videos + metadata from Hugging Face
#   3) Install Python deps
#   4) Extract MediaPipe landmarks (cached)
#   5) Train all three models (T4 presets, FP16 AMP)
#   6) Evaluate on held-out test split
#   7) Persist weights under models/_weights/
#
# Usage:
#   chmod +x scripts/run_pipeline_t4.sh
#   ./scripts/run_pipeline_t4.sh
#
# Optional env:
#   REPO_URL=https://github.com/Vidit-01/isl-isolated-40words.git
#   HF_DATASET=vidit031/isl-isolated-40words
#   WORKDIR=$HOME/isl-isolated-40words
#   MODELS="landmark_tcn mediapipe_transformer videomae_finetune"
#   SKIP_CLONE=1          # already inside a checkout
#   SKIP_HF_DOWNLOAD=1    # ISL_DATASET already present
#   SKIP_LANDMARKS=1
#   HF_TOKEN=hf_...       # if dataset/model downloads need auth
#   UNFREEZE_VIDEOMAE=1   # full VideoMAE fine-tune (tight on 16GB)

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Vidit-01/isl-isolated-40words.git}"
HF_DATASET="${HF_DATASET:-vidit031/isl-isolated-40words}"
WORKDIR="${WORKDIR:-$HOME/isl-isolated-40words}"
MODELS="${MODELS:-landmark_tcn mediapipe_transformer videomae_finetune}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_FRAMES_LM="${NUM_FRAMES_LM:-30}"

echo "=== ISL T4 pipeline ==="
echo "WORKDIR=$WORKDIR"
echo "HF_DATASET=$HF_DATASET"
echo "MODELS=$MODELS"
echo "NUM_FRAMES_LM=$NUM_FRAMES_LM"

# ---------- GPU sanity ----------
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found"
fi

# ---------- clone ----------
if [[ "${SKIP_CLONE:-0}" != "1" ]]; then
  if [[ ! -d "$WORKDIR/.git" ]]; then
    echo "=== Cloning $REPO_URL ==="
    git clone "$REPO_URL" "$WORKDIR"
  else
    echo "=== Updating existing clone ==="
    git -C "$WORKDIR" fetch --all --prune
    git -C "$WORKDIR" pull --ff-only || true
  fi
fi
cd "$WORKDIR"

# ---------- python env ----------
echo "=== Python env ==="
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r models/requirements.txt
python -m pip install -r requirements.txt
# ensure HF CLI
python -m pip install -U "huggingface_hub[cli]"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

# ---------- Hugging Face dataset ----------
if [[ "${SKIP_HF_DOWNLOAD:-0}" != "1" ]]; then
  echo "=== Downloading dataset from Hugging Face: $HF_DATASET ==="
  mkdir -p ISL_DATASET
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
  fi
  hf download "$HF_DATASET" \
    --repo-type dataset \
    --local-dir ISL_DATASET \
    --local-dir-use-symlinks False
fi

if [[ ! -f ISL_DATASET/metadata.csv ]]; then
  echo "ERROR: ISL_DATASET/metadata.csv missing after download"
  exit 1
fi

N_MP4=$(find ISL_DATASET -type f -name '*.mp4' | wc -l | tr -d ' ')
echo "Found $N_MP4 mp4 clips"
if [[ "$N_MP4" -lt 10 ]]; then
  echo "ERROR: too few videos downloaded (got $N_MP4). Check HF LFS / auth."
  exit 1
fi

# ---------- landmarks ----------
if [[ "${SKIP_LANDMARKS:-0}" != "1" ]]; then
  echo "=== Extracting MediaPipe landmarks (T=$NUM_FRAMES_LM) ==="
  python models/mediapipe_transformer/extract_landmarks.py --num-frames "$NUM_FRAMES_LM"
fi

# ---------- train ----------
echo "=== Training models (T4 presets) ==="
TRAIN_EXTRA=()
if [[ "${UNFREEZE_VIDEOMAE:-0}" == "1" ]]; then
  TRAIN_EXTRA+=(--unfreeze)
fi
# shellcheck disable=SC2086
python models/train_t4.py --models $MODELS --num-frames "$NUM_FRAMES_LM" "${TRAIN_EXTRA[@]}"

# ---------- eval / confirm test metrics ----------
echo "=== Evaluating on held-out test split ==="
# shellcheck disable=SC2086
python models/eval_t4.py --models $MODELS --num-frames "$NUM_FRAMES_LM"

# ---------- summarize ----------
echo "=== Weights written ==="
find models/_weights -maxdepth 2 -type f | sort
if [[ -f models/_weights/summary.json ]]; then
  echo "--- summary.json ---"
  cat models/_weights/summary.json
fi
if [[ -f models/_weights/eval_summary.json ]]; then
  echo "--- eval_summary.json ---"
  cat models/_weights/eval_summary.json
fi

echo "=== PIPELINE COMPLETE ==="
echo "Deployable weights: $WORKDIR/models/_weights/"
