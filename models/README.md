# ISL recognition models

Three training pipelines for `ISL_DATASET/` (~640 clips, 40 words), plus an **NVIDIA T4** (~16GB) full pipeline.

| Model | Folder | Idea |
|-------|--------|------|
| **1. MediaPipe → Transformer** | `mediapipe_transformer/` | 30 frames → Holistic landmarks → normalize → Transformer |
| **2. VideoMAE fine-tune** | `videomae_finetune/` | Pretrained video foundation model fine-tuned on ISL |
| **3. Landmark TCN** | `landmark_tcn/` | Same landmarks, Temporal CNN (strong on small data) |

## T4 full pipeline (Linux)

One script: clone GitHub → download HF videos → landmarks → train → test → save weights.

```bash
chmod +x scripts/run_pipeline_t4.sh
./scripts/run_pipeline_t4.sh
```

Optional:

```bash
export WORKDIR=$HOME/isl-run
export HF_TOKEN=hf_xxx          # if needed
export MODELS="landmark_tcn mediapipe_transformer videomae_finetune"
export UNFREEZE_VIDEOMAE=1      # optional; full VideoMAE (tight on 16GB)
./scripts/run_pipeline_t4.sh
```

Weights land in `models/_weights/<model>/`:

- `model.pt` — PyTorch state dict + meta  
- `labels.json` — word ↔ id  
- `history.json` — train/val curves  
- `test_metrics.json` — held-out test accuracy  
- `videomae_finetune/hf/` — HF `save_pretrained` tree  

T4 presets (in `models/train_t4.py`): FP16 AMP, moderate batches (TCN 64 / Transformer 32 / VideoMAE 2), `num_workers=4`, VideoMAE **frozen backbone by default** (`--unfreeze` for full fine-tune).

Manual T4 commands:

```bash
python scripts/download_hf_dataset.py
python models/mediapipe_transformer/extract_landmarks.py --num-frames 30
python models/train_t4.py --models landmark_tcn mediapipe_transformer videomae_finetune
python models/eval_t4.py
```

## Local / low-compute (Windows or laptop)

```powershell
python -m pip install -r models/requirements.txt
python models/mediapipe_transformer/extract_landmarks.py --num-frames 30
python models/landmark_tcn/train.py --num-frames 30
python models/mediapipe_transformer/train.py --num-frames 30
python models/videomae_finetune/train.py --epochs 15 --batch-size 2
```

## Data split

Stratified **train / val / test** (~70/15/15 per class). Best **val** checkpoint is kept; **test** is reported once and stored with weights.

## Layout

```
models/
  common/                  # metadata, splits, landmarks, AMP engine
  train_t4.py              # T4 multi-model trainer
  eval_t4.py               # test-set evaluation
  train_l40s.py            # alias → train_t4
  eval_l40s.py             # alias → eval_t4
  mediapipe_transformer/
  videomae_finetune/
  landmark_tcn/
  _cache/                  # landmark .npy
  _checkpoints/            # intermediate best.pt
  _weights/                # deployable weights + test metrics
scripts/
  run_pipeline_t4.sh       # end-to-end Linux pipeline (T4)
  run_pipeline_l40s.sh     # alias → run_pipeline_t4.sh
  download_hf_dataset.py
```
