# ISL recognition models

Three training pipelines for the local `ISL_DATASET/` (~640 clips, 40 words).  
Designed for **small data + low compute**.

| Model | Folder | Idea | Compute |
|-------|--------|------|---------|
| **1. MediaPipe → Transformer** | `mediapipe_transformer/` | Sample 30/60 frames → Holistic pose+hands+face landmarks → normalize → Transformer | Low–medium (CPU OK after extract) |
| **2. VideoMAE fine-tune** | `videomae_finetune/` | Pretrained video foundation model, freeze backbone, train head | Medium (GPU preferred; batch 2) |
| **3. Landmark TCN (recommended first)** | `landmark_tcn/` | Same landmarks as (1), Temporal CNN instead of Transformer | **Lowest** after extract |

## Why model 3 is TCN

With only hundreds of clips and heavy class imbalance, a full video backbone or a deep Transformer overfits easily. A **Temporal Convolutional Network on MediaPipe landmarks**:

- Reuses the same cheap landmark cache as model 1  
- Far fewer parameters than VideoMAE / Transformer  
- Often stronger than Transformers on short sequential datasets  
- Trains on CPU in minutes once landmarks are cached  

## Setup

```powershell
cd C:\Users\Vidit\OneDrive\Desktop\IPD
python -m pip install -r models/requirements.txt
```

## Shared landmark cache (models 1 & 3)

```powershell
# 30 frames (faster) or 60
python models/mediapipe_transformer/extract_landmarks.py --num-frames 30
```

Caches to `models/_cache/landmarks_T30/`.

Normalization: mid-hip origin, scale by shoulder width (position/scale independence).

## Train

```powershell
# 3) start here (fastest)
python models/landmark_tcn/train.py --num-frames 30 --epochs 50

# 1) landmark Transformer
python models/mediapipe_transformer/train.py --num-frames 30 --epochs 40

# 2) VideoMAE head-only fine-tune (downloads weights once)
python models/videomae_finetune/train.py --epochs 15 --batch-size 2 --freeze-backbone
# full fine-tune (more VRAM/time):
python models/videomae_finetune/train.py --unfreeze --batch-size 1 --epochs 10
```

Checkpoints → `models/_checkpoints/<model_name>/`.

Classes with fewer than `--min-clips` (default 2) are skipped so train/val split is stable. Raise coverage in `ISL_DATASET` for rare words before expecting high accuracy.

## Data notes

- Labels from `ISL_DATASET/metadata.csv`  
- Stratified ~80/20 split per word  
- Class-weighted cross-entropy for imbalance  
- Rare words (1 clip) stay train-only  

## Layout

```
models/
  common/                 # metadata, splits, MediaPipe landmarks
  mediapipe_transformer/  # model 1
  videomae_finetune/      # model 2
  landmark_tcn/           # model 3
  _cache/                 # landmark .npy
  _checkpoints/           # weights + history.json
```
