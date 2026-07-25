# ISL recognition models

Three training pipelines for `ISL_DATASET/` (~640 clips, 40 words), plus an **NVIDIA L40S** full pipeline.

| Model | Folder | Idea |
|-------|--------|------|
| **1. MediaPipe → Transformer** | `mediapipe_transformer/` | 30/60 frames → Holistic landmarks → normalize → Transformer |
| **2. VideoMAE fine-tune** | `videomae_finetune/` | Pretrained video foundation model fine-tuned on ISL |
| **3. Landmark TCN** | `landmark_tcn/` | Same landmarks, Temporal CNN (strong on small data) |

## L40S full pipeline (Linux)

One script: clone GitHub → download HF videos → landmarks → train → test → save weights.

```bash
chmod +x scripts/run_pipeline_l40s.sh
./scripts/run_pipeline_l40s.sh
```

Optional:

```bash
export WORKDIR=$HOME/isl-run
export HF_TOKEN=hf_xxx          # if needed
export MODELS="landmark_tcn mediapipe_transformer videomae_finetune"
./scripts/run_pipeline_l40s.sh
```

Weights land in `models/_weights/<model>/`:

- `model.pt` — PyTorch state dict + meta  
- `labels.json` — word ↔ id  
- `history.json` — train/val curves  
- `test_metrics.json` — held-out test accuracy  
- `videomae_finetune/hf/` — HF `save_pretrained` tree  

L40S presets (in `models/train_l40s.py`): TF32, AMP, large batches (TCN 256 / Transformer 128 / VideoMAE 16), `num_workers=8`, full VideoMAE fine-tune.

Manual L40S commands:

```bash
python scripts/download_hf_dataset.py
python models/mediapipe_transformer/extract_landmarks.py --num-frames 60
python models/train_l40s.py --models landmark_tcn mediapipe_transformer videomae_finetune
python models/eval_l40s.py
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
  train_l40s.py            # L40S multi-model trainer
  eval_l40s.py             # test-set evaluation
  mediapipe_transformer/
  videomae_finetune/
  landmark_tcn/
  _cache/                  # landmark .npy
  _checkpoints/            # intermediate best.pt
  _weights/                # deployable weights + test metrics
scripts/
  run_pipeline_l40s.sh     # end-to-end Linux pipeline
  download_hf_dataset.py
```
