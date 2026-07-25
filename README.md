# ISL Dataset Collection Agent

Reproducible pipeline to discover public Indian Sign Language (ISL) video datasets, match a target word list, download/validate/normalize clips, deduplicate, and emit a unified training corpus with full provenance.

## Critical: supply your word list

Edit `config/target_words.json` and set `"_status"` to `"ready"`:

```json
{
  "_status": "ready",
  "words": [
    "hello",
    "thank you",
    "eat",
    "...your remaining words..."
  ]
}
```

The agent **never invents** the vocabulary. Only this file is read.

## Run

```powershell
# optional: python -m pip install -r requirements.txt
# ffmpeg required for H.264 normalization (installed via winget Gyan.FFmpeg)
$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')
python src/isl_dataset_agent.py
```

## Outputs

| Path | Description |
|------|-------------|
| `ISL_DATASET/<word>/` | Normalized MP4 clips + per-video JSON provenance |
| `ISL_DATASET/metadata.csv` | Unified metadata |
| `reports/summary.md` | Coverage & collection summary |
| `reports/dataset_inventory.csv` | All discovered datasets |
| `reports/license_report.csv` | License / redistribution |
| `reports/missing_words.csv` | Words below 40-video target |
| `reports/duplicate_report.csv` | Removed duplicates |
| `reports/normalization_map.csv` | Label mapping |
| `logs/agent_actions.log` | Full action log |

## Discovered sources (inventory)

INCLUDE (Zenodo CC-BY-4.0), CISLR (HF), iSign/ISLTranslate, ISL500, ISLRTC data.gov dictionary (re-encode), ISL-CSLTR (Mendeley), ISL-50/52 (Zenodo/Drive), IIITA-ROBITA (restricted).

See `reports/dataset_inventory.csv` after the first run.
