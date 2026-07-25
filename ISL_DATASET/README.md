---
pretty_name: ISL Isolated Word Dataset (40 words)
license: other
license_name: multi-source-aggregate
license_link: https://huggingface.co/datasets/vidit031/isl-isolated-40words
task_categories:
  - video-classification
language:
  - en
  - sgn
tags:
  - indian-sign-language
  - isl
  - sign-language
  - isolated-signs
  - video
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files: metadata.csv
---

# ISL Isolated Word Dataset (40 words)

Normalized isolated-sign video corpus for Indian Sign Language (ISL), built for Transformer / isolated SLR training.

- **642** H.264 MP4 clips
- **40** target glosses
- Clips resized to height **480**, ~**30 FPS**
- Full per-clip provenance in `metadata.csv`

## Vocabulary

hello, goodbye, thank you, sorry, please, yes, no, help, stop, okay, me, you, he, she, mother, father, brother, sister, friend, teacher, student, home, school, hospital, market, eat, drink, water, food, tea, come, go, sit, stand, read, write, what, where, when, today

## Layout

```
ISL_DATASET/
  README.md
  metadata.csv
  <word_slug>/*.mp4
  <word_slug>/sources.txt
```

`video_path` in `metadata.csv` is relative to the dataset root (forward slashes).

## Source mix (this release)

| Source | Clips | Upstream license (as used here) |
|--------|------:|----------------------------------|
| ISL500 / ISL-DATA | 405 | See [ISL500/ISL-DATA](https://huggingface.co/datasets/ISL500/ISL-DATA) |
| INCLUDE | 143 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) |
| CISLR | 81 | [AFL-3.0](https://huggingface.co/datasets/Exploration-Lab/CISLR) |
| ISLRTC dictionary (data.gov re-encode) | 13 | [MIT](https://huggingface.co/datasets/Vignesh3816/Indian_Sign_Language_Data.gov_Rencoded) (also check data.gov.in terms) |

This Hub dataset is a **derived aggregate**. It is **not** dual-licensed as a single open license. Redistribution of subsets must respect each upstream license. Prefer citing and linking originals when possible.

Some rows use careful label aliases (e.g. INCLUDE `alright` → `okay`, `house` → `home`) and may be marked `review_status=Needs Manual Review` in metadata.

## Load

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
import pandas as pd

ds = load_dataset("vidit031/isl-isolated-40words")
df = ds["train"].to_pandas()  # metadata rows

# download one clip
path = hf_hub_download(
    repo_id="vidit031/isl-isolated-40words",
    repo_type="dataset",
    filename=df.loc[0, "video_path"],
)
```

## Citations

Please cite the upstream datasets you use:

**INCLUDE** — Sridhar et al., INCLUDE (AI4Bharat), Zenodo / CC-BY-4.0  
**CISLR** — Exploration-Lab CISLR (EMNLP 2022)  
**ISL500 / ISL-DATA** — Hugging Face `ISL500/ISL-DATA`  
**ISLRTC dictionary** — ISLRTC / data.gov.in (re-encode on Hub)

## Intended use

Research and education on isolated Indian Sign Language recognition. Not a clinical or accessibility certification dataset.
