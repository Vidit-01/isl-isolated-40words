# ISL Dataset Build — Status

**Updated:** 2026-07-25 afternoon — **NOT DONE** (0/40 words at ≥40 clips yet)

## Current unified corpus

| Metric | Value |
|--------|-------|
| Accepted videos | **588** |
| Words at ≥40 | **0** |
| Closest | thank you **39**, school **37**, hello **36**, market **36** |
| Sources | ISL500 405, INCLUDE 102, CISLR 81 |

Still downloading INCLUDE: People / Jobs / Days / remaining Pronouns+Places parts.  
Also pulling ISL-50 (Drive) for sparse words.

```powershell
python src/finalize_local.py
python src/download_include_parallel.py
```


## How to re-run

```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')
$env:PYTHONUNBUFFERED=1
# After more raw downloads:
python src/finalize_local.py
# Or full agent (will download missing INCLUDE categories):
python src/isl_dataset_agent.py
# Priority INCLUDE categories (Greetings, Pronouns, People, Jobs, Places, Days_and_Time):
python src/download_include_priority.py
```

## Path to ≥40 videos / word

INCLUDE metadata shows **406 exact-match videos** across 16 target words once these Zenodo categories are fully downloaded:

| Category | Helps words |
|----------|-------------|
| Greetings | hello (~42), thank you (~42) |
| Pronouns | he, she, you (~21 each) |
| People | brother (~42), father (~40), mother, sister, friend |
| Jobs | teacher, student |
| Places | school, hospital, market |
| Days_and_Time | today |

Combined with ISL500 (~15/word when present) + CISLR (1–6), several words should exceed 40.

## In progress

1. `Greetings_2of2.zip` download from Zenodo (resumable curl)  
2. Final rebuild including INCLUDE `48. Hello` (21 clips from part 1)  
3. Remaining INCLUDE priority categories (~10–15 GB total)

## Manual review (not auto-merged)

See `reports/manual_review_candidates.csv` — e.g. `me`↔`I`, `home`↔`house`, `hello`↔`namaste`.

## Licenses

See `reports/license_report.csv`. INCLUDE is CC-BY-4.0. CISLR/iSign research use. ISL500 per HF card. Do not redistribute restricted corpora.
