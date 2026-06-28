# Assignment 4 — Project Status

**Purpose:** Self-study / portfolio · **Updated:** 2026-06-28

## Status: Complete (self-study track)

| Item | Status |
|------|--------|
| Repo vendored into monorepo | Done |
| All unit tests | **21/21 pass** |
| HTML extraction | Done |
| Language ID + PII + classifiers | Done |
| Gopher quality filter | Done |
| Exact line + MinHash dedup | Done |
| `is_english` for WET filtering | Done |
| Writeup | Done — `writeup.md` |
| English WET smoke download (4 files) | Done — 28% English text kept |
| Smoke train on processed WET | Done — 10 steps, loss ~10.78 |

## Quick start

```bash
cd assignments/assignment4-data
uv sync
bash scripts/finalize_assignment.sh
```

## Key files

| Path | Role |
|------|------|
| `cs336_data/` | Data processing library |
| `tests/adapters.py` | Test wiring |
| `scripts/download_data.py` | Offline + Modal data download |
| `local-shared-data/` | Classifiers + sample CC (gitignored, ~2GB) |
