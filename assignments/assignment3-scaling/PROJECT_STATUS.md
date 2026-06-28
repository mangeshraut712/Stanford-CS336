# Assignment 3 — Project Status

**Purpose:** Self-study / portfolio (learning project).  
**Updated:** 2026-06-28

## Status: In progress

| Item | Status |
|------|--------|
| Repo vendored into monorepo | Done |
| `uv sync --extra server` | Done |
| Local pytest (PostgreSQL) | **7/7 pass** |
| Isoflops + scaling-law fits | Done |
| API check / probe submit | Optional — needs enrolled `A3_API_KEY` |
| Hosted GPU experiments | Optional — enrollment only |
| Writeup | Not started |

Unlike Assignments 1–2, **most code is staff-provided**. Your work is:

1. **Run scaling experiments** via the hosted API (`http://hyperturing.stanford.edu:8000`)
2. **Fit scaling laws** (Chinchilla / isoflops) from your runs + `data/isoflops_curves.json`
3. **Predict final loss** for a large validation run and submit via `save_final_submission()`
4. **Writeup** — math and analysis (see `cs336_assignment3_scaling.pdf`)

## Local workflow (recommended for self-study)

```bash
brew services start postgresql@15
cp .env.example .env
uv sync --extra server
uv run pytest -q
uv run python scripts/analyze_isoflops.py
uv run python scripts/fit_scaling_laws.py
```

See `GOALS.md` for assignment goal and when you need `A3_API_KEY`.

## Key files

| Path | Role |
|------|------|
| `cs336_scaling/client.py` | Submit/list experiments on hosted API |
| `examples/client_example.ipynb` | Walkthrough |
| `data/isoflops_curves.json` | Reference isoflops data for analysis |
| `cs336_assignment3_scaling.pdf` | Full handout |

## What this assignment is
