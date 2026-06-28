# Assignment 3 — Project Status

**Purpose:** Self-study / portfolio (learning project).  
**Updated:** 2026-06-28

## Status: In progress

| Item | Status |
|------|--------|
| Repo vendored into monorepo | Done |
| `uv sync --extra server` | Done |
| Isoflops analysis script | Done — `scripts/analyze_isoflops.py` |
| API check / probe submit scripts | Done — needs `A3_API_KEY` |
| Local pytest | Blocked until Postgres + `.env` |
| Hosted API experiments | Not started (need student key + API access) |
| Scaling-law fits from your runs | Not started |
| Writeup | Not started |

Unlike Assignments 1–2, **most code is staff-provided**. Your work is:

1. **Run scaling experiments** via the hosted API (`http://hyperturing.stanford.edu:8000`)
2. **Fit scaling laws** (Chinchilla / isoflops) from your runs + `data/isoflops_curves.json`
3. **Predict final loss** for a large validation run and submit via `save_final_submission()`
4. **Writeup** — math and analysis (see `cs336_assignment3_scaling.pdf`)

## Setup

```bash
cd assignments/assignment3-scaling
uv sync --extra server
cp .env.example .env   # edit A3_API_KEY if using hosted API
export A3_API_KEY=06123456   # 8-digit key for hyperturing API
```

## Local tests (optional — staff server code)

Requires **PostgreSQL** running locally (`pg_isready`).

```bash
export INTERNAL_API_KEY=local-dev-internal-key
uv run pytest -q   # 7 tests — API + scheduler
```

On Mac without Postgres, skip pytest; use the hosted API and notebook instead.

## Key files

| Path | Role |
|------|------|
| `cs336_scaling/client.py` | Submit/list experiments on hosted API |
| `examples/client_example.ipynb` | Walkthrough |
| `data/isoflops_curves.json` | Reference isoflops data for analysis |
| `cs336_assignment3_scaling.pdf` | Full handout |

## What this assignment is
