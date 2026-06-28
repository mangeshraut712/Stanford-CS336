# Assignment 3 — Project Status

**Purpose:** Self-study / portfolio · **Updated:** 2026-06-28

## Status: Complete (self-study track)

| Item | Status |
|------|--------|
| Repo vendored into monorepo | Done |
| Local pytest (PostgreSQL) | **7/7 pass** |
| Isoflops + scaling-law fits | Done |
| Parabola fits + loss prediction | Done |
| Writeup | Done — `writeup.md` |
| Verification | `scripts/verify_complete.sh` |
| Hosted GPU experiments | Optional — `A3_API_KEY` |

## Quick start

```bash
brew services start postgresql@15
cd assignments/assignment3-scaling
bash scripts/finalize_assignment.sh
```

## Key files

| Path | Role |
|------|------|
| `writeup.md` | Full analysis |
| `cs336_scaling/analysis/scaling.py` | Scaling-law library |
| `artifacts/` | JSON + SVG outputs |
| `cs336_scaling/client.py` | Hosted API client |

## Optional (enrollment)

Refit on your API runs, then `predict_final_submission.py --submit`.
