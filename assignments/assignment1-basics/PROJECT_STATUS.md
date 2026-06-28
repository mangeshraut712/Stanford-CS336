# Assignment 1 — Project Status

**Purpose:** Self-study / portfolio project (not a Gradescope submission).  
Updated: 2026-06-26T06:31:15.885064+00:00

## Code & tests
- [x] `uv run pytest -q` — 47 passed, 1 xfail
- [x] TinyStories BPE 10k (`artifacts/bpe/tinystories_10k/`)
- [x] OWT BPE 32k (`artifacts/bpe/owt_32k/`, 1GB Mac subset)
- [x] BPE experiments (`artifacts/bpe_experiments.json`)

## Training runs
- [x] smoke_test
- [x] tinystories_main — val 2.0679
- [x] owt_main — val 5.3765
- [ ] leaderboard_mod (optional — skipped on Mac fast path)

## Documentation
- [x] `writeup.md` + `writeup.pdf` (learning notes)
- [x] `SOLUTION.md`, `LEARNINGS.md`, `artifacts/experiment_log.md`

## Refresh docs after new runs
```bash
cd /Users/mangeshraut/Downloads/Standford_CS336/assignments/assignment1-basics
bash scripts/finalize_assignment.sh
```

## Next

Assignments 2–3 complete. See repo root [`README.md`](../../README.md) and [`PROJECT_PURPOSE.md`](../../PROJECT_PURPOSE.md).
