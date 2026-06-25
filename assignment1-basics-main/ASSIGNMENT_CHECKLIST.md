# Assignment 1 Completion Checklist

Updated: 2026-06-25T17:13:53.377154+00:00

## Code & tests
- [x] `uv run pytest -q` — 47 passed, 1 xfail
- [x] TinyStories BPE 10k (`artifacts/bpe/tinystories_10k/`)
- [ ] OWT BPE 32k (`artifacts/bpe/owt_32k/`)
- [ ] BPE experiments (`artifacts/bpe_experiments.json`)

## Training runs
- [x] smoke_test
- [x] tinystories_main — val 2.0679
- [ ] owt_main
- [ ] leaderboard_mod (optional / skipped on fast path)

## Deliverables
- [x] `writeup.md` (export to PDF for Gradescope)
- [x] `cs336-spring2025-assignment-1-submission.zip`
- [x] `artifacts/experiment_log.md`

## After OWT pipeline finishes
```bash
cd /Users/mangeshraut/Downloads/Standford_CS336/assignment1-basics-main
uv run python scripts/finalize_assignment.py
bash make_submission.sh
```

## Gradescope upload
1. `writeup.pdf` — `pandoc writeup.md -o writeup.pdf` or print from Markdown preview
2. `cs336-spring2025-assignment-1-submission.zip`
