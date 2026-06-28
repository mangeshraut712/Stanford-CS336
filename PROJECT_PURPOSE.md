# Project Purpose

**Stanford CS336 — Language Modeling from Scratch**  
Self-study portfolio by [Mangesh Raut](https://github.com/mangeshraut712).

This monorepo documents learning-by-building: implement each assignment from scratch (or analyze scaling laws for A3), pass staff tests, run experiments on Mac where feasible, and publish code + writeups on GitHub.

**Not enrolled for course credit** — no Gradescope submissions; quality bar is passing tests, reproducible scripts, and clear documentation.

## What is here

| Assignment | Topic | Status | Verify |
|------------|-------|--------|--------|
| [1 — Basics](assignments/assignment1-basics/) | BPE, transformer, training | **Complete** | `uv run pytest -q` |
| [2 — Systems](assignments/assignment2-systems/) | Flash attention, DDP, FSDP | **Complete** | `uv run pytest -q` |
| [3 — Scaling](assignments/assignment3-scaling/) | Isoflops, scaling laws | **Complete** (self-study) | `bash scripts/finalize_assignment.sh` |
| 4 — Data | — | Not started | — |
| 5 — Alignment | — | Not started | — |

## Cross-check entire repo

```bash
bash scripts/verify_all.sh
```

## Design principles

1. **Each assignment is a standalone `uv` project** under `assignments/`.
2. **Small artifacts in git** (BPE vocab, run summaries, scaling fits); large data and checkpoints stay local.
3. **Docs per assignment:** `PROJECT_STATUS.md`, `SOLUTION.md`, `LEARNINGS.md`, `writeup.md` where applicable.
4. **Mac-first:** CPU/gloo for distributed tests; CUDA/Triton skipped where noted.

## Links

- Course: https://cs336.stanford.edu
- Repo: https://github.com/mangeshraut712/Stanford-CS336
- Lectures: https://github.com/stanford-cs336/spring2025-lectures
