# Project purpose

**Stanford CS336 — Language Modeling from Scratch** (Spring 2026)  
Self-study portfolio by [Mangesh Raut](https://github.com/mangeshraut712).

This monorepo is learning-by-building: implement each assignment (or analyze scaling laws for A3), pass staff tests, run what is feasible on Mac, and keep code plus writeups on GitHub.

**Not enrolled for course credit** — no Gradescope submissions. The quality bar is passing local tests, reproducible scripts, and clear documentation. This is not a product and not official Stanford staff materials.

## What is here

| Assignment | Topic | Status | Verify |
|------------|-------|--------|--------|
| [1 — Basics](../assignments/assignment1-basics/) | BPE, transformer, training | **Complete** | `pytest` + `scripts/verify_complete.sh` |
| [2 — Systems](../assignments/assignment2-systems/) | Flash attention, DDP, FSDP | **Complete** | `uv run pytest -q` |
| [3 — Scaling](../assignments/assignment3-scaling/) | Isoflops, scaling laws | **Complete** (self-study) | `bash scripts/finalize_assignment.sh` |
| [4 — Data](../assignments/assignment4-data/) | Filtering, dedup, classifiers | **Complete** (self-study) | `bash scripts/finalize_assignment.sh` |
| [5 — Alignment](../assignments/assignment5-alignment/) | GRPO, SFT, DPO, eval parsers | **Complete** (self-study) | `bash scripts/finalize_assignment.sh` |

## Cross-check entire repo

```bash
bash scripts/verify_all.sh
```

See [`LAYOUT.md`](LAYOUT.md) for the directory tree and doc conventions.

## Design principles

1. **Each assignment is a standalone `uv` project** under `assignments/`.
2. **Small artifacts in git** (BPE vocab, run summaries, scaling fits); large data and checkpoints stay local.
3. **Docs per assignment:** `PROJECT_STATUS.md`, `SOLUTION.md`, `LEARNINGS.md`, `writeup.md` where applicable.
4. **Mac-first:** CPU/gloo for distributed tests; CUDA/Triton skipped where noted.

## Links

- Course: https://cs336.stanford.edu/ (Spring 2026)
- Repo: https://github.com/mangeshraut712/Stanford-CS336
- Lectures: https://github.com/stanford-cs336/lectures
- Previous lectures: https://github.com/stanford-cs336/spring2025-lectures
